"""Core rf-ingress runtime wiring adapters, clock alignment, solver, gate, and publisher.

This runtime owns no radio-specific logic and imports no concrete adapters. It
consumes adapters strictly through :class:`RangeAdapter`,
:class:`PositionAdapter`, and :class:`RoomAdapter`, performs per-epoch solving
without cross-epoch smoothing, enforces the registration/binding gate, and
publishes validated hardware-neutral fixes to :class:`TagFixPublisher`.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import structlog
from cts_contracts.wearables import FixKind

from rf_ingress.adapters.base import (
    AdapterHealth,
    AnyAdapter,
    PositionAdapter,
    RangeAdapter,
    RoomAdapter,
)
from rf_ingress.clock import (
    AdapterClockSkewEstimator,
    BurstClockOffsetTracker,
    resolve_report_timestamp,
)
from rf_ingress.domain import (
    EmittedFix,
    PositionReport,
    RangeEpoch,
    RangeMeasurement,
    RoomReport,
    SiteConfigSnapshot,
    ensure_utc,
)
from rf_ingress.observability import (
    ANCHOR_OUTLIERS_TOTAL,
    FIXES_DROPPED_TOTAL,
    REPORTS_TOTAL,
    SOLVE_SECONDS,
    DropReason,
    ReportKind,
)
from rf_ingress.publish import TagFixContractError, TagFixPublisher
from rf_ingress.registration import RegistrationGate
from rf_ingress.site_config import SiteConfigProvider
from rf_ingress.solver.epoch import EpochBuilder, room_fallback_from_epoch
from rf_ingress.solver.frame_transform import SimilarityTransform2D
from rf_ingress.solver.multilateration import MultilaterationConfig, MultilaterationSolver

logger = structlog.get_logger(__name__)

_IDENTITY_TRANSFORM = SimilarityTransform2D(
    scale=1.0,
    rotation_rad=0.0,
    tx_m=0.0,
    ty_m=0.0,
    rms_residual_m=0.0,
    point_count=0,
)


@dataclass(frozen=True)
class RuntimeHealthSnapshot:
    """Snapshot exposed by ``GET /health``."""

    adapters: tuple[AdapterHealth, ...]
    site_config_loaded: bool
    site_config_version: str
    site_config_floor_plan_id: str | None
    site_config_age_s: float | None
    node_count: int
    device_count: int
    binding_count: int
    publisher_healthy: bool
    node_last_seen_age_s: dict[str, float]


class RfIngressRuntime:
    """Orchestrates site-config polling, adapter ingestion, solving, and publishing."""

    def __init__(
        self,
        *,
        adapters: Sequence[AnyAdapter],
        site_config: SiteConfigProvider,
        publisher: TagFixPublisher,
        solver_config: MultilaterationConfig | None = None,
        epoch_window_s: float = 0.25,
        site_config_poll_interval_s: float = 60.0,
        capability_heartbeat_interval_s: float = 60.0,
        skew_window_s: float = 60.0,
        skew_warn_threshold_s: float = 0.5,
        burst_window_size: int = 20,
        frame_transforms: Mapping[str, SimilarityTransform2D] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._adapters: tuple[AnyAdapter, ...] = tuple(adapters)
        self._site_config: SiteConfigProvider = site_config
        self._publisher: TagFixPublisher = publisher
        self._epoch_builder: EpochBuilder = EpochBuilder(epoch_window_s=epoch_window_s)
        self._solver: MultilaterationSolver = MultilaterationSolver(solver_config)
        self._gate: RegistrationGate = RegistrationGate()
        self._burst_clock: BurstClockOffsetTracker = BurstClockOffsetTracker(
            window_size=burst_window_size
        )
        self._skew_estimator: AdapterClockSkewEstimator = AdapterClockSkewEstimator(
            window_s=skew_window_s,
            warn_threshold_s=skew_warn_threshold_s,
        )
        self._frame_transforms: dict[str, SimilarityTransform2D] = dict(frame_transforms or {})
        self._poll_interval_s: float = site_config_poll_interval_s
        self._heartbeat_interval_s: float = capability_heartbeat_interval_s
        self._clock: Callable[[], datetime] = clock
        self._tasks: list[asyncio.Task[None]] = []
        self._stop_event: asyncio.Event = asyncio.Event()
        self._node_last_seen_at: dict[str, datetime] = {}
        self._position_seq_by_key: dict[tuple[str, str], int] = {}
        self._room_seq_by_key: dict[tuple[str, str], int] = {}
        self._last_applied_version: str = ""

    @property
    def burst_clock(self) -> BurstClockOffsetTracker:
        return self._burst_clock

    @property
    def skew_estimator(self) -> AdapterClockSkewEstimator:
        return self._skew_estimator

    def _sync_epoch_builder_with_snapshot(self, snapshot: SiteConfigSnapshot | None) -> None:
        if snapshot is None:
            return
        if snapshot.version == self._last_applied_version and self._last_applied_version != "":
            return
        nodes_by_adapter: dict[str, set[str]] = {}
        for node in snapshot.nodes:
            nodes_by_adapter.setdefault(node.adapter_id, set()).add(node.anchor_id)
        frozen_map = {aid: frozenset(ids) for aid, ids in nodes_by_adapter.items()}
        dropped = self._epoch_builder.on_site_config_version(
            snapshot.version,
            nodes_by_adapter=frozen_map,
        )
        self._last_applied_version = snapshot.version
        if dropped > 0:
            FIXES_DROPPED_TOTAL.labels(reason=DropReason.SITE_CONFIG_RELOADED).inc(dropped)
            logger.info(
                "rf_epochs_dropped_on_site_config_reload",
                dropped_epochs=dropped,
                version=snapshot.version,
            )

    def _next_seq(
        self,
        store: dict[tuple[str, str], int],
        adapter_id: str,
        tag_id: str,
        explicit_seq: int | None,
    ) -> int:
        key = (adapter_id, tag_id)
        if explicit_seq is not None:
            store[key] = max(store.get(key, 0), explicit_seq + 1)
            return explicit_seq
        seq = store.get(key, 1)
        store[key] = seq + 1
        return seq

    async def handle_range_measurement(
        self,
        adapter: AnyAdapter,
        measurement: RangeMeasurement,
        *,
        arrival_at: datetime | None = None,
    ) -> list[EmittedFix]:
        """Ingest one range report, update clock stats, and publish any completed epochs."""
        arrival_utc = ensure_utc(arrival_at if arrival_at is not None else self._clock())
        REPORTS_TOTAL.labels(adapter_id=adapter.adapter_id, type=ReportKind.RANGE.value).inc()
        self._node_last_seen_at[measurement.anchor_id] = arrival_utc

        snapshot = self._site_config.snapshot()
        self._sync_epoch_builder_with_snapshot(snapshot)
        is_registered = (
            snapshot is not None and snapshot.device_by_tag_id(measurement.tag_id) is not None
        )

        # Measure adapter skew against sent_at for batched traffic or measured_at otherwise.
        device_clock_sample = (
            measurement.sent_at if measurement.sent_at is not None else measurement.measured_at
        )
        self._skew_estimator.observe(
            adapter.adapter_id,
            device_time=device_clock_sample,
            arrival_time=arrival_utc,
        )

        corrected_ts = resolve_report_timestamp(
            measured_at=measurement.measured_at,
            arrival_at=arrival_utc,
            ntp_synced=adapter.capability.ntp_synced,
            latency_s=adapter.capability.latency_s,
            tag_id=measurement.tag_id,
            sent_at=measurement.sent_at,
            burst_tracker=self._burst_clock,
            record_tag_metric=is_registered,
        )
        aligned = replace(
            measurement,
            measured_at=corrected_ts,
            adapter_id=adapter.adapter_id,
        )

        completed_epochs = self._epoch_builder.add(aligned)
        published: list[EmittedFix] = []
        for epoch in completed_epochs:
            fix = await self._process_epoch(adapter, epoch, snapshot)
            if fix is not None:
                published.append(fix)
        return published

    async def _process_epoch(
        self,
        adapter: AnyAdapter,
        epoch: RangeEpoch,
        snapshot: SiteConfigSnapshot | None,
    ) -> EmittedFix | None:
        if snapshot is None:
            FIXES_DROPPED_TOTAL.labels(reason=DropReason.NO_SITE_CONFIG).inc()
            return None

        nodes = snapshot.node_map()
        known_count = sum(1 for m in epoch.measurements if m.anchor_id in nodes)

        start = time.perf_counter()
        result = self._solver.solve(epoch, nodes)
        SOLVE_SECONDS.observe(time.perf_counter() - start)

        if result is not None:
            if result.dropped_outlier_anchor_id is not None:
                ANCHOR_OUTLIERS_TOTAL.labels(anchor_id=result.dropped_outlier_anchor_id).inc()
            fix = EmittedFix(
                tag_id=epoch.tag_id,
                measured_at=epoch.measured_at,
                kind=FixKind.POSITION,
                epoch_seq=epoch.epoch_seq,
                adapter_id=adapter.adapter_id,
                adapter_kind=adapter.capability.adapter_kind,
                adapter_version=adapter.capability.adapter_version,
                floor_plan_id=snapshot.floor_plan_id or "",
                x_mm=round(result.x_m * 1000.0),
                y_mm=round(result.y_m * 1000.0),
                z_mm=round(result.z_m * 1000.0) if result.z_m is not None else None,
                cov_xx_m2=result.cov_xx_m2,
                cov_xy_m2=result.cov_xy_m2,
                cov_yy_m2=result.cov_yy_m2,
                wear_state=epoch.wear_state,
                motion_state=epoch.motion_state,
                quality=result.quality,
                anchors_used=result.anchors_used,
                residual_rms_m=result.residual_rms_m,
                battery_fraction=epoch.battery_fraction,
            )
            return await self._gate_and_publish(fix, snapshot)

        # Core room fallback when fewer than 3 nodes contributed ranges.
        if known_count < 3:
            fallback = room_fallback_from_epoch(epoch, nodes)
            if fallback is not None:
                return await self._emit_room_report(adapter, fallback, snapshot)

        FIXES_DROPPED_TOTAL.labels(reason=DropReason.SOLVER_REJECTED).inc()
        return None

    async def handle_position_report(
        self,
        adapter: AnyAdapter,
        report: PositionReport,
        *,
        arrival_at: datetime | None = None,
    ) -> EmittedFix | None:
        """Transform a native-frame position report and publish it if eligible."""
        arrival_utc = ensure_utc(arrival_at if arrival_at is not None else self._clock())
        REPORTS_TOTAL.labels(adapter_id=adapter.adapter_id, type=ReportKind.POSITION.value).inc()

        snapshot = self._site_config.snapshot()
        self._sync_epoch_builder_with_snapshot(snapshot)
        is_registered = (
            snapshot is not None and snapshot.device_by_tag_id(report.tag_id) is not None
        )

        device_clock_sample = report.sent_at if report.sent_at is not None else report.measured_at
        self._skew_estimator.observe(
            adapter.adapter_id,
            device_time=device_clock_sample,
            arrival_time=arrival_utc,
        )
        corrected_ts = resolve_report_timestamp(
            measured_at=report.measured_at,
            arrival_at=arrival_utc,
            ntp_synced=adapter.capability.ntp_synced,
            latency_s=adapter.capability.latency_s,
            tag_id=report.tag_id,
            sent_at=report.sent_at,
            burst_tracker=self._burst_clock,
            record_tag_metric=is_registered,
        )

        transform = self._frame_transforms.get(adapter.adapter_id, _IDENTITY_TRANSFORM)
        x_m, y_m = transform.transform_point(report.x_native_m, report.y_native_m)
        if report.cov_native_m2 is None:
            FIXES_DROPPED_TOTAL.labels(reason=DropReason.INVALID_CONTRACT).inc()
            return None
        cov_xx, cov_xy, cov_yy = transform.transform_covariance(*report.cov_native_m2)
        det = cov_xx * cov_yy - cov_xy * cov_xy
        if (
            not (math.isfinite(cov_xx) and math.isfinite(cov_xy) and math.isfinite(cov_yy))
            or cov_xx <= 0.0
            or cov_yy <= 0.0
            or det <= 0.0
        ):
            FIXES_DROPPED_TOTAL.labels(reason=DropReason.INVALID_CONTRACT).inc()
            return None

        sigma_m = math.sqrt(0.5 * (cov_xx + cov_yy))
        quality = max(0.0, min(1.0, math.exp(-sigma_m / 0.5)))
        seq = self._next_seq(
            self._position_seq_by_key, adapter.adapter_id, report.tag_id, report.epoch_seq
        )

        fix = EmittedFix(
            tag_id=report.tag_id,
            measured_at=corrected_ts,
            kind=FixKind.POSITION,
            epoch_seq=seq,
            adapter_id=adapter.adapter_id,
            adapter_kind=adapter.capability.adapter_kind,
            adapter_version=adapter.capability.adapter_version,
            floor_plan_id=snapshot.floor_plan_id or "" if snapshot is not None else "",
            x_mm=round(x_m * 1000.0),
            y_mm=round(y_m * 1000.0),
            z_mm=round(report.z_native_m * 1000.0) if report.z_native_m is not None else None,
            cov_xx_m2=cov_xx,
            cov_xy_m2=cov_xy,
            cov_yy_m2=cov_yy,
            wear_state=report.wear_state,
            motion_state=report.motion_state,
            quality=quality,
            residual_rms_m=transform.rms_residual_m if transform.point_count >= 2 else None,
            battery_fraction=report.battery_fraction,
        )
        return await self._gate_and_publish(fix, snapshot)

    async def handle_room_report(
        self,
        adapter: AnyAdapter,
        report: RoomReport,
        *,
        arrival_at: datetime | None = None,
    ) -> EmittedFix | None:
        """Process a direct room presence report and publish it if eligible."""
        arrival_utc = ensure_utc(arrival_at if arrival_at is not None else self._clock())
        REPORTS_TOTAL.labels(adapter_id=adapter.adapter_id, type=ReportKind.ROOM.value).inc()

        snapshot = self._site_config.snapshot()
        self._sync_epoch_builder_with_snapshot(snapshot)
        is_registered = (
            snapshot is not None and snapshot.device_by_tag_id(report.tag_id) is not None
        )

        device_clock_sample = report.sent_at if report.sent_at is not None else report.measured_at
        self._skew_estimator.observe(
            adapter.adapter_id,
            device_time=device_clock_sample,
            arrival_time=arrival_utc,
        )
        corrected_ts = resolve_report_timestamp(
            measured_at=report.measured_at,
            arrival_at=arrival_utc,
            ntp_synced=adapter.capability.ntp_synced,
            latency_s=adapter.capability.latency_s,
            tag_id=report.tag_id,
            sent_at=report.sent_at,
            burst_tracker=self._burst_clock,
            record_tag_metric=is_registered,
        )
        aligned = replace(report, measured_at=corrected_ts, adapter_id=adapter.adapter_id)
        return await self._emit_room_report(adapter, aligned, snapshot)

    async def _emit_room_report(
        self,
        adapter: AnyAdapter,
        report: RoomReport,
        snapshot: SiteConfigSnapshot | None,
    ) -> EmittedFix | None:
        seq = self._next_seq(
            self._room_seq_by_key, adapter.adapter_id, report.tag_id, report.epoch_seq
        )
        fix = EmittedFix(
            tag_id=report.tag_id,
            measured_at=report.measured_at,
            kind=FixKind.ROOM,
            epoch_seq=seq,
            adapter_id=adapter.adapter_id,
            adapter_kind=adapter.capability.adapter_kind,
            adapter_version=adapter.capability.adapter_version,
            floor_plan_id=snapshot.floor_plan_id or "" if snapshot is not None else "",
            room_id=report.room_id,
            room_probability=report.probability,
            anchor_id=report.anchor_id,
            distance_m=report.distance_m,
            wear_state=report.wear_state,
            motion_state=report.motion_state,
            anchors_used=report.anchors_used,
            battery_fraction=report.battery_fraction,
        )
        return await self._gate_and_publish(fix, snapshot)

    async def _gate_and_publish(
        self,
        fix: EmittedFix,
        snapshot: SiteConfigSnapshot | None,
    ) -> EmittedFix | None:
        decision = self._gate.check(
            snapshot=snapshot,
            tag_id=fix.tag_id,
            adapter_id=fix.adapter_id,
            measured_at=fix.measured_at,
        )
        if not decision.allowed:
            return None
        try:
            await self._publisher.publish(fix)
        except TagFixContractError as exc:
            FIXES_DROPPED_TOTAL.labels(reason=DropReason.INVALID_CONTRACT).inc()
            logger.warning(
                "rf_fix_contract_rejected",
                adapter_id=fix.adapter_id,
                error=str(exc),
            )
            return None
        logger.debug(
            "rf_fix_published",
            adapter_id=fix.adapter_id,
            tag_id=fix.tag_id,
            kind=fix.kind.value,
            epoch_seq=fix.epoch_seq,
        )
        return fix

    async def publish_capabilities_once(self) -> int:
        """Publish one capability heartbeat per configured adapter."""
        now = ensure_utc(self._clock())
        count = 0
        for adapter in self._adapters:
            await self._publisher.publish_capability(adapter.capability, reported_at=now)
            count += 1
        return count

    async def _drain_adapter(self, adapter: AnyAdapter) -> list[EmittedFix]:
        emitted: list[EmittedFix] = []
        if isinstance(adapter, RangeAdapter):
            async for meas in adapter.ranges():
                if self._stop_event.is_set():
                    break
                emitted.extend(await self.handle_range_measurement(adapter, meas))
        if isinstance(adapter, PositionAdapter):
            async for pos in adapter.positions():
                if self._stop_event.is_set():
                    break
                fix = await self.handle_position_report(adapter, pos)
                if fix is not None:
                    emitted.append(fix)
        if isinstance(adapter, RoomAdapter):
            async for rm in adapter.rooms():
                if self._stop_event.is_set():
                    break
                fix = await self.handle_room_report(adapter, rm)
                if fix is not None:
                    emitted.append(fix)
        return emitted

    async def run_once(self) -> list[EmittedFix]:
        """Refresh site config, emit capabilities, drain all adapters, and flush epochs."""
        snapshot = await self._site_config.refresh()
        self._sync_epoch_builder_with_snapshot(snapshot)
        await self.publish_capabilities_once()

        published: list[EmittedFix] = []
        adapters_by_id = {a.adapter_id: a for a in self._adapters}
        for adapter in self._adapters:
            published.extend(await self._drain_adapter(adapter))

        snapshot = self._site_config.snapshot()
        for epoch in self._epoch_builder.flush_all():
            owner = adapters_by_id.get(epoch.adapter_id)
            if owner is not None:
                fix = await self._process_epoch(owner, epoch, snapshot)
                if fix is not None:
                    published.append(fix)
        return published

    async def _site_config_loop(self) -> None:
        while not self._stop_event.is_set():
            snapshot = await self._site_config.refresh()
            self._sync_epoch_builder_with_snapshot(snapshot)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval_s)
            except TimeoutError:
                continue

    async def _capability_loop(self) -> None:
        while not self._stop_event.is_set():
            await self.publish_capabilities_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._heartbeat_interval_s)
            except TimeoutError:
                continue

    async def _adapter_loop(self, adapter: AnyAdapter) -> None:
        await self._drain_adapter(adapter)

    async def start(self) -> None:
        """Start background polling, heartbeat, and adapter ingestion tasks."""
        self._stop_event.clear()
        snapshot = await self._site_config.refresh()
        self._sync_epoch_builder_with_snapshot(snapshot)
        self._tasks = [
            asyncio.create_task(self._site_config_loop(), name="rf-site-config-poll"),
            asyncio.create_task(self._capability_loop(), name="rf-capability-heartbeat"),
            *[
                asyncio.create_task(
                    self._adapter_loop(adapter), name=f"rf-adapter-{adapter.adapter_id}"
                )
                for adapter in self._adapters
            ],
        ]

    async def stop(self) -> None:
        """Cancel background tasks cleanly and close adapters, site config, and publisher."""
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        for adapter in self._adapters:
            await adapter.close()
        await self._site_config.close()
        await self._publisher.close()

    def health_snapshot(self) -> RuntimeHealthSnapshot:
        """Build a point-in-time health snapshot for ``GET /health``."""
        now = ensure_utc(self._clock())
        snapshot = self._site_config.snapshot()
        synced_at = self._site_config.last_synced_at()
        config_age_s = (
            max(0.0, (now - ensure_utc(synced_at)).total_seconds())
            if synced_at is not None
            else None
        )
        node_ages = {
            anchor_id: max(0.0, (now - ensure_utc(ts)).total_seconds())
            for anchor_id, ts in sorted(self._node_last_seen_at.items())
        }
        return RuntimeHealthSnapshot(
            adapters=tuple(adapter.health() for adapter in self._adapters),
            site_config_loaded=snapshot is not None,
            site_config_version=snapshot.version if snapshot is not None else "",
            site_config_floor_plan_id=snapshot.floor_plan_id if snapshot is not None else None,
            site_config_age_s=config_age_s,
            node_count=len(snapshot.nodes) if snapshot is not None else 0,
            device_count=len(snapshot.devices) if snapshot is not None else 0,
            binding_count=len(snapshot.bindings) if snapshot is not None else 0,
            publisher_healthy=self._publisher.is_healthy(),
            node_last_seen_age_s=node_ages,
        )
