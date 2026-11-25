"""Deterministic seeded simulator and JSONL replay adapter for rf-ingress.

Covers all three adapter protocols (:class:`RangeAdapter`,
:class:`PositionAdapter`, :class:`RoomAdapter`) and models every ranging
technology class (``uwb``, ``ble_channel_sounding``, ``ble_rssi`` with
body-shadow steps and adjacent-room receivers), burst-batched delivery with
``sent_at`` timestamps, wear-state and motion-state schedules, battery
discharge, NLOS bias regions, dropouts, and JSONL raw-report replay.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
import yaml
from cts_contracts.wearables import (
    AdapterKind,
    DeviceKind,
    FixKind,
    MotionState,
    RangingTechnology,
    RfNodeRole,
    WearState,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rf_ingress.adapters.base import AdapterDeps, AdapterHealth
from rf_ingress.config import AdapterConfig
from rf_ingress.domain import (
    AdapterCapabilityDescriptor,
    AnonymousPositionTrackReport,
    AzimuthReferenceFrame,
    PositionReport,
    PositionTrackFrame,
    RangeMeasurement,
    RfNodeGeometry,
    RoomReport,
    SiteBinding,
    SiteConfigSnapshot,
    SiteDevice,
    ensure_utc,
)
from rf_ingress.observability import (
    FIXES_DROPPED_TOTAL,
    POSITION_TRACKS_DROPPED_TOTAL,
    UNREGISTERED_TAG_TOTAL,
    DropReason,
    PositionTrackDropReason,
)

logger = structlog.get_logger(__name__)

SIMULATOR_ADAPTER_VERSION = "0.1.0"
SIMULATOR_PROFILES: frozenset[str] = frozenset({"scenario_yaml_v1", "replay_jsonl_v1"})

# Technology-class baseline 1-sigma models in metres.
_UWB_DEFAULT_SIGMA_M = 0.10
_CS_DEFAULT_SIGMA_M = 0.50
_BLE_RSSI_MIN_SIGMA_M = 1.0
_BLE_RSSI_RANGE_FRACTION = 0.50
# Indoor log-distance path-loss exponent used to map a 5-10 dB body-shadow
# step into an equivalent RSSI range inflation factor: 10^(dB / (10 * n)).
_BLE_PATH_LOSS_EXPONENT = 2.5


class WaypointSpec(BaseModel):
    """One scripted 3D waypoint along a tag's trajectory (floor-frame metres)."""

    model_config = ConfigDict(extra="forbid")

    t_s: float = Field(ge=0.0)
    x_m: float
    y_m: float
    z_m: float = 1.0


class WearWindowSpec(BaseModel):
    """Scheduled wear state over ``[start_s, end_s)``."""

    model_config = ConfigDict(extra="forbid")

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    state: WearState


class MotionWindowSpec(BaseModel):
    """Scheduled motion state over ``[start_s, end_s)``."""

    model_config = ConfigDict(extra="forbid")

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    state: MotionState


class ActiveWindowSpec(BaseModel):
    """Optional transmission window ``[start_s, end_s)`` for burst-mode tags."""

    model_config = ConfigDict(extra="forbid")

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)


class NlosBiasSpec(BaseModel):
    """Optional systematic NLOS range bias for one anchor inside a floor box."""

    model_config = ConfigDict(extra="forbid")

    anchor_id: str = Field(min_length=1)
    bias_m: float
    x_min_m: float = -1e6
    y_min_m: float = -1e6
    x_max_m: float = 1e6
    y_max_m: float = 1e6


class ScenarioNodeSpec(BaseModel):
    """RF node declared in a scenario file (used for standalone site-config fixtures)."""

    model_config = ConfigDict(extra="forbid")

    anchor_id: str = Field(min_length=1)
    adapter_id: str = "sim-1"
    node_role: RfNodeRole = RfNodeRole.RANGING_ANCHOR
    technology: RangingTechnology = RangingTechnology.UWB
    x_m: float
    y_m: float
    z_m: float = 2.2
    room_id: str | None = None


class ScenarioRoomSpec(BaseModel):
    """Axis-aligned room rectangle in floor-plan metres for simulated room reports."""

    model_config = ConfigDict(extra="forbid")

    room_id: str = Field(min_length=1)
    x_min_m: float
    y_min_m: float
    x_max_m: float
    y_max_m: float


class ScenarioSurveyTransformSpec(BaseModel):
    """Native-to-floor similarity transform parameters for position-report scenarios."""

    model_config = ConfigDict(extra="forbid")

    scale: float = Field(default=1.0, gt=0.0)
    rotation_deg: float = 0.0
    tx_m: float = 0.0
    ty_m: float = 0.0


class ScenarioTagSpec(BaseModel):
    """Scripted wearable tag emitter inside a deterministic scenario."""

    model_config = ConfigDict(extra="forbid")

    tag_id: str = Field(min_length=1)
    member_id: str | None = None
    device_kind: DeviceKind = DeviceKind.BLE_BEACON
    report_protocol: Literal["range", "position", "room"] = "range"
    technology: RangingTechnology = RangingTechnology.UWB
    sigma_m: float | None = Field(default=None, gt=0.0)
    body_shadow_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    body_shadow_db_min: float = Field(default=5.0, ge=0.0)
    body_shadow_db_max: float = Field(default=10.0, ge=0.0)
    dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    clock_skew_s: float = 0.0
    batch_interval_s: float | None = Field(default=None, gt=0.0)
    batch_jitter_min_s: float = Field(default=0.05, ge=0.0)
    batch_jitter_max_s: float = Field(default=0.25, ge=0.0)
    max_range_m: float | None = Field(default=None, gt=0.0)
    max_nodes_per_epoch: int | None = Field(default=None, ge=1)
    anchor_ids: list[str] | None = None
    emit_device_local_azimuth: bool = False
    waypoints: list[WaypointSpec] = Field(min_length=1)
    wear_schedule: list[WearWindowSpec] = Field(default_factory=list)
    motion_schedule: list[MotionWindowSpec] = Field(default_factory=list)
    active_windows: list[ActiveWindowSpec] = Field(default_factory=list)
    nlos_biases: list[NlosBiasSpec] = Field(default_factory=list)
    battery_start: float | None = Field(default=None, ge=0.0, le=1.0)
    battery_end: float | None = Field(default=None, ge=0.0, le=1.0)
    position_sigma_m: float = Field(default=0.25, gt=0.0)


class SimulatorScenario(BaseModel):
    """Complete deterministic simulator scenario loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    seed: int = 42
    start_time: datetime = Field(
        default_factory=lambda: ensure_utc(datetime.fromisoformat("2026-11-19T12:00:00+00:00"))
    )
    duration_s: float = Field(gt=0.0)
    step_s: float = Field(default=1.0, gt=0.0)
    realtime: bool = False
    floor_plan_id: str = "household:1"
    nodes: list[ScenarioNodeSpec] = Field(default_factory=list)
    rooms: list[ScenarioRoomSpec] = Field(default_factory=list)
    survey_transform: ScenarioSurveyTransformSpec | None = None
    tags: list[ScenarioTagSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_start_time_utc(self) -> SimulatorScenario:
        self.start_time = ensure_utc(self.start_time)
        return self


class SimulatorAdapterConfig(BaseModel):
    """Adapter-specific configuration validated when ``kind == "simulator"``."""

    model_config = ConfigDict(extra="forbid")

    payload_profile: str = "scenario_yaml_v1"
    scenario: str | None = None
    replay_path: str | None = None
    realtime: bool | None = None
    reports_wear_state: bool | None = None
    reports_motion_state: bool | None = None

    @model_validator(mode="after")
    def _validate_profile_and_source(self) -> SimulatorAdapterConfig:
        if self.payload_profile not in SIMULATOR_PROFILES:
            known = ", ".join(sorted(SIMULATOR_PROFILES))
            raise ValueError(
                f"Unknown simulator payload_profile {self.payload_profile!r}; "
                f"expected one of: {known}"
            )
        if self.payload_profile == "scenario_yaml_v1" and not self.scenario:
            raise ValueError("Simulator profile 'scenario_yaml_v1' requires 'scenario' path")
        if self.payload_profile == "replay_jsonl_v1" and not self.replay_path:
            raise ValueError("Simulator profile 'replay_jsonl_v1' requires 'replay_path'")
        return self


class _ReplayRangePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_type: Literal["range"] = "range"
    tag_id: str = Field(min_length=1)
    anchor_id: str = Field(min_length=1)
    distance_m: float = Field(ge=0.0)
    variance_m2: float = Field(gt=0.0)
    technology: RangingTechnology
    measured_at: datetime
    los_confidence: float | None = None
    azimuth_deg: float | None = None
    azimuth_variance_deg2: float | None = None
    elevation_deg: float | None = None
    azimuth_reference_frame: AzimuthReferenceFrame | None = None
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    epoch_seq: int | None = None
    sent_at: datetime | None = None


class _ReplayPositionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_type: Literal["position"] = "position"
    tag_id: str = Field(min_length=1)
    measured_at: datetime
    x_native_m: float
    y_native_m: float
    z_native_m: float | None = None
    cov_native_m2: tuple[float, float, float] | None = None
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    epoch_seq: int | None = None
    sent_at: datetime | None = None


class _ReplayRoomPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_type: Literal["room"] = "room"
    tag_id: str = Field(min_length=1)
    measured_at: datetime
    room_id: str = Field(min_length=1)
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    epoch_seq: int | None = None
    sent_at: datetime | None = None
    anchor_id: str = ""
    distance_m: float | None = Field(default=None, ge=0.0)
    anchors_used: int = Field(default=0, ge=0)


def load_scenario(path: Path | str) -> SimulatorScenario:
    """Load and validate a YAML simulator scenario."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario file {path} must contain a YAML mapping")
    return SimulatorScenario.model_validate(raw)


def build_site_config_from_scenario(
    scenario: SimulatorScenario,
    *,
    adapter_id: str = "sim-1",
    version: str = "sim:v1",
) -> SiteConfigSnapshot:
    """Build a :class:`SiteConfigSnapshot` fixture from a scenario's nodes and tags."""
    nodes = tuple(
        RfNodeGeometry(
            anchor_id=n.anchor_id,
            x_m=n.x_m,
            y_m=n.y_m,
            z_m=n.z_m,
            adapter_id=n.adapter_id or adapter_id,
            node_role=n.node_role,
            technology=n.technology,
            room_id=n.room_id,
        )
        for n in scenario.nodes
    )
    devices: list[SiteDevice] = []
    bindings: list[SiteBinding] = []
    seen_tags: set[str] = set()
    for tag in scenario.tags:
        if tag.tag_id in seen_tags:
            continue
        seen_tags.add(tag.tag_id)
        devices.append(
            SiteDevice(
                tag_id=tag.tag_id,
                adapter_id=adapter_id,
                hardware_key=f"sim:{tag.tag_id}",
                device_kind=tag.device_kind,
                is_active=True,
            )
        )
        if tag.member_id:
            bindings.append(
                SiteBinding(
                    tag_id=tag.tag_id,
                    member_id=tag.member_id,
                    valid_from=scenario.start_time - timedelta(hours=1),
                    valid_to=None,
                )
            )
    return SiteConfigSnapshot(
        floor_plan_id=scenario.floor_plan_id,
        nodes=nodes,
        bindings=tuple(bindings),
        devices=tuple(devices),
        version=version,
    )


def _interpolate_waypoint(
    waypoints: list[WaypointSpec],
    t_s: float,
) -> tuple[float, float, float, float]:
    """Return ``(x_m, y_m, z_m, speed_m_s)`` at elapsed time *t_s*."""
    ordered = sorted(waypoints, key=lambda w: w.t_s)
    if t_s <= ordered[0].t_s or len(ordered) == 1:
        w = ordered[0]
        return (w.x_m, w.y_m, w.z_m, 0.0)
    if t_s >= ordered[-1].t_s:
        w = ordered[-1]
        return (w.x_m, w.y_m, w.z_m, 0.0)
    for prev_w, next_w in itertools.pairwise(ordered):
        if prev_w.t_s <= t_s <= next_w.t_s:
            span = max(next_w.t_s - prev_w.t_s, 1e-9)
            frac = (t_s - prev_w.t_s) / span
            x = prev_w.x_m + frac * (next_w.x_m - prev_w.x_m)
            y = prev_w.y_m + frac * (next_w.y_m - prev_w.y_m)
            z = prev_w.z_m + frac * (next_w.z_m - prev_w.z_m)
            dist = math.hypot(next_w.x_m - prev_w.x_m, next_w.y_m - prev_w.y_m)
            return (x, y, z, dist / span)
    last = ordered[-1]
    return (last.x_m, last.y_m, last.z_m, 0.0)


def _schedule_wear_state(schedule: list[WearWindowSpec], t_s: float) -> WearState:
    for window in schedule:
        if window.start_s <= t_s < window.end_s:
            return window.state
    return WearState.UNSPECIFIED


def _schedule_motion_state(
    schedule: list[MotionWindowSpec],
    t_s: float,
    speed_m_s: float,
    *,
    infer_from_speed: bool,
) -> MotionState:
    for window in schedule:
        if window.start_s <= t_s < window.end_s:
            return window.state
    if infer_from_speed:
        return MotionState.MOVING if speed_m_s > 0.05 else MotionState.STILL
    return MotionState.UNSPECIFIED


def _is_active_at(windows: list[ActiveWindowSpec], t_s: float) -> bool:
    if not windows:
        return True
    return any(w.start_s <= t_s < w.end_s for w in windows)


def _range_sigma_m(tag: ScenarioTagSpec, true_distance_m: float) -> float:
    if tag.sigma_m is not None:
        return tag.sigma_m
    if tag.technology == RangingTechnology.UWB:
        return _UWB_DEFAULT_SIGMA_M
    if tag.technology == RangingTechnology.BLE_CHANNEL_SOUNDING:
        return _CS_DEFAULT_SIGMA_M
    return max(_BLE_RSSI_MIN_SIGMA_M, _BLE_RSSI_RANGE_FRACTION * true_distance_m)


def _room_for_point(rooms: list[ScenarioRoomSpec], x_m: float, y_m: float) -> str | None:
    for room in rooms:
        if room.x_min_m <= x_m <= room.x_max_m and room.y_min_m <= y_m <= room.y_max_m:
            return room.room_id
    return None


def _nlos_bias_for(tag: ScenarioTagSpec, anchor_id: str, x_m: float, y_m: float) -> float:
    bias = 0.0
    for spec in tag.nlos_biases:
        if (
            spec.anchor_id == anchor_id
            and spec.x_min_m <= x_m <= spec.x_max_m
            and spec.y_min_m <= y_m <= spec.y_max_m
        ):
            bias += spec.bias_m
    return bias


def _floor_to_native(
    x_m: float,
    y_m: float,
    tf: ScenarioSurveyTransformSpec | None,
) -> tuple[float, float]:
    if tf is None:
        return (x_m, y_m)
    theta = math.radians(tf.rotation_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    dx = (x_m - tf.tx_m) / tf.scale
    dy = (y_m - tf.ty_m) / tf.scale
    x_native = c * dx + s * dy
    y_native = -s * dx + c * dy
    return (x_native, y_native)


class SimulatorAdapter:
    """Deterministic seeded simulator and JSONL replay adapter.

    Implements :class:`RangeAdapter`, :class:`PositionAdapter`, and
    :class:`RoomAdapter` behind one consistent capability descriptor.
    """

    def __init__(
        self,
        *,
        adapter_id: str,
        deps: AdapterDeps,
        scenario: SimulatorScenario | None = None,
        replay_lines: Iterable[str] | None = None,
        replay_path: Path | None = None,
        realtime: bool = False,
        ntp_synced: bool = True,
        latency_s: float = 0.0,
        reports_wear_state_override: bool | None = None,
        reports_motion_state_override: bool | None = None,
    ) -> None:
        if scenario is None and replay_lines is None and replay_path is None:
            raise ValueError("SimulatorAdapter requires scenario, replay_lines, or replay_path")
        self.adapter_id: str = adapter_id
        self._deps: AdapterDeps = deps
        self._scenario: SimulatorScenario | None = scenario
        self._replay_lines: list[str] | None = (
            list(replay_lines) if replay_lines is not None else None
        )
        self._replay_path: Path | None = replay_path
        self._realtime: bool = realtime
        self._closed: bool = False
        self._last_report_at: datetime | None = None
        self._malformed_reports: int = 0
        self._unregistered_reports: int = 0

        self.capability: AdapterCapabilityDescriptor = self._build_capability(
            ntp_synced=ntp_synced,
            latency_s=latency_s,
            reports_wear_state_override=reports_wear_state_override,
            reports_motion_state_override=reports_motion_state_override,
        )

    def _build_capability(
        self,
        *,
        ntp_synced: bool,
        latency_s: float,
        reports_wear_state_override: bool | None,
        reports_motion_state_override: bool | None,
    ) -> AdapterCapabilityDescriptor:
        if self._scenario is not None:
            protocols = {t.report_protocol for t in self._scenario.tags}
            fix_kinds: list[FixKind] = []
            if "range" in protocols or "position" in protocols:
                fix_kinds.append(FixKind.POSITION)
            if "room" in protocols or "range" in protocols:
                # Ranges with < 3 nodes trigger core room fallback.
                fix_kinds.append(FixKind.ROOM)
            if not fix_kinds:
                fix_kinds.append(FixKind.POSITION)

            reports_wear = (
                reports_wear_state_override
                if reports_wear_state_override is not None
                else any(len(t.wear_schedule) > 0 for t in self._scenario.tags)
            )
            reports_motion = (
                reports_motion_state_override
                if reports_motion_state_override is not None
                else any(len(t.motion_schedule) > 0 for t in self._scenario.tags)
            )
            reports_z = any(t.report_protocol in ("range", "position") for t in self._scenario.tags)
            return AdapterCapabilityDescriptor(
                adapter_id=self.adapter_id,
                adapter_kind=AdapterKind.SIMULATOR,
                adapter_version=SIMULATOR_ADAPTER_VERSION,
                fix_kinds=tuple(fix_kinds),
                nominal_rate_hz=1.0 / self._scenario.step_s,
                reports_covariance=FixKind.POSITION in fix_kinds,
                reports_wear_state=reports_wear,
                reports_z=reports_z,
                reports_motion_state=reports_motion,
                ntp_synced=ntp_synced,
                latency_s=latency_s,
            )

        # Replay mode: inspect available lines without raising on malformed ones.
        parsed_ranges, parsed_positions, parsed_rooms = self._parse_replay_source(
            record_malformed=False
        )
        replay_kinds: list[FixKind] = []
        if parsed_ranges or parsed_positions:
            replay_kinds.append(FixKind.POSITION)
        if parsed_rooms or parsed_ranges:
            replay_kinds.append(FixKind.ROOM)
        if not replay_kinds:
            replay_kinds.append(FixKind.POSITION)

        all_wear = (
            [r.wear_state for r in parsed_ranges]
            + [p.wear_state for p in parsed_positions]
            + [rm.wear_state for rm in parsed_rooms]
        )
        all_motion = (
            [r.motion_state for r in parsed_ranges]
            + [p.motion_state for p in parsed_positions]
            + [rm.motion_state for rm in parsed_rooms]
        )
        return AdapterCapabilityDescriptor(
            adapter_id=self.adapter_id,
            adapter_kind=AdapterKind.SIMULATOR,
            adapter_version=SIMULATOR_ADAPTER_VERSION,
            fix_kinds=tuple(replay_kinds),
            nominal_rate_hz=1.0,
            reports_covariance=FixKind.POSITION in replay_kinds,
            reports_wear_state=(
                reports_wear_state_override
                if reports_wear_state_override is not None
                else any(w != WearState.UNSPECIFIED for w in all_wear)
            ),
            reports_z=any(p.z_native_m is not None for p in parsed_positions)
            or bool(parsed_ranges),
            reports_motion_state=(
                reports_motion_state_override
                if reports_motion_state_override is not None
                else any(m != MotionState.UNSPECIFIED for m in all_motion)
            ),
            ntp_synced=ntp_synced,
            latency_s=latency_s,
        )

    def health(self) -> AdapterHealth:
        return AdapterHealth(
            adapter_id=self.adapter_id,
            connected=not self._closed,
            last_report_at=self._last_report_at,
            malformed_reports=self._malformed_reports,
            unregistered_reports=self._unregistered_reports,
        )

    async def close(self) -> None:
        self._closed = True

    def _is_tag_registered(self, tag_id: str) -> bool:
        snapshot = self._deps.site_config.snapshot()
        if snapshot is None:
            return True
        if snapshot.device_by_tag_id(tag_id) is not None:
            return True
        self._unregistered_reports += 1
        UNREGISTERED_TAG_TOTAL.labels(adapter_id=self.adapter_id).inc()
        FIXES_DROPPED_TOTAL.labels(reason=DropReason.UNREGISTERED_TAG).inc()
        return False

    def _active_nodes(self) -> list[RfNodeGeometry]:
        snapshot = self._deps.site_config.snapshot()
        if snapshot is not None and snapshot.nodes:
            return list(snapshot.nodes)
        if self._scenario is not None and self._scenario.nodes:
            return [
                RfNodeGeometry(
                    anchor_id=n.anchor_id,
                    x_m=n.x_m,
                    y_m=n.y_m,
                    z_m=n.z_m,
                    adapter_id=n.adapter_id or self.adapter_id,
                    node_role=n.node_role,
                    technology=n.technology,
                    room_id=n.room_id,
                )
                for n in self._scenario.nodes
            ]
        return []

    def _read_replay_raw_lines(self) -> list[str]:
        if self._replay_lines is not None:
            return list(self._replay_lines)
        if self._replay_path is not None:
            return self._replay_path.read_text(encoding="utf-8").splitlines()
        return []

    def _parse_replay_source(
        self,
        *,
        record_malformed: bool,
    ) -> tuple[list[RangeMeasurement], list[PositionReport], list[RoomReport]]:
        ranges: list[RangeMeasurement] = []
        positions: list[PositionReport] = []
        rooms: list[RoomReport] = []
        for raw_line in self._read_replay_raw_lines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    raise ValueError("JSONL line must decode to an object")
                report_type = obj.get("report_type")
                if report_type == "range":
                    rp = _ReplayRangePayload.model_validate(obj)
                    ranges.append(
                        RangeMeasurement(
                            tag_id=rp.tag_id,
                            anchor_id=rp.anchor_id,
                            distance_m=rp.distance_m,
                            variance_m2=rp.variance_m2,
                            technology=rp.technology,
                            measured_at=ensure_utc(rp.measured_at),
                            los_confidence=rp.los_confidence,
                            azimuth_deg=rp.azimuth_deg,
                            azimuth_variance_deg2=rp.azimuth_variance_deg2,
                            elevation_deg=rp.elevation_deg,
                            azimuth_reference_frame=rp.azimuth_reference_frame,
                            wear_state=rp.wear_state,
                            motion_state=rp.motion_state,
                            battery_fraction=rp.battery_fraction,
                            adapter_id=self.adapter_id,
                            epoch_seq=rp.epoch_seq,
                            sent_at=ensure_utc(rp.sent_at) if rp.sent_at is not None else None,
                        )
                    )
                elif report_type == "position":
                    pp = _ReplayPositionPayload.model_validate(obj)
                    positions.append(
                        PositionReport(
                            tag_id=pp.tag_id,
                            measured_at=ensure_utc(pp.measured_at),
                            x_native_m=pp.x_native_m,
                            y_native_m=pp.y_native_m,
                            z_native_m=pp.z_native_m,
                            cov_native_m2=pp.cov_native_m2,
                            wear_state=pp.wear_state,
                            motion_state=pp.motion_state,
                            battery_fraction=pp.battery_fraction,
                            adapter_id=self.adapter_id,
                            epoch_seq=pp.epoch_seq,
                            sent_at=ensure_utc(pp.sent_at) if pp.sent_at is not None else None,
                        )
                    )
                elif report_type == "room":
                    rmp = _ReplayRoomPayload.model_validate(obj)
                    rooms.append(
                        RoomReport(
                            tag_id=rmp.tag_id,
                            measured_at=ensure_utc(rmp.measured_at),
                            room_id=rmp.room_id,
                            probability=rmp.probability,
                            wear_state=rmp.wear_state,
                            motion_state=rmp.motion_state,
                            battery_fraction=rmp.battery_fraction,
                            adapter_id=self.adapter_id,
                            epoch_seq=rmp.epoch_seq,
                            sent_at=ensure_utc(rmp.sent_at) if rmp.sent_at is not None else None,
                            anchor_id=rmp.anchor_id,
                            distance_m=rmp.distance_m,
                            anchors_used=rmp.anchors_used,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported report_type {report_type!r}")
            except (json.JSONDecodeError, ValidationError, ValueError):
                if record_malformed:
                    self._malformed_reports += 1
                    FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT).inc()
                    logger.warning(
                        "rf_simulator_malformed_replay_line",
                        adapter_id=self.adapter_id,
                    )
        return ranges, positions, rooms

    def _generate_scenario_ranges(self) -> list[RangeMeasurement]:
        assert self._scenario is not None
        scenario = self._scenario
        rng = np.random.default_rng(scenario.seed)
        nodes = self._active_nodes()
        if not nodes:
            return []

        n_steps = math.floor(scenario.duration_s / scenario.step_s) + 1
        out: list[RangeMeasurement] = []

        for tag_idx, tag in enumerate(scenario.tags):
            if tag.report_protocol != "range":
                continue

            eligible_nodes = [
                n
                for n in nodes
                if (tag.anchor_ids is None or n.anchor_id in tag.anchor_ids)
                and (n.technology == tag.technology or tag.anchor_ids is not None)
            ]
            if not eligible_nodes:
                eligible_nodes = list(nodes)

            # Collect epochs so batched senders can stamp one shared sent_at per batch.
            pending_batch: list[RangeMeasurement] = []
            batch_start_s: float | None = None
            epoch_counter = 0

            for step_idx in range(n_steps):
                t_s = step_idx * scenario.step_s
                if t_s > scenario.duration_s + 1e-9:
                    break
                if not _is_active_at(tag.active_windows, t_s):
                    continue

                epoch_counter += 1
                x_m, y_m, z_m, speed_m_s = _interpolate_waypoint(tag.waypoints, t_s)
                measured_at = scenario.start_time + timedelta(seconds=t_s + tag.clock_skew_s)
                wear_state = _schedule_wear_state(tag.wear_schedule, t_s)
                motion_state = _schedule_motion_state(
                    tag.motion_schedule,
                    t_s,
                    speed_m_s,
                    infer_from_speed=False,
                )
                battery: float | None = None
                if tag.battery_start is not None:
                    end_batt = tag.battery_end if tag.battery_end is not None else tag.battery_start
                    frac = min(1.0, max(0.0, t_s / max(scenario.duration_s, 1e-9)))
                    battery = max(
                        0.0, min(1.0, tag.battery_start + frac * (end_batt - tag.battery_start))
                    )

                step_candidates: list[tuple[float, RfNodeGeometry]] = []
                for node in eligible_nodes:
                    true_d = math.sqrt(
                        (x_m - node.x_m) ** 2 + (y_m - node.y_m) ** 2 + (z_m - node.z_m) ** 2
                    )
                    if tag.max_range_m is not None and true_d > tag.max_range_m:
                        continue
                    step_candidates.append((true_d, node))

                step_candidates.sort(key=lambda item: (item[0], item[1].anchor_id))
                if tag.max_nodes_per_epoch is not None:
                    step_candidates = step_candidates[: tag.max_nodes_per_epoch]

                epoch_reports: list[RangeMeasurement] = []
                for true_d, node in step_candidates:
                    if tag.dropout_prob > 0.0 and float(rng.random()) < tag.dropout_prob:
                        continue
                    sigma_m = _range_sigma_m(tag, true_d)
                    noise = float(rng.normal(0.0, sigma_m))
                    nlos = _nlos_bias_for(tag, node.anchor_id, x_m, y_m)

                    shadow_mult = 1.0
                    if tag.body_shadow_prob > 0.0 and float(rng.random()) < tag.body_shadow_prob:
                        shadow_db = float(
                            rng.uniform(tag.body_shadow_db_min, tag.body_shadow_db_max)
                        )
                        shadow_mult = 10.0 ** (shadow_db / (10.0 * _BLE_PATH_LOSS_EXPONENT))

                    measured_d = max(0.05, true_d * shadow_mult + nlos + noise)
                    az_deg: float | None = None
                    az_frame: AzimuthReferenceFrame | None = None
                    if tag.emit_device_local_azimuth:
                        # Arbitrary wrist rotation angle in device_local frame.
                        az_deg = float(((step_idx * 47 + tag_idx * 29) % 360) - 180)
                        az_frame = "device_local"

                    epoch_reports.append(
                        RangeMeasurement(
                            tag_id=tag.tag_id,
                            anchor_id=node.anchor_id,
                            distance_m=measured_d,
                            variance_m2=sigma_m * sigma_m,
                            technology=tag.technology,
                            measured_at=measured_at,
                            azimuth_deg=az_deg,
                            azimuth_reference_frame=az_frame,
                            wear_state=wear_state,
                            motion_state=motion_state,
                            battery_fraction=battery,
                            adapter_id=self.adapter_id,
                            epoch_seq=epoch_counter,
                        )
                    )

                if tag.batch_interval_s is None:
                    out.extend(epoch_reports)
                else:
                    if batch_start_s is None:
                        batch_start_s = t_s
                    pending_batch.extend(epoch_reports)
                    if (t_s - batch_start_s) + scenario.step_s >= tag.batch_interval_s - 1e-9:
                        sent_at = scenario.start_time + timedelta(seconds=t_s + tag.clock_skew_s)
                        for item in pending_batch:
                            out.append(
                                RangeMeasurement(
                                    tag_id=item.tag_id,
                                    anchor_id=item.anchor_id,
                                    distance_m=item.distance_m,
                                    variance_m2=item.variance_m2,
                                    technology=item.technology,
                                    measured_at=item.measured_at,
                                    los_confidence=item.los_confidence,
                                    azimuth_deg=item.azimuth_deg,
                                    azimuth_variance_deg2=item.azimuth_variance_deg2,
                                    elevation_deg=item.elevation_deg,
                                    azimuth_reference_frame=item.azimuth_reference_frame,
                                    wear_state=item.wear_state,
                                    motion_state=item.motion_state,
                                    battery_fraction=item.battery_fraction,
                                    adapter_id=item.adapter_id,
                                    epoch_seq=item.epoch_seq,
                                    sent_at=sent_at,
                                )
                            )
                        pending_batch.clear()
                        batch_start_s = None

            if pending_batch:
                flush_t = pending_batch[-1].measured_at
                for item in pending_batch:
                    out.append(
                        RangeMeasurement(
                            tag_id=item.tag_id,
                            anchor_id=item.anchor_id,
                            distance_m=item.distance_m,
                            variance_m2=item.variance_m2,
                            technology=item.technology,
                            measured_at=item.measured_at,
                            los_confidence=item.los_confidence,
                            azimuth_deg=item.azimuth_deg,
                            azimuth_variance_deg2=item.azimuth_variance_deg2,
                            elevation_deg=item.elevation_deg,
                            azimuth_reference_frame=item.azimuth_reference_frame,
                            wear_state=item.wear_state,
                            motion_state=item.motion_state,
                            battery_fraction=item.battery_fraction,
                            adapter_id=item.adapter_id,
                            epoch_seq=item.epoch_seq,
                            sent_at=flush_t,
                        )
                    )

        return out

    def _generate_scenario_positions(self) -> list[PositionReport]:
        assert self._scenario is not None
        scenario = self._scenario
        rng = np.random.default_rng(scenario.seed + 10_000)
        n_steps = math.floor(scenario.duration_s / scenario.step_s) + 1
        out: list[PositionReport] = []

        for tag in scenario.tags:
            if tag.report_protocol != "position":
                continue
            epoch_counter = 0
            for step_idx in range(n_steps):
                t_s = step_idx * scenario.step_s
                if t_s > scenario.duration_s + 1e-9:
                    break
                if not _is_active_at(tag.active_windows, t_s):
                    continue
                if tag.dropout_prob > 0.0 and float(rng.random()) < tag.dropout_prob:
                    continue
                epoch_counter += 1
                x_m, y_m, z_m, speed_m_s = _interpolate_waypoint(tag.waypoints, t_s)
                noisy_x = x_m + float(rng.normal(0.0, tag.position_sigma_m))
                noisy_y = y_m + float(rng.normal(0.0, tag.position_sigma_m))
                x_native, y_native = _floor_to_native(noisy_x, noisy_y, scenario.survey_transform)
                scale = scenario.survey_transform.scale if scenario.survey_transform else 1.0
                var_native = (tag.position_sigma_m / scale) ** 2
                measured_at = scenario.start_time + timedelta(seconds=t_s + tag.clock_skew_s)
                battery: float | None = None
                if tag.battery_start is not None:
                    end_batt = tag.battery_end if tag.battery_end is not None else tag.battery_start
                    frac = min(1.0, max(0.0, t_s / max(scenario.duration_s, 1e-9)))
                    battery = max(
                        0.0, min(1.0, tag.battery_start + frac * (end_batt - tag.battery_start))
                    )
                out.append(
                    PositionReport(
                        tag_id=tag.tag_id,
                        measured_at=measured_at,
                        x_native_m=x_native,
                        y_native_m=y_native,
                        z_native_m=z_m,
                        cov_native_m2=(var_native, 0.0, var_native),
                        wear_state=_schedule_wear_state(tag.wear_schedule, t_s),
                        motion_state=_schedule_motion_state(
                            tag.motion_schedule, t_s, speed_m_s, infer_from_speed=False
                        ),
                        battery_fraction=battery,
                        adapter_id=self.adapter_id,
                        epoch_seq=epoch_counter,
                    )
                )
        return out

    def _generate_scenario_rooms(self) -> list[RoomReport]:
        assert self._scenario is not None
        scenario = self._scenario
        rng = np.random.default_rng(scenario.seed + 20_000)
        n_steps = math.floor(scenario.duration_s / scenario.step_s) + 1
        out: list[RoomReport] = []

        for tag in scenario.tags:
            if tag.report_protocol != "room":
                continue
            epoch_counter = 0
            for step_idx in range(n_steps):
                t_s = step_idx * scenario.step_s
                if t_s > scenario.duration_s + 1e-9:
                    break
                if not _is_active_at(tag.active_windows, t_s):
                    continue
                if tag.dropout_prob > 0.0 and float(rng.random()) < tag.dropout_prob:
                    continue
                x_m, y_m, _z_m, speed_m_s = _interpolate_waypoint(tag.waypoints, t_s)
                room_id = _room_for_point(scenario.rooms, x_m, y_m)
                if room_id is None:
                    continue
                epoch_counter += 1
                measured_at = scenario.start_time + timedelta(seconds=t_s + tag.clock_skew_s)
                battery: float | None = None
                if tag.battery_start is not None:
                    end_batt = tag.battery_end if tag.battery_end is not None else tag.battery_start
                    frac = min(1.0, max(0.0, t_s / max(scenario.duration_s, 1e-9)))
                    battery = max(
                        0.0, min(1.0, tag.battery_start + frac * (end_batt - tag.battery_start))
                    )
                out.append(
                    RoomReport(
                        tag_id=tag.tag_id,
                        measured_at=measured_at,
                        room_id=room_id,
                        probability=0.88,
                        wear_state=_schedule_wear_state(tag.wear_schedule, t_s),
                        motion_state=_schedule_motion_state(
                            tag.motion_schedule, t_s, speed_m_s, infer_from_speed=False
                        ),
                        battery_fraction=battery,
                        adapter_id=self.adapter_id,
                        epoch_seq=epoch_counter,
                        anchors_used=1,
                    )
                )
        return out

    async def ranges(self) -> AsyncIterator[RangeMeasurement]:
        if self._scenario is not None:
            items = self._generate_scenario_ranges()
        else:
            items, _, _ = self._parse_replay_source(record_malformed=True)

        prev_ts: datetime | None = None
        for item in items:
            if self._closed:
                break
            if not self._is_tag_registered(item.tag_id):
                continue
            if self._realtime and prev_ts is not None:
                delay = max(0.0, (item.measured_at - prev_ts).total_seconds())
                if delay > 0.0:
                    await asyncio.sleep(delay)
            prev_ts = item.measured_at
            self._last_report_at = self._deps.clock()
            yield item

    async def positions(self) -> AsyncIterator[PositionReport]:
        if self._scenario is not None:
            items = self._generate_scenario_positions()
        else:
            _, items, _ = self._parse_replay_source(record_malformed=True)

        prev_ts: datetime | None = None
        for item in items:
            if self._closed:
                break
            if not self._is_tag_registered(item.tag_id):
                continue
            if self._realtime and prev_ts is not None:
                delay = max(0.0, (item.measured_at - prev_ts).total_seconds())
                if delay > 0.0:
                    await asyncio.sleep(delay)
            prev_ts = item.measured_at
            self._last_report_at = self._deps.clock()
            yield item

    async def rooms(self) -> AsyncIterator[RoomReport]:
        if self._scenario is not None:
            items = self._generate_scenario_rooms()
        else:
            _, _, items = self._parse_replay_source(record_malformed=True)

        prev_ts: datetime | None = None
        for item in items:
            if self._closed:
                break
            if not self._is_tag_registered(item.tag_id):
                continue
            if self._realtime and prev_ts is not None:
                delay = max(0.0, (item.measured_at - prev_ts).total_seconds())
                if delay > 0.0:
                    await asyncio.sleep(delay)
            prev_ts = item.measured_at
            self._last_report_at = self._deps.clock()
            yield item


def build_simulator_adapter(config: AdapterConfig, deps: AdapterDeps) -> SimulatorAdapter:
    """Registry factory for ``kind: simulator``."""
    raw = config.extra_fields()
    if config.payload_profile is not None:
        raw["payload_profile"] = config.payload_profile
    parsed = SimulatorAdapterConfig.model_validate(raw)

    if parsed.payload_profile == "scenario_yaml_v1":
        assert parsed.scenario is not None
        scenario = load_scenario(parsed.scenario)
        realtime = parsed.realtime if parsed.realtime is not None else scenario.realtime
        return SimulatorAdapter(
            adapter_id=config.id,
            deps=deps,
            scenario=scenario,
            realtime=realtime,
            ntp_synced=config.ntp_synced,
            latency_s=config.latency_s,
            reports_wear_state_override=parsed.reports_wear_state,
            reports_motion_state_override=parsed.reports_motion_state,
        )

    assert parsed.replay_path is not None
    return SimulatorAdapter(
        adapter_id=config.id,
        deps=deps,
        replay_path=Path(parsed.replay_path),
        realtime=bool(parsed.realtime),
        ntp_synced=config.ntp_synced,
        latency_s=config.latency_s,
        reports_wear_state_override=parsed.reports_wear_state,
        reports_motion_state_override=parsed.reports_motion_state,
    )


class PositionTrackSimulatorConfig(BaseModel):
    """Adapter configuration validated when ``kind == "position_track_simulator"``."""

    model_config = ConfigDict(extra="forbid")

    payload_profile: str = "position_track_sim_v1"
    seed: int = 42
    steps: int = Field(default=6, ge=1)
    step_s: float = Field(default=0.5, gt=0.0)
    frame: PositionTrackFrame = "floor_plan"
    realtime: bool = False
    inject_unregistered: bool = False
    inject_malformed: bool = False

    @model_validator(mode="after")
    def _validate_profile(self) -> PositionTrackSimulatorConfig:
        if self.payload_profile != "position_track_sim_v1":
            raise ValueError(
                f"Unknown position_track_simulator payload_profile {self.payload_profile!r}; "
                "expected 'position_track_sim_v1'"
            )
        return self


class PositionTrackSimulatorAdapter:
    """Seeded deterministic :class:`PositionTrackAdapter` for conformance and integration tests."""

    def __init__(
        self,
        *,
        adapter_id: str,
        deps: AdapterDeps,
        seed: int = 42,
        steps: int = 6,
        step_s: float = 0.5,
        frame: PositionTrackFrame = "floor_plan",
        realtime: bool = False,
        inject_unregistered: bool = False,
        inject_malformed: bool = False,
    ) -> None:
        self.adapter_id: str = adapter_id
        self._deps: AdapterDeps = deps
        self._seed: int = seed
        self._steps: int = steps
        self._step_s: float = step_s
        self._frame: PositionTrackFrame = frame
        self._realtime: bool = realtime
        self._inject_unregistered: bool = inject_unregistered
        self._inject_malformed: bool = inject_malformed
        self._closed: bool = False
        self._last_report_at: datetime | None = None
        self._malformed_reports: int = 0
        self._unregistered_reports: int = 0

    def health(self) -> AdapterHealth:
        return AdapterHealth(
            adapter_id=self.adapter_id,
            connected=not self._closed,
            last_report_at=self._last_report_at,
            malformed_reports=self._malformed_reports,
            unregistered_reports=self._unregistered_reports,
        )

    async def close(self) -> None:
        self._closed = True

    async def position_tracks(self) -> AsyncIterator[AnonymousPositionTrackReport]:
        if self._closed:
            return
        snapshot = self._deps.site_config.snapshot()
        if snapshot is None or not snapshot.position_sensors:
            return

        if self._inject_malformed:
            self._malformed_reports += 1
            POSITION_TRACKS_DROPPED_TOTAL.labels(
                reason=PositionTrackDropReason.MALFORMED_INPUT
            ).inc()
        if self._inject_unregistered:
            self._unregistered_reports += 1
            POSITION_TRACKS_DROPPED_TOTAL.labels(
                reason=PositionTrackDropReason.UNREGISTERED_SENSOR
            ).inc()

        rng = np.random.default_rng(self._seed)
        base_time = ensure_utc(self._deps.clock()) - timedelta(seconds=self._steps * self._step_s)
        for step_idx in range(self._steps):
            if self._closed:
                break
            if self._realtime and step_idx > 0:
                await asyncio.sleep(self._step_s)
            measured_at = base_time + timedelta(seconds=step_idx * self._step_s)
            for sensor in snapshot.position_sensors:
                if self._closed:
                    break
                dx = float(rng.normal(0.0, 0.05))
                dy = float(rng.normal(0.0, 0.05))
                if self._frame == "sensor_local" and sensor.yaw_deg is not None:
                    rx = 1.0 + step_idx * 0.25 + dx
                    ry = 0.2 + dy
                else:
                    rx = sensor.x_m + 0.5 + step_idx * 0.2 + dx
                    ry = sensor.y_m + 0.3 + dy
                self._last_report_at = self._deps.clock()
                yield AnonymousPositionTrackReport(
                    sensor_id=sensor.sensor_id,
                    sensor_track_id="sim-t1",
                    measured_at=measured_at,
                    x_m=rx,
                    y_m=ry,
                    cov_xx_m2=0.04,
                    cov_xy_m2=0.0,
                    cov_yy_m2=0.04,
                    speed_m_s=0.4,
                    frame="floor_plan" if sensor.yaw_deg is None else self._frame,
                    floor_plan_id=snapshot.floor_plan_id or "",
                    adapter_id=self.adapter_id,
                    sent_at=measured_at,
                )


def build_position_track_simulator_adapter(
    config: AdapterConfig,
    deps: AdapterDeps,
) -> PositionTrackSimulatorAdapter:
    """Registry factory for ``kind: position_track_simulator``."""
    raw = config.extra_fields()
    if config.payload_profile is not None:
        raw["payload_profile"] = config.payload_profile
    parsed = PositionTrackSimulatorConfig.model_validate(raw)
    return PositionTrackSimulatorAdapter(
        adapter_id=config.id,
        deps=deps,
        seed=parsed.seed,
        steps=parsed.steps,
        step_s=parsed.step_s,
        frame=parsed.frame,
        realtime=parsed.realtime,
        inject_unregistered=parsed.inject_unregistered,
        inject_malformed=parsed.inject_malformed,
    )
