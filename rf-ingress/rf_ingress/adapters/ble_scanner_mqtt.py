"""BLE room-receiver MQTT adapter (``espresense_node_v1`` payload profile).

Consumes per-device, per-receiver MQTT reports published by fixed BLE room
receivers (such as ESP32 boards running ESPresense on ``espresense/devices/+/+``),
maps each beacon's ``hardware_key`` to its registered ``tag_id``, computes
per-epoch median RSSI per receiver inside non-overlapping ``epoch_window_s``
buckets (no cross-epoch smoothing), converts median RSSI to a calibrated
log-distance range and honest variance, and emits hardware-neutral
:class:`rf_ingress.domain.RangeMeasurement` values with ``technology = ble_rssi``.

Vendor payload schemas stay strictly inside this module. Core multilateration
and 1-2 node room fallback remain owned by ``rf-ingress`` core.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import statistics
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from cts_contracts.wearables import (
    AdapterKind,
    FixKind,
    MotionState,
    RangingTechnology,
    WearState,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rf_ingress.adapters._mqtt_transport import (
    AiomqttMessageStream,
    InMemoryMqttMessageStream,
    MqttMessageStream,
)
from rf_ingress.adapters.base import AdapterDeps, AdapterHealth
from rf_ingress.adapters.ble_motion import BleMotionConfig, BleMotionInferer
from rf_ingress.config import AdapterConfig
from rf_ingress.domain import (
    AdapterCapabilityDescriptor,
    RangeMeasurement,
    SiteConfigSnapshot,
    ensure_utc,
)
from rf_ingress.observability import (
    FIXES_DROPPED_TOTAL,
    UNREGISTERED_TAG_TOTAL,
    DropReason,
)

logger = structlog.get_logger(__name__)

BLE_SCANNER_ADAPTER_VERSION = "0.1.0"
BLE_SCANNER_PROFILES: frozenset[str] = frozenset({"espresense_node_v1"})

__all__ = [
    "BLE_SCANNER_ADAPTER_VERSION",
    "BLE_SCANNER_PROFILES",
    "AiomqttMessageStream",
    "BeaconModelConfig",
    "BleNodeCalibrationConfig",
    "BleScannerMqttAdapter",
    "BleScannerMqttConfig",
    "EspresenseNodeReportV1",
    "InMemoryMqttMessageStream",
    "MqttMessageStream",
    "build_ble_scanner_mqtt_adapter",
    "range_variance_m2",
    "rssi_to_range_m",
]


def rssi_to_range_m(
    rssi_dbm: float,
    *,
    rssi_at_1m_dbm: float,
    path_loss_exponent: float,
) -> float:
    """Convert BLE RSSI (dBm) to distance (m) via the log-distance path-loss model."""
    if not (
        math.isfinite(rssi_dbm)
        and math.isfinite(rssi_at_1m_dbm)
        and math.isfinite(path_loss_exponent)
    ):
        raise ValueError("RSSI conversion inputs must be finite")
    if path_loss_exponent <= 0.0:
        raise ValueError(f"path_loss_exponent must be > 0, got {path_loss_exponent!r}")
    return float(10.0 ** ((rssi_at_1m_dbm - rssi_dbm) / (10.0 * path_loss_exponent)))


def range_variance_m2(
    range_m: float,
    *,
    sigma_floor_m: float,
    sigma_range_factor: float,
) -> float:
    """Compute honest 1-D range variance (m^2) growing monotonically with range."""
    if not (
        math.isfinite(range_m)
        and math.isfinite(sigma_floor_m)
        and math.isfinite(sigma_range_factor)
    ):
        raise ValueError("Variance inputs must be finite")
    if range_m < 0.0:
        raise ValueError(f"range_m must be >= 0, got {range_m!r}")
    if sigma_floor_m <= 0.0:
        raise ValueError(f"sigma_floor_m must be > 0, got {sigma_floor_m!r}")
    if sigma_range_factor <= 0.0:
        raise ValueError(f"sigma_range_factor must be > 0, got {sigma_range_factor!r}")
    sigma_m = max(sigma_floor_m, sigma_range_factor * range_m)
    return float(sigma_m * sigma_m)


class BeaconModelConfig(BaseModel):
    """Per-beacon-model reference RSSI at 1 metre."""

    model_config = ConfigDict(extra="forbid")

    rssi_at_1m_dbm: float = Field(default=-59.0, ge=-100.0, le=0.0)


class BleNodeCalibrationConfig(BaseModel):
    """Per-receiver path-loss calibration derived from bench measurements."""

    model_config = ConfigDict(extra="forbid")

    path_loss_exponent: float = Field(default=2.5, gt=0.5, le=6.0)


class BleScannerMqttConfig(BaseModel):
    """Adapter-specific configuration validated when ``kind == "ble_scanner_mqtt"``."""

    model_config = ConfigDict(extra="forbid")

    payload_profile: str = "espresense_node_v1"
    broker_url: str = "mqtt://localhost:1883"
    topic: str = "espresense/devices/+/+"
    username: str | None = None
    password: str | None = None
    epoch_window_s: float = Field(default=1.0, gt=0.0, le=10.0)
    max_range_m: float = Field(default=12.0, gt=0.0)
    sigma_floor_m: float = Field(default=1.0, gt=0.0)
    sigma_range_factor: float = Field(default=0.5, gt=0.0)
    motion_rssi_std_db: float = Field(default=3.0, gt=0.0)
    motion_window_s: float = Field(default=30.0, gt=0.0)
    motion_still_duration_s: float = Field(default=120.0, gt=0.0)
    motion_max_gap_s: float = Field(default=15.0, gt=0.0)
    default_path_loss_exponent: float = Field(default=2.5, gt=0.5, le=6.0)
    beacon_models: dict[str, BeaconModelConfig] = Field(
        default_factory=lambda: {"default": BeaconModelConfig(rssi_at_1m_dbm=-59.0)}
    )
    device_beacon_models: dict[str, str] = Field(default_factory=dict)
    nodes: dict[str, BleNodeCalibrationConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_profile_and_models(self) -> BleScannerMqttConfig:
        if self.payload_profile not in BLE_SCANNER_PROFILES:
            known = ", ".join(sorted(BLE_SCANNER_PROFILES))
            raise ValueError(
                f"Unknown ble_scanner_mqtt payload_profile {self.payload_profile!r}; "
                f"expected one of: {known}"
            )
        if "default" not in self.beacon_models:
            raise ValueError("beacon_models must include a 'default' entry")
        topic_parts = [p for p in self.topic.split("/") if p]
        if len(topic_parts) < 2:
            raise ValueError(f"topic must contain device and node segments, got {self.topic!r}")
        return self


class EspresenseNodeReportV1(BaseModel):
    """Pinned ESPresense v3.3.x per-device, per-node MQTT JSON payload.

    Uses ``extra="forbid"`` so firmware schema drift fails loudly (dropped and
    counted in ``malformed_input``) instead of guessing unknown fields.
    The firmware's smoothed ``distance`` is validated when present but ignored;
    ranging uses raw ``rssi`` with the adapter's calibrated log-distance model.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, max_length=256)
    rssi: float = Field(ge=-127.0, le=20.0)
    distance: float | None = Field(default=None, ge=0.0)
    mac: str | None = None
    id_type: int | None = Field(default=None, alias="idType")
    disc: str | None = None
    rssi_at_1m: float | None = Field(default=None, alias="rssi@1m")
    raw: float | None = None
    var: float | None = None
    interval_ms: int | None = Field(default=None, alias="int")
    name: str | None = None
    measured_at: datetime | None = None
    battery: float | None = Field(default=None, ge=0.0, le=100.0)
    battery_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    moving: bool | None = None

    @field_validator("rssi", "distance", "rssi_at_1m", "raw", "var", "battery", "battery_fraction")
    @classmethod
    def _check_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("numeric payload fields must be finite")
        return value

    @field_validator("measured_at")
    @classmethod
    def _check_tz_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            return ensure_utc(value)
        return None

    def resolved_battery_fraction(self) -> float | None:
        if self.battery_fraction is not None:
            return float(self.battery_fraction)
        if self.battery is not None:
            return max(0.0, min(1.0, float(self.battery) / 100.0))
        return None


@dataclass
class _TagRssiBucket:
    tag_id: str
    bucket_idx: int
    bucket_end: datetime
    latest_measured_at: datetime
    rssi_by_node: dict[str, list[float]] = field(default_factory=dict)
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = None


def _match_topic_and_extract_node(topic: str, topic_pattern: str) -> tuple[str, str] | None:
    """Validate *topic* against *topic_pattern* and return ``(topic_device_id, node_id)``."""
    topic_parts = topic.split("/")
    pattern_parts = topic_pattern.split("/")
    if len(topic_parts) != len(pattern_parts) or len(topic_parts) < 2:
        return None
    for actual, pat in zip(topic_parts, pattern_parts, strict=True):
        if pat == "+":
            if not actual.strip():
                return None
        elif actual != pat:
            return None
    device_seg = topic_parts[-2].strip()
    node_seg = topic_parts[-1].strip()
    if not device_seg or not node_seg:
        return None
    return (device_seg, node_seg)


class BleScannerMqttAdapter:
    """MQTT :class:`RangeAdapter` consuming fixed BLE room-receiver RSSI reports."""

    def __init__(
        self,
        *,
        adapter_id: str,
        deps: AdapterDeps,
        stream: MqttMessageStream,
        config: BleScannerMqttConfig | None = None,
        ntp_synced: bool = True,
        latency_s: float = 0.0,
    ) -> None:
        self.adapter_id: str = adapter_id
        self._deps: AdapterDeps = deps
        self._stream: MqttMessageStream = stream
        self._cfg: BleScannerMqttConfig = config or BleScannerMqttConfig()
        self._motion: BleMotionInferer = BleMotionInferer(
            BleMotionConfig(
                moving_window_s=self._cfg.motion_window_s,
                still_duration_s=self._cfg.motion_still_duration_s,
                motion_rssi_std_db=self._cfg.motion_rssi_std_db,
                max_gap_s=self._cfg.motion_max_gap_s,
            )
        )
        self._closed: bool = False
        self._last_report_at: datetime | None = None
        self._malformed_reports: int = 0
        self._unregistered_reports: int = 0
        self._open_buckets: dict[str, _TagRssiBucket] = {}
        self._next_epoch_seq: dict[str, int] = {}

        self.capability: AdapterCapabilityDescriptor = AdapterCapabilityDescriptor(
            adapter_id=self.adapter_id,
            adapter_kind=AdapterKind.ANCHOR_RANGING,
            adapter_version=BLE_SCANNER_ADAPTER_VERSION,
            fix_kinds=(FixKind.POSITION, FixKind.ROOM),
            nominal_rate_hz=1.0 / self._cfg.epoch_window_s,
            reports_covariance=True,
            reports_wear_state=False,
            reports_z=False,
            reports_motion_state=True,
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
        await self._stream.close()

    def _record_malformed(self, reason: str) -> None:
        self._malformed_reports += 1
        FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT).inc()
        logger.warning(
            "rf_ble_scanner_malformed_payload",
            adapter_id=self.adapter_id,
            reason=reason,
        )

    def _resolve_registered_tag_id(
        self,
        hardware_key: str,
        snapshot: SiteConfigSnapshot | None,
    ) -> str | None:
        if snapshot is None:
            return None
        device = snapshot.device_by_hardware_key(self.adapter_id, hardware_key)
        if device is not None:
            return device.tag_id
        self._unregistered_reports += 1
        UNREGISTERED_TAG_TOTAL.labels(adapter_id=self.adapter_id).inc()
        FIXES_DROPPED_TOTAL.labels(reason=DropReason.UNREGISTERED_TAG).inc()
        return None

    def _rssi_at_1m_for(self, tag_id: str) -> float:
        model_name = self._cfg.device_beacon_models.get(tag_id, "default")
        model = self._cfg.beacon_models.get(model_name) or self._cfg.beacon_models["default"]
        return model.rssi_at_1m_dbm

    def _path_loss_for(self, node_id: str) -> float:
        node_cfg = self._cfg.nodes.get(node_id)
        if node_cfg is not None:
            return node_cfg.path_loss_exponent
        return self._cfg.default_path_loss_exponent

    def _allocate_epoch_seq(self, tag_id: str) -> int:
        seq = self._next_epoch_seq.get(tag_id, 1)
        self._next_epoch_seq[tag_id] = seq + 1
        return seq

    def _flush_bucket(self, bucket: _TagRssiBucket) -> list[RangeMeasurement]:
        epoch_seq = self._allocate_epoch_seq(bucket.tag_id)
        rssi_at_1m = self._rssi_at_1m_for(bucket.tag_id)
        emitted: list[RangeMeasurement] = []
        for node_id in sorted(bucket.rssi_by_node.keys()):
            rssis = bucket.rssi_by_node[node_id]
            median_rssi = float(statistics.median(rssis))
            dist_m = rssi_to_range_m(
                median_rssi,
                rssi_at_1m_dbm=rssi_at_1m,
                path_loss_exponent=self._path_loss_for(node_id),
            )
            if dist_m > self._cfg.max_range_m:
                FIXES_DROPPED_TOTAL.labels(reason=DropReason.RANGE_OUT_OF_BOUNDS).inc()
                continue
            var_m2 = range_variance_m2(
                dist_m,
                sigma_floor_m=self._cfg.sigma_floor_m,
                sigma_range_factor=self._cfg.sigma_range_factor,
            )
            emitted.append(
                RangeMeasurement(
                    tag_id=bucket.tag_id,
                    anchor_id=node_id,
                    distance_m=dist_m,
                    variance_m2=var_m2,
                    technology=RangingTechnology.BLE_RSSI,
                    measured_at=bucket.latest_measured_at,
                    wear_state=WearState.UNSPECIFIED,
                    motion_state=bucket.motion_state,
                    battery_fraction=bucket.battery_fraction,
                    adapter_id=self.adapter_id,
                    epoch_seq=epoch_seq,
                )
            )
        return emitted

    def _flush_expired_buckets(self, now_utc: datetime) -> list[RangeMeasurement]:
        expired = [bucket for bucket in self._open_buckets.values() if bucket.bucket_end <= now_utc]
        expired.sort(key=lambda b: (b.latest_measured_at, b.tag_id))
        emitted: list[RangeMeasurement] = []
        for bucket in expired:
            self._open_buckets.pop(bucket.tag_id, None)
            emitted.extend(self._flush_bucket(bucket))
        return emitted

    def _flush_all_buckets(self) -> list[RangeMeasurement]:
        remaining = sorted(
            self._open_buckets.values(),
            key=lambda b: (b.latest_measured_at, b.tag_id),
        )
        self._open_buckets.clear()
        emitted: list[RangeMeasurement] = []
        for bucket in remaining:
            emitted.extend(self._flush_bucket(bucket))
        return emitted

    def _ingest_message(self, topic: str, raw_payload: bytes) -> list[RangeMeasurement]:
        extracted = _match_topic_and_extract_node(topic, self._cfg.topic)
        if extracted is None:
            self._record_malformed("invalid_topic")
            return []
        _topic_device_id, node_id = extracted

        try:
            decoded = json.loads(raw_payload.decode("utf-8"))
            report = EspresenseNodeReportV1.model_validate(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            self._record_malformed(type(exc).__name__)
            return []

        observed_at = (
            report.measured_at if report.measured_at is not None else ensure_utc(self._deps.clock())
        )
        emitted = self._flush_expired_buckets(observed_at)

        snapshot = self._deps.site_config.snapshot()
        tag_id = self._resolve_registered_tag_id(report.id, snapshot)
        if tag_id is None:
            return emitted

        self._last_report_at = observed_at
        motion_state = self._motion.observe(
            tag_id=tag_id,
            node_id=node_id,
            rssi_dbm=report.rssi,
            observed_at=observed_at,
            telemetry_moving=report.moving,
        )
        battery = report.resolved_battery_fraction()

        window_s = self._cfg.epoch_window_s
        bucket_idx = math.floor(observed_at.timestamp() / window_s)
        bucket_end = datetime.fromtimestamp((bucket_idx + 1) * window_s, tz=UTC)

        current = self._open_buckets.get(tag_id)
        if current is not None and current.bucket_idx != bucket_idx:
            if bucket_idx < current.bucket_idx:
                # Out-of-order hearing older than the active epoch is dropped to keep
                # per-tag epoch timestamps monotonic without cross-epoch smoothing.
                return emitted
            self._open_buckets.pop(tag_id, None)
            emitted.extend(self._flush_bucket(current))
            current = None

        if current is None:
            current = _TagRssiBucket(
                tag_id=tag_id,
                bucket_idx=bucket_idx,
                bucket_end=bucket_end,
                latest_measured_at=observed_at,
                motion_state=motion_state,
                battery_fraction=battery,
            )
            self._open_buckets[tag_id] = current
        else:
            if observed_at > current.latest_measured_at:
                current.latest_measured_at = observed_at
            current.motion_state = motion_state
            if battery is not None:
                current.battery_fraction = battery

        current.rssi_by_node.setdefault(node_id, []).append(report.rssi)
        return emitted

    async def ranges(self) -> AsyncIterator[RangeMeasurement]:
        if self._closed:
            return

        queue: asyncio.Queue[tuple[str, bytes] | None] = asyncio.Queue()

        async def _reader() -> None:
            try:
                async for item in self._stream.messages():
                    if self._closed:
                        break
                    await queue.put(item)
            finally:
                await queue.put(None)

        reader_task = asyncio.create_task(_reader())
        try:
            while not self._closed:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=self._cfg.epoch_window_s)
                except TimeoutError:
                    now_utc = ensure_utc(self._deps.clock())
                    for meas in self._flush_expired_buckets(now_utc):
                        yield meas
                    continue

                if item is None:
                    break
                topic, raw_payload = item
                for meas in self._ingest_message(topic, raw_payload):
                    yield meas

            if not self._closed:
                for meas in self._flush_all_buckets():
                    yield meas
        finally:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task


def build_ble_scanner_mqtt_adapter(
    config: AdapterConfig,
    deps: AdapterDeps,
) -> BleScannerMqttAdapter:
    """Registry factory for ``kind: ble_scanner_mqtt``."""
    raw = config.extra_fields()
    if config.payload_profile is not None:
        raw["payload_profile"] = config.payload_profile
    parsed = BleScannerMqttConfig.model_validate(raw)
    stream = AiomqttMessageStream(
        broker_url=parsed.broker_url,
        topic=parsed.topic,
        username=parsed.username,
        password=parsed.password,
    )
    return BleScannerMqttAdapter(
        adapter_id=config.id,
        deps=deps,
        stream=stream,
        config=parsed,
        ntp_synced=config.ntp_synced,
        latency_s=config.latency_s,
    )
