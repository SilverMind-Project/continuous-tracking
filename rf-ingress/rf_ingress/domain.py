"""Hardware-neutral frozen domain objects for rf-ingress.

All floor-plane positions are expressed in metres (``_m``) and covariances in
square metres (``_m2``) in the shared household floor-plan frame defined by
Cognitive Companion. Vendor-frame positions on :class:`PositionReport` are
converted to metres by the producing adapter before entering the frame
transform. Closed vocabularies are imported directly from
``cts_contracts.wearables``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from cts_contracts.wearables import (
    AdapterKind,
    DeviceKind,
    FixKind,
    MotionState,
    RangingTechnology,
    RfNodeRole,
    WearState,
)

AzimuthReferenceFrame = Literal["device_local", "floor_plan"]
PositionTrackFrame = Literal["floor_plan", "sensor_local"]


def ensure_utc(dt: datetime) -> datetime:
    """Validate *dt* is timezone-aware and return it normalized to UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamps in rf-ingress must be timezone-aware")
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class RfNodeGeometry:
    """Fixed RF node (receiver or ranging anchor) in the shared floor frame.

    ``x_m``, ``y_m``, and ``z_m`` are metres in the household floor-plan frame
    (``x_m``/``y_m`` on the floor plane, ``z_m`` mounting height above floor).
    """

    anchor_id: str
    x_m: float
    y_m: float
    z_m: float
    adapter_id: str = ""
    node_role: RfNodeRole = RfNodeRole.RANGING_ANCHOR
    technology: RangingTechnology = RangingTechnology.UWB
    room_id: str | None = None
    uwb_address_hex: str | None = None
    uwb_channel: int | None = None
    uwb_preamble_index: int | None = None


# Alias matching the geometric solver terminology.
AnchorGeometry = RfNodeGeometry


@dataclass(frozen=True)
class SiteDevice:
    """One registered wearable device synced from Cognitive Companion."""

    tag_id: str
    adapter_id: str
    hardware_key: str
    device_kind: DeviceKind
    is_active: bool


@dataclass(frozen=True)
class SiteBinding:
    """Time-bounded binding of a wearable ``tag_id`` to a household member.

    Valid on the half-open UTC interval ``[valid_from, valid_to)`` (inclusive
    start, exclusive end; open-ended when ``valid_to`` is ``None``).
    """

    tag_id: str
    member_id: str
    valid_from: datetime
    valid_to: datetime | None = None

    def covers(self, at: datetime) -> bool:
        """Return ``True`` when *at* falls inside ``[valid_from, valid_to)``."""
        ts = ensure_utc(at)
        start = ensure_utc(self.valid_from)
        if ts < start:
            return False
        if self.valid_to is None:
            return True
        return ts < ensure_utc(self.valid_to)


@dataclass(frozen=True)
class PositionSensorConfig:
    """One registered, active anonymous position sensor from CC site-config."""

    sensor_id: str
    room_id: str | None
    x_m: float
    y_m: float
    yaw_deg: float | None
    calibration_residual_m: float


@dataclass(frozen=True)
class SiteConfigSnapshot:
    """Immutable snapshot of household RF nodes, devices, bindings, and position sensors from CC.

    ``version`` is the content hash produced by Cognitive Companion. Whenever
    ``version`` changes, in-flight epochs are dropped so no fix mixes two
    floor-plan scales or node layouts.
    """

    floor_plan_id: str | None = None
    nodes: tuple[RfNodeGeometry, ...] = ()
    bindings: tuple[SiteBinding, ...] = ()
    devices: tuple[SiteDevice, ...] = ()
    position_sensors: tuple[PositionSensorConfig, ...] = ()
    version: str = ""

    def node_map(self) -> dict[str, RfNodeGeometry]:
        """Return active RF nodes indexed by ``anchor_id``."""
        return {node.anchor_id: node for node in self.nodes}

    def position_sensor(self, sensor_id: str) -> PositionSensorConfig | None:
        """Return the registered active position sensor for *sensor_id*, or ``None``."""
        for sensor in self.position_sensors:
            if sensor.sensor_id == sensor_id:
                return sensor
        return None

    def device_by_tag_id(self, tag_id: str) -> SiteDevice | None:
        """Return the registered device for *tag_id*, or ``None``."""
        for device in self.devices:
            if device.tag_id == tag_id:
                return device
        return None

    def device_by_hardware_key(self, adapter_id: str, hardware_key: str) -> SiteDevice | None:
        """Return the registered device matching (*adapter_id*, *hardware_key*)."""
        for device in self.devices:
            if device.adapter_id == adapter_id and device.hardware_key == hardware_key:
                return device
        return None

    def member_at(self, tag_id: str, at: datetime) -> str | None:
        """Resolve the bound member id for *tag_id* at *at*, or ``None``."""
        for binding in self.bindings:
            if binding.tag_id == tag_id and binding.covers(at):
                return binding.member_id
        return None


@dataclass(frozen=True)
class RangeMeasurement:
    """Single tag-to-node distance observation produced by a ``RangeAdapter``.

    ``distance_m`` is metres between the tag and ``anchor_id``; ``variance_m2``
    is the strictly positive 1-D ranging variance in square metres. Optional
    angular telemetry (``azimuth_deg``, ``elevation_deg``,
    ``azimuth_variance_deg2``) is expressed in degrees in
    ``azimuth_reference_frame`` (``"device_local"`` for wrist/tag local boresight
    or ``"floor_plan"`` for calibrated fixed-anchor bearings).
    """

    tag_id: str
    anchor_id: str
    distance_m: float
    variance_m2: float
    technology: RangingTechnology
    measured_at: datetime
    los_confidence: float | None = None
    azimuth_deg: float | None = None
    azimuth_variance_deg2: float | None = None
    elevation_deg: float | None = None
    azimuth_reference_frame: AzimuthReferenceFrame | None = None
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = None
    adapter_id: str = ""
    epoch_seq: int | None = None
    sent_at: datetime | None = None

    def __post_init__(self) -> None:
        ensure_utc(self.measured_at)
        if self.sent_at is not None:
            ensure_utc(self.sent_at)
        if not math.isfinite(self.distance_m) or self.distance_m < 0.0:
            raise ValueError(f"distance_m must be finite and >= 0, got {self.distance_m!r}")
        if not math.isfinite(self.variance_m2) or self.variance_m2 <= 0.0:
            raise ValueError(f"variance_m2 must be finite and > 0, got {self.variance_m2!r}")
        if self.battery_fraction is not None and not (0.0 <= self.battery_fraction <= 1.0):
            raise ValueError(f"battery_fraction must be in [0, 1], got {self.battery_fraction!r}")


@dataclass(frozen=True)
class PositionReport:
    """Position report already solved by an external system in its native frame.

    ``x_native_m``, ``y_native_m``, and optional ``z_native_m`` are metres in
    the adapter's native coordinate frame (converted from vendor units by the
    adapter). ``cov_native_m2`` is ``(cov_xx_m2, cov_xy_m2, cov_yy_m2)`` in
    square metres in that same frame.
    """

    tag_id: str
    measured_at: datetime
    x_native_m: float
    y_native_m: float
    z_native_m: float | None = None
    cov_native_m2: tuple[float, float, float] | None = None
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = None
    adapter_id: str = ""
    epoch_seq: int | None = None
    sent_at: datetime | None = None

    def __post_init__(self) -> None:
        ensure_utc(self.measured_at)
        if self.sent_at is not None:
            ensure_utc(self.sent_at)
        if not (math.isfinite(self.x_native_m) and math.isfinite(self.y_native_m)):
            raise ValueError("PositionReport native coordinates must be finite")
        if self.z_native_m is not None and not math.isfinite(self.z_native_m):
            raise ValueError("PositionReport z_native_m must be finite")
        if self.battery_fraction is not None and not (0.0 <= self.battery_fraction <= 1.0):
            raise ValueError(f"battery_fraction must be in [0, 1], got {self.battery_fraction!r}")


@dataclass(frozen=True)
class RoomReport:
    """Room-level presence report in Cognitive Companion room-id vocabulary."""

    tag_id: str
    measured_at: datetime
    room_id: str
    probability: float | None = None
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = None
    adapter_id: str = ""
    epoch_seq: int | None = None
    sent_at: datetime | None = None
    anchor_id: str = ""
    distance_m: float | None = None
    anchors_used: int = 0

    def __post_init__(self) -> None:
        ensure_utc(self.measured_at)
        if self.sent_at is not None:
            ensure_utc(self.sent_at)
        if not self.room_id:
            raise ValueError("RoomReport.room_id must be non-empty")
        if self.probability is not None and not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be in [0, 1], got {self.probability!r}")
        if self.battery_fraction is not None and not (0.0 <= self.battery_fraction <= 1.0):
            raise ValueError(f"battery_fraction must be in [0, 1], got {self.battery_fraction!r}")


@dataclass(frozen=True)
class AdapterCapabilityDescriptor:
    """Declared capabilities advertised on ``wearable.capabilities``."""

    adapter_id: str
    adapter_kind: AdapterKind
    adapter_version: str
    fix_kinds: tuple[FixKind, ...]
    nominal_rate_hz: float
    reports_covariance: bool
    reports_wear_state: bool
    reports_z: bool
    reports_motion_state: bool
    ntp_synced: bool = True
    latency_s: float = 0.0


@dataclass(frozen=True)
class RangeEpoch:
    """One grouped solver epoch of range measurements for a single tag."""

    tag_id: str
    adapter_id: str
    epoch_seq: int
    measured_at: datetime
    measurements: tuple[RangeMeasurement, ...]
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    battery_fraction: float | None = None
    sent_at: datetime | None = None


@dataclass(frozen=True)
class SolveResult:
    """Solved position and 2x2 horizontal covariance in the shared floor frame.

    ``x_m``, ``y_m``, and optional ``z_m`` are metres in the household floor
    frame; ``cov_xx_m2``, ``cov_xy_m2``, and ``cov_yy_m2`` form a symmetric
    positive-definite 2x2 horizontal covariance matrix in square metres.
    """

    x_m: float
    y_m: float
    z_m: float | None
    cov_xx_m2: float
    cov_xy_m2: float
    cov_yy_m2: float
    horizontal_sigma_m: float
    quality: float
    anchors_used: int
    residual_rms_m: float
    dropped_outlier_anchor_id: str | None = None
    mode_3d: bool = False


@dataclass(frozen=True)
class EmittedFix:
    """Validated hardware-neutral fix ready for protobuf serialization.

    Floor-plane position coordinates ``x_mm``, ``y_mm``, and optional ``z_mm``
    are millimetres in the household floor-plan frame; covariance is in square
    metres.
    """

    tag_id: str
    measured_at: datetime
    kind: FixKind
    epoch_seq: int
    adapter_id: str
    adapter_kind: AdapterKind
    adapter_version: str
    floor_plan_id: str = ""
    x_mm: int | None = None
    y_mm: int | None = None
    z_mm: int | None = None
    cov_xx_m2: float = 0.0
    cov_xy_m2: float = 0.0
    cov_yy_m2: float = 0.0
    room_id: str = ""
    room_probability: float | None = None
    anchor_id: str = ""
    distance_m: float | None = None
    wear_state: WearState = WearState.UNSPECIFIED
    motion_state: MotionState = MotionState.UNSPECIFIED
    quality: float | None = None
    anchors_used: int = 0
    residual_rms_m: float | None = None
    battery_fraction: float | None = None


@dataclass(frozen=True)
class AnonymousPositionTrackReport:
    """Hardware-neutral anonymous target track report from a position-sensor adapter."""

    sensor_id: str
    sensor_track_id: str
    measured_at: datetime
    x_m: float
    y_m: float
    cov_xx_m2: float
    cov_xy_m2: float
    cov_yy_m2: float
    speed_m_s: float | None = None
    frame: PositionTrackFrame = "floor_plan"
    floor_plan_id: str = ""
    adapter_id: str = ""
    sent_at: datetime | None = None

    def __post_init__(self) -> None:
        ensure_utc(self.measured_at)
        if self.sent_at is not None:
            ensure_utc(self.sent_at)


@dataclass(frozen=True)
class EmittedPositionTrack:
    """Validated floor-frame anonymous position track ready for ``position.tracks``."""

    sensor_id: str
    sensor_track_id: str
    measured_at: datetime
    x_mm: int
    y_mm: int
    cov_xx_m2: float
    cov_xy_m2: float
    cov_yy_m2: float
    speed_m_s: float | None = None
    floor_plan_id: str = ""

    def __post_init__(self) -> None:
        ensure_utc(self.measured_at)
