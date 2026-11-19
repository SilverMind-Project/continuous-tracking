"""Protobuf encoding, validation, and Redis Streams publishers for wearable fixes.

All messages travel on Redis Streams as raw protobuf bytes (never JSON or
base64):
- ``wearable.fixes`` field ``"fix"`` carrying ``TagFix``, trimmed with
  ``XADD ... MINID ~ <now - retention_hours>`` (at most 72 hours).
- ``wearable.capabilities`` field ``"capability"`` carrying
  ``AdapterCapability``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from cts_contracts.wearables import AdapterKind, FixKind, MotionState, WearState
from redis.asyncio import Redis

from rf_ingress.domain import AdapterCapabilityDescriptor, EmittedFix, ensure_utc
from rf_ingress.observability import (
    FIX_SIGMA_M,
    FIXES_PUBLISHED_TOTAL,
    OFF_BODY_FIXES_TOTAL,
)
from rf_ingress.proto.continuoustracking.v1 import wearable_pb2

FIELD_WEARABLE_FIX = "fix"
FIELD_WEARABLE_CAPABILITY = "capability"
MAX_RETENTION_HOURS = 72.0

_FIX_KIND_TO_PROTO: dict[FixKind, wearable_pb2.FixKind] = {
    FixKind.POSITION: wearable_pb2.FIX_KIND_POSITION,
    FixKind.ROOM: wearable_pb2.FIX_KIND_ROOM,
    FixKind.PROXIMITY: wearable_pb2.FIX_KIND_PROXIMITY,
}

_ADAPTER_KIND_TO_PROTO: dict[AdapterKind, wearable_pb2.AdapterKind] = {
    AdapterKind.DEVICE_RANGING: wearable_pb2.ADAPTER_KIND_DEVICE_RANGING,
    AdapterKind.ANCHOR_RANGING: wearable_pb2.ADAPTER_KIND_ANCHOR_RANGING,
    AdapterKind.SOLVED_POSITION: wearable_pb2.ADAPTER_KIND_SOLVED_POSITION,
    AdapterKind.ROOM_PRESENCE: wearable_pb2.ADAPTER_KIND_ROOM_PRESENCE,
    AdapterKind.SIMULATOR: wearable_pb2.ADAPTER_KIND_SIMULATOR,
}

_WEAR_STATE_TO_PROTO: dict[WearState, wearable_pb2.WearState] = {
    WearState.UNSPECIFIED: wearable_pb2.WEAR_STATE_UNSPECIFIED,
    WearState.ON_BODY: wearable_pb2.WEAR_STATE_ON_BODY,
    WearState.OFF_BODY: wearable_pb2.WEAR_STATE_OFF_BODY,
}

_MOTION_STATE_TO_PROTO: dict[MotionState, wearable_pb2.MotionState] = {
    MotionState.UNSPECIFIED: wearable_pb2.MOTION_STATE_UNSPECIFIED,
    MotionState.MOVING: wearable_pb2.MOTION_STATE_MOVING,
    MotionState.STILL: wearable_pb2.MOTION_STATE_STILL,
}


class TagFixContractError(ValueError):
    """Raised when an outgoing ``TagFix`` violates wire-contract invariants."""


def datetime_to_unix_ns(dt: datetime) -> int:
    """Convert a timezone-aware datetime to integer Unix nanoseconds."""
    utc_dt = ensure_utc(dt)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc_dt - epoch
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def compute_minid_ms(now: datetime, retention_hours: float = MAX_RETENTION_HOURS) -> str:
    """Compute a Redis Stream ``MINID`` entry-id threshold for *retention_hours*."""
    if retention_hours <= 0.0 or retention_hours > MAX_RETENTION_HOURS:
        raise ValueError(f"retention_hours must be in (0, {MAX_RETENTION_HOURS}]")
    cutoff = ensure_utc(now) - timedelta(hours=retention_hours)
    ms = max(0, int(cutoff.timestamp() * 1000))
    return f"{ms}-0"


def validate_tag_fix_proto(msg: wearable_pb2.TagFix) -> None:
    """Validate *msg* against the hardware-neutral ``TagFix`` wire contract."""
    if not msg.tag_id:
        raise TagFixContractError("TagFix.tag_id must be non-empty")
    if msg.measured_at_unix_ns == 0:
        raise TagFixContractError("TagFix.measured_at_unix_ns must be non-zero")
    if msg.kind == wearable_pb2.FIX_KIND_UNSPECIFIED:
        raise TagFixContractError("TagFix.kind must not be FIX_KIND_UNSPECIFIED")

    if msg.kind == wearable_pb2.FIX_KIND_POSITION:
        if not msg.has_position:
            raise TagFixContractError("FIX_KIND_POSITION requires has_position=True")
        xx = msg.cov_xx_m2
        xy = msg.cov_xy_m2
        yy = msg.cov_yy_m2
        if not (math.isfinite(xx) and math.isfinite(xy) and math.isfinite(yy)):
            raise TagFixContractError("FIX_KIND_POSITION covariance values must be finite")
        det = xx * yy - xy * xy
        if xx <= 0.0 or yy <= 0.0 or det <= 0.0:
            raise TagFixContractError(
                "FIX_KIND_POSITION requires a positive-definite 2x2 covariance"
            )

    if msg.has_room_probability and not (0.0 <= msg.room_probability <= 1.0):
        raise TagFixContractError("TagFix.room_probability must be in [0, 1]")
    if msg.has_distance and (not math.isfinite(msg.distance_m) or msg.distance_m < 0.0):
        raise TagFixContractError("TagFix.distance_m must be finite and non-negative")
    if msg.has_quality and not (0.0 <= msg.quality <= 1.0):
        raise TagFixContractError("TagFix.quality must be in [0, 1]")
    if msg.has_residual and (not math.isfinite(msg.residual_rms_m) or msg.residual_rms_m < 0.0):
        raise TagFixContractError("TagFix.residual_rms_m must be finite and non-negative")
    if msg.has_battery and not (0.0 <= msg.battery_fraction <= 1.0):
        raise TagFixContractError("TagFix.battery_fraction must be in [0, 1]")


def build_tag_fix_proto(fix: EmittedFix) -> wearable_pb2.TagFix:
    """Encode and validate a frozen :class:`EmittedFix` as a ``TagFix`` protobuf."""
    has_position = fix.x_mm is not None and fix.y_mm is not None
    msg = wearable_pb2.TagFix(
        tag_id=fix.tag_id,
        measured_at_unix_ns=datetime_to_unix_ns(fix.measured_at),
        kind=_FIX_KIND_TO_PROTO[fix.kind],
        epoch_seq=max(0, fix.epoch_seq),
        has_position=has_position,
        x_mm=fix.x_mm if fix.x_mm is not None else 0,
        y_mm=fix.y_mm if fix.y_mm is not None else 0,
        has_z=fix.z_mm is not None,
        z_mm=fix.z_mm if fix.z_mm is not None else 0,
        cov_xx_m2=float(fix.cov_xx_m2),
        cov_xy_m2=float(fix.cov_xy_m2),
        cov_yy_m2=float(fix.cov_yy_m2),
        floor_plan_id=fix.floor_plan_id,
        room_id=fix.room_id,
        has_room_probability=fix.room_probability is not None,
        room_probability=float(fix.room_probability) if fix.room_probability is not None else 0.0,
        anchor_id=fix.anchor_id,
        has_distance=fix.distance_m is not None,
        distance_m=float(fix.distance_m) if fix.distance_m is not None else 0.0,
        wear_state=_WEAR_STATE_TO_PROTO[fix.wear_state],
        has_quality=fix.quality is not None,
        quality=float(fix.quality) if fix.quality is not None else 0.0,
        anchors_used=max(0, fix.anchors_used),
        has_residual=fix.residual_rms_m is not None,
        residual_rms_m=float(fix.residual_rms_m) if fix.residual_rms_m is not None else 0.0,
        adapter_id=fix.adapter_id,
        adapter_kind=_ADAPTER_KIND_TO_PROTO[fix.adapter_kind],
        adapter_version=fix.adapter_version,
        motion_state=_MOTION_STATE_TO_PROTO[fix.motion_state],
        has_battery=fix.battery_fraction is not None,
        battery_fraction=float(fix.battery_fraction) if fix.battery_fraction is not None else 0.0,
    )
    validate_tag_fix_proto(msg)
    return msg


def build_capability_proto(
    capability: AdapterCapabilityDescriptor,
    *,
    reported_at: datetime,
) -> wearable_pb2.AdapterCapability:
    """Encode an :class:`AdapterCapabilityDescriptor` as an ``AdapterCapability`` protobuf."""
    return wearable_pb2.AdapterCapability(
        adapter_id=capability.adapter_id,
        adapter_kind=_ADAPTER_KIND_TO_PROTO[capability.adapter_kind],
        fix_kinds=[_FIX_KIND_TO_PROTO[kind] for kind in capability.fix_kinds],
        nominal_rate_hz=float(capability.nominal_rate_hz),
        reports_covariance=capability.reports_covariance,
        reports_wear_state=capability.reports_wear_state,
        reports_z=capability.reports_z,
        reported_at_unix_ns=datetime_to_unix_ns(reported_at),
        reports_motion_state=capability.reports_motion_state,
    )


def _record_publish_metrics(fix: EmittedFix) -> None:
    FIXES_PUBLISHED_TOTAL.labels(kind=fix.kind.value).inc()
    if fix.wear_state == WearState.OFF_BODY:
        OFF_BODY_FIXES_TOTAL.labels(adapter_id=fix.adapter_id).inc()
    if fix.kind == FixKind.POSITION:
        sigma_m = math.sqrt(0.5 * (fix.cov_xx_m2 + fix.cov_yy_m2))
        if math.isfinite(sigma_m):
            FIX_SIGMA_M.observe(sigma_m)


class TagFixPublisher(Protocol):
    """Protocol for publishing validated wearable fixes and capability heartbeats."""

    async def publish(self, fix: EmittedFix) -> str: ...

    async def publish_capability(
        self,
        capability: AdapterCapabilityDescriptor,
        *,
        reported_at: datetime,
    ) -> str: ...

    def is_healthy(self) -> bool: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class RecordedFixPublication:
    """Captured publication record stored by :class:`InMemoryTagFixPublisher`."""

    entry_id: str
    stream: str
    minid: str
    payload: bytes
    proto: wearable_pb2.TagFix
    fix: EmittedFix


@dataclass(frozen=True)
class RecordedCapabilityPublication:
    """Captured capability heartbeat stored by :class:`InMemoryTagFixPublisher`."""

    entry_id: str
    stream: str
    payload: bytes
    proto: wearable_pb2.AdapterCapability


class InMemoryTagFixPublisher:
    """In-memory :class:`TagFixPublisher` peer for unit and conformance tests."""

    def __init__(
        self,
        *,
        fixes_stream: str = "wearable.fixes",
        capabilities_stream: str = "wearable.capabilities",
        retention_hours: float = MAX_RETENTION_HOURS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._fixes_stream: str = fixes_stream
        self._capabilities_stream: str = capabilities_stream
        self._retention_hours: float = retention_hours
        self._clock: Callable[[], datetime] = clock
        self._seq: int = 0
        self.fixes: list[RecordedFixPublication] = []
        self.capabilities: list[RecordedCapabilityPublication] = []
        self._closed: bool = False

    async def publish(self, fix: EmittedFix) -> str:
        proto = build_tag_fix_proto(fix)
        payload = proto.SerializeToString()
        minid = compute_minid_ms(self._clock(), self._retention_hours)
        self._seq += 1
        entry_id = f"{self._seq}-0"
        self.fixes.append(
            RecordedFixPublication(
                entry_id=entry_id,
                stream=self._fixes_stream,
                minid=minid,
                payload=payload,
                proto=proto,
                fix=fix,
            )
        )
        _record_publish_metrics(fix)
        return entry_id

    async def publish_capability(
        self,
        capability: AdapterCapabilityDescriptor,
        *,
        reported_at: datetime,
    ) -> str:
        proto = build_capability_proto(capability, reported_at=reported_at)
        payload = proto.SerializeToString()
        self._seq += 1
        entry_id = f"{self._seq}-0"
        self.capabilities.append(
            RecordedCapabilityPublication(
                entry_id=entry_id,
                stream=self._capabilities_stream,
                payload=payload,
                proto=proto,
            )
        )
        return entry_id

    def is_healthy(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True


class RedisTagFixPublisher:
    """Redis Streams publisher emitting raw protobuf with 72-hour ``MINID`` trim."""

    def __init__(
        self,
        redis_client: Redis,
        *,
        fixes_stream: str = "wearable.fixes",
        capabilities_stream: str = "wearable.capabilities",
        retention_hours: float = MAX_RETENTION_HOURS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if retention_hours <= 0.0 or retention_hours > MAX_RETENTION_HOURS:
            raise ValueError(f"retention_hours must be in (0, {MAX_RETENTION_HOURS}]")
        self._redis: Redis = redis_client
        self._fixes_stream: str = fixes_stream
        self._capabilities_stream: str = capabilities_stream
        self._retention_hours: float = retention_hours
        self._clock: Callable[[], datetime] = clock
        self._healthy: bool = True

    async def publish(self, fix: EmittedFix) -> str:
        proto = build_tag_fix_proto(fix)
        payload = proto.SerializeToString()
        minid = compute_minid_ms(self._clock(), self._retention_hours)
        raw_id = await self._redis.xadd(
            self._fixes_stream,
            {FIELD_WEARABLE_FIX: payload},
            minid=minid,
            approximate=True,
        )
        self._healthy = True
        _record_publish_metrics(fix)
        return raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)

    async def publish_capability(
        self,
        capability: AdapterCapabilityDescriptor,
        *,
        reported_at: datetime,
    ) -> str:
        proto = build_capability_proto(capability, reported_at=reported_at)
        payload = proto.SerializeToString()
        raw_id = await self._redis.xadd(
            self._capabilities_stream,
            {FIELD_WEARABLE_CAPABILITY: payload},
            maxlen=1000,
            approximate=True,
        )
        self._healthy = True
        return raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)

    def is_healthy(self) -> bool:
        return self._healthy

    async def close(self) -> None:
        self._healthy = False
        await self._redis.aclose()
