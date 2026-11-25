"""Transform, protobuf encoding, and Redis Streams publishing for anonymous position tracks.

Device-free position sensors (mmWave radar, room LiDAR) report anonymous floor-plane tracks
on ``position.tracks`` under field ``"track"``, trimmed with ``XADD ... MINID ~ <now - 1h>``.
Tracks may arrive either already in the shared floor-plan frame (``frame="floor_plan"``) or
in the sensor's local frame (``frame="sensor_local"``), where the registered pose rotates both
coordinates and the 2x2 covariance matrix into the household floor frame.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from cts_contracts.position_sensors import is_valid_sensor_id
from redis.asyncio import Redis

from rf_ingress.domain import (
    AnonymousPositionTrackReport,
    EmittedPositionTrack,
    PositionSensorConfig,
    ensure_utc,
)
from rf_ingress.observability import (
    POSITION_TRACKS_PUBLISHED_TOTAL,
    PositionTrackDropReason,
)
from rf_ingress.proto.continuoustracking.v1 import position_sensor_pb2
from rf_ingress.publish import MAX_RETENTION_HOURS, compute_minid_ms, datetime_to_unix_ns

FIELD_POSITION_TRACK = "track"
DEFAULT_POSITION_TRACK_RETENTION_HOURS = 1.0
_MIN_COV_DET_M4 = 1e-9


class PositionTrackContractError(ValueError):
    """Raised when an outgoing ``AnonymousPositionTrack`` violates wire-contract invariants."""


class PositionTrackDropError(ValueError):
    """Raised when a reported track must be dropped with a closed metric reason."""

    def __init__(self, reason: PositionTrackDropReason, detail: str) -> None:
        super().__init__(detail)
        self.reason: PositionTrackDropReason = reason


def validate_position_track_cov(cov_xx_m2: float, cov_xy_m2: float, cov_yy_m2: float) -> None:
    """Validate that a 2x2 floor covariance matrix is finite and positive definite."""
    if not (math.isfinite(cov_xx_m2) and math.isfinite(cov_xy_m2) and math.isfinite(cov_yy_m2)):
        raise PositionTrackDropError(
            PositionTrackDropReason.INVALID_COVARIANCE,
            "Position track covariance entries must be finite",
        )
    det = cov_xx_m2 * cov_yy_m2 - cov_xy_m2 * cov_xy_m2
    if cov_xx_m2 <= 0.0 or cov_yy_m2 <= 0.0 or det <= _MIN_COV_DET_M4:
        raise PositionTrackDropError(
            PositionTrackDropReason.INVALID_COVARIANCE,
            f"Position track covariance is not positive definite (det={det!r})",
        )


def transform_position_track(
    report: AnonymousPositionTrackReport,
    sensor: PositionSensorConfig,
    *,
    site_floor_plan_id: str | None,
) -> EmittedPositionTrack:
    """Validate *report* and transform it into the household floor-plan frame."""
    if not is_valid_sensor_id(report.sensor_id) or not report.sensor_track_id:
        raise PositionTrackDropError(
            PositionTrackDropReason.MALFORMED_INPUT,
            "sensor_id or sensor_track_id is invalid",
        )
    if not (math.isfinite(report.x_m) and math.isfinite(report.y_m)):
        raise PositionTrackDropError(
            PositionTrackDropReason.MALFORMED_INPUT,
            "Position track coordinates must be finite",
        )
    if report.speed_m_s is not None and (
        not math.isfinite(report.speed_m_s) or report.speed_m_s < 0.0
    ):
        raise PositionTrackDropError(
            PositionTrackDropReason.MALFORMED_INPUT,
            f"speed_m_s must be finite and non-negative, got {report.speed_m_s!r}",
        )

    validate_position_track_cov(report.cov_xx_m2, report.cov_xy_m2, report.cov_yy_m2)

    if report.floor_plan_id and site_floor_plan_id and report.floor_plan_id != site_floor_plan_id:
        raise PositionTrackDropError(
            PositionTrackDropReason.FLOOR_PLAN_MISMATCH,
            f"Report floor_plan_id {report.floor_plan_id!r} != {site_floor_plan_id!r}",
        )

    xx = float(report.cov_xx_m2)
    xy = float(report.cov_xy_m2)
    yy = float(report.cov_yy_m2)

    if report.frame == "floor_plan":
        fx_m = float(report.x_m)
        fy_m = float(report.y_m)
        cxx, cxy, cyy = xx, xy, yy
    elif report.frame == "sensor_local":
        if sensor.yaw_deg is None or not math.isfinite(sensor.yaw_deg):
            raise PositionTrackDropError(
                PositionTrackDropReason.MISSING_SENSOR_YAW,
                "sensor_local report requires registered finite sensor yaw_deg",
            )
        theta = math.radians(sensor.yaw_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        fx_m = sensor.x_m + c * report.x_m - s * report.y_m
        fy_m = sensor.y_m + s * report.x_m + c * report.y_m
        # R Sigma R^T for planar rotation R = [[c, -s], [s, c]].
        cxx = c * c * xx - 2.0 * c * s * xy + s * s * yy
        cxy = c * s * (xx - yy) + (c * c - s * s) * xy
        cyy = s * s * xx + 2.0 * c * s * xy + c * c * yy
    else:
        raise PositionTrackDropError(
            PositionTrackDropReason.MALFORMED_INPUT,
            f"Unsupported coordinate frame {report.frame!r}",
        )

    validate_position_track_cov(cxx, cxy, cyy)

    return EmittedPositionTrack(
        sensor_id=report.sensor_id,
        sensor_track_id=report.sensor_track_id,
        measured_at=ensure_utc(report.measured_at),
        x_mm=round(fx_m * 1000.0),
        y_mm=round(fy_m * 1000.0),
        cov_xx_m2=cxx,
        cov_xy_m2=cxy,
        cov_yy_m2=cyy,
        speed_m_s=report.speed_m_s,
        floor_plan_id=site_floor_plan_id or report.floor_plan_id,
    )


def validate_position_track_proto(msg: position_sensor_pb2.AnonymousPositionTrack) -> None:
    """Validate *msg* against the ``AnonymousPositionTrack`` wire contract."""
    if not is_valid_sensor_id(msg.sensor_id):
        raise PositionTrackContractError("AnonymousPositionTrack.sensor_id must be a valid slug")
    if not msg.sensor_track_id:
        raise PositionTrackContractError("AnonymousPositionTrack.sensor_track_id must be non-empty")
    if msg.measured_at_unix_ns == 0:
        raise PositionTrackContractError(
            "AnonymousPositionTrack.measured_at_unix_ns must be non-zero"
        )
    if not (
        math.isfinite(msg.cov_xx_m2)
        and math.isfinite(msg.cov_xy_m2)
        and math.isfinite(msg.cov_yy_m2)
    ):
        raise PositionTrackContractError("Covariance values must be finite")
    det = msg.cov_xx_m2 * msg.cov_yy_m2 - msg.cov_xy_m2 * msg.cov_xy_m2
    if msg.cov_xx_m2 <= 0.0 or msg.cov_yy_m2 <= 0.0 or det <= _MIN_COV_DET_M4:
        raise PositionTrackContractError("Covariance matrix must be positive definite")
    if msg.has_speed and (not math.isfinite(msg.speed_m_s) or msg.speed_m_s < 0.0):
        raise PositionTrackContractError("speed_m_s must be finite and non-negative")


def build_position_track_proto(
    track: EmittedPositionTrack,
) -> position_sensor_pb2.AnonymousPositionTrack:
    """Serialize *track* to an ``AnonymousPositionTrack`` protobuf message and validate it."""
    msg = position_sensor_pb2.AnonymousPositionTrack(
        sensor_id=track.sensor_id,
        sensor_track_id=track.sensor_track_id,
        measured_at_unix_ns=datetime_to_unix_ns(track.measured_at),
        x_mm=track.x_mm,
        y_mm=track.y_mm,
        cov_xx_m2=track.cov_xx_m2,
        cov_xy_m2=track.cov_xy_m2,
        cov_yy_m2=track.cov_yy_m2,
        has_speed=track.speed_m_s is not None,
        floor_plan_id=track.floor_plan_id,
    )
    if track.speed_m_s is not None:
        msg.speed_m_s = track.speed_m_s
    validate_position_track_proto(msg)
    return msg


class PositionTrackPublisher(Protocol):
    """Hardware-neutral stream publisher for validated anonymous position tracks."""

    async def publish(self, track: EmittedPositionTrack, *, adapter_id: str) -> str: ...

    def is_healthy(self) -> bool: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class RecordedPositionTrackPublication:
    """Captured position-track publication stored by :class:`InMemoryPositionTrackPublisher`."""

    entry_id: str
    stream: str
    minid: str
    payload: bytes
    proto: position_sensor_pb2.AnonymousPositionTrack
    track: EmittedPositionTrack
    adapter_id: str


class InMemoryPositionTrackPublisher:
    """In-memory :class:`PositionTrackPublisher` peer for unit and conformance tests."""

    def __init__(
        self,
        *,
        stream: str = "position.tracks",
        retention_hours: float = DEFAULT_POSITION_TRACK_RETENTION_HOURS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if retention_hours <= 0.0 or retention_hours > MAX_RETENTION_HOURS:
            raise ValueError(f"retention_hours must be in (0, {MAX_RETENTION_HOURS}]")
        self._stream: str = stream
        self._retention_hours: float = retention_hours
        self._clock: Callable[[], datetime] = clock
        self._seq: int = 0
        self.tracks: list[RecordedPositionTrackPublication] = []
        self._closed: bool = False

    async def publish(self, track: EmittedPositionTrack, *, adapter_id: str) -> str:
        proto = build_position_track_proto(track)
        payload = proto.SerializeToString()
        minid = compute_minid_ms(self._clock(), self._retention_hours)
        self._seq += 1
        entry_id = f"{self._seq}-0"
        self.tracks.append(
            RecordedPositionTrackPublication(
                entry_id=entry_id,
                stream=self._stream,
                minid=minid,
                payload=payload,
                proto=proto,
                track=track,
                adapter_id=adapter_id,
            )
        )
        POSITION_TRACKS_PUBLISHED_TOTAL.labels(adapter_id=adapter_id).inc()
        return entry_id

    def is_healthy(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True


class RedisPositionTrackPublisher:
    """Redis Streams publisher emitting raw protobuf ``AnonymousPositionTrack`` with 1h MINID."""

    def __init__(
        self,
        redis_client: Redis,
        *,
        stream: str = "position.tracks",
        retention_hours: float = DEFAULT_POSITION_TRACK_RETENTION_HOURS,
        close_client: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if retention_hours <= 0.0 or retention_hours > MAX_RETENTION_HOURS:
            raise ValueError(f"retention_hours must be in (0, {MAX_RETENTION_HOURS}]")
        self._redis: Redis = redis_client
        self._stream: str = stream
        self._retention_hours: float = retention_hours
        self._close_client: bool = close_client
        self._clock: Callable[[], datetime] = clock
        self._healthy: bool = True

    async def publish(self, track: EmittedPositionTrack, *, adapter_id: str) -> str:
        proto = build_position_track_proto(track)
        payload = proto.SerializeToString()
        minid = compute_minid_ms(self._clock(), self._retention_hours)
        raw_id = await self._redis.xadd(
            self._stream,
            {FIELD_POSITION_TRACK: payload},
            minid=minid,
            approximate=True,
        )
        self._healthy = True
        POSITION_TRACKS_PUBLISHED_TOTAL.labels(adapter_id=adapter_id).inc()
        return raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)

    def is_healthy(self) -> bool:
        return self._healthy

    async def close(self) -> None:
        self._healthy = False
        if self._close_client:
            await self._redis.aclose()
