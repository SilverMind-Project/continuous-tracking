"""Redis Streams subscriber for ``AnonymousPositionTrack`` messages on ``position.tracks``.

Device-free position sensors (mmWave radar, room LiDAR) report where people are, never who.
Each message is decoded, validated (positive-definite covariance, freshness, a registered and
active sensor from the CC site-config, the site-config's floor plan), and buffered for
device-free tracker rounds. Malformed messages are dead-lettered (XACK + warning + counter),
never silently dropped or left pending.
"""

from __future__ import annotations

import asyncio
import math
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from google.protobuf.message import DecodeError
from structlog import get_logger

from app.domain import AnonymousPositionObservation, WearableSiteConfig
from app.observability import metrics as _metrics
from app.observability.metrics import DeadLetterStream, PositionTrackIngestResult
from app.proto.continuoustracking.v1 import position_sensor_pb2
from app.services.position_track_buffer import PositionTrackBuffer
from app.services.wearable_site_config_map import WearableSiteConfigMap

logger = get_logger(__name__)

FIELD_POSITION_TRACK = b"track"

# Smallest 2x2 covariance determinant (m^4) accepted as positive definite.
_MIN_COV_DET_M4: float = 1e-9
# A sensor clock may run this far ahead of the orchestrator before a sample is malformed.
_MAX_FUTURE_SKEW_S: float = 2.0


def decode_position_track(
    raw_bytes: bytes,
    *,
    site_config: WearableSiteConfig,
    now: datetime,
    max_track_age_s: float,
) -> tuple[AnonymousPositionObservation | None, PositionTrackIngestResult]:
    """Decode and validate one serialized ``AnonymousPositionTrack``.

    Returns ``(observation_or_none, ingest_result)``. The speed is read only when its
    presence flag is set. The calibration residual comes from the sensor's registry entry.
    """
    try:
        msg = position_sensor_pb2.AnonymousPositionTrack()
        msg.ParseFromString(raw_bytes)
    except (DecodeError, TypeError, ValueError):
        return None, PositionTrackIngestResult.DROPPED_MALFORMED

    if not msg.sensor_id or not msg.sensor_track_id or msg.measured_at_unix_ns == 0:
        return None, PositionTrackIngestResult.DROPPED_MALFORMED

    xx = float(msg.cov_xx_m2)
    xy = float(msg.cov_xy_m2)
    yy = float(msg.cov_yy_m2)
    if not (math.isfinite(xx) and math.isfinite(xy) and math.isfinite(yy)):
        return None, PositionTrackIngestResult.DROPPED_MALFORMED
    if xx <= 0.0 or yy <= 0.0 or xx * yy - xy * xy <= _MIN_COV_DET_M4:
        return None, PositionTrackIngestResult.DROPPED_MALFORMED

    speed_m_s: float | None = None
    if msg.has_speed:
        speed = float(msg.speed_m_s)
        if not math.isfinite(speed) or speed < 0.0:
            return None, PositionTrackIngestResult.DROPPED_MALFORMED
        speed_m_s = speed

    try:
        measured_at = datetime.fromtimestamp(msg.measured_at_unix_ns / 1e9, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None, PositionTrackIngestResult.DROPPED_MALFORMED
    if measured_at > now + timedelta(seconds=_MAX_FUTURE_SKEW_S):
        return None, PositionTrackIngestResult.DROPPED_MALFORMED
    if (now - measured_at).total_seconds() > max_track_age_s:
        return None, PositionTrackIngestResult.DROPPED_STALE

    sensor = site_config.position_sensor(msg.sensor_id)
    if sensor is None:
        return None, PositionTrackIngestResult.DROPPED_UNREGISTERED

    expected_plan = site_config.floor_plan_id
    if expected_plan and msg.floor_plan_id and msg.floor_plan_id != expected_plan:
        return None, PositionTrackIngestResult.DROPPED_FLOOR_PLAN

    observation = AnonymousPositionObservation(
        sensor_id=msg.sensor_id,
        sensor_track_id=msg.sensor_track_id,
        measured_at=measured_at,
        x_m=msg.x_mm / 1000.0,
        y_m=msg.y_mm / 1000.0,
        cov_m2=(xx, xy, xy, yy),
        calibration_residual_m=sensor.calibration_residual_m,
        floor_plan_id=msg.floor_plan_id or (expected_plan or ""),
        speed_m_s=speed_m_s,
    )
    return observation, PositionTrackIngestResult.ACCEPTED


class PositionTrackSubscriber:
    """Background consumer of the ``position.tracks`` Redis stream."""

    def __init__(
        self,
        *,
        redis_client: Any,
        buffer: PositionTrackBuffer,
        site_config_map: WearableSiteConfigMap,
        stream: str = "position.tracks",
        group: str = "cts-tracking-orchestrator",
        consumer: str = "orchestrator-1",
        max_track_age_s: float = 5.0,
    ) -> None:
        self._redis = redis_client
        self._buffer = buffer
        self._site_config_map = site_config_map
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._max_track_age_s = max_track_age_s
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start consuming ``position.tracks`` in the background."""
        self._running = True
        self._task = asyncio.create_task(self._consume_loop(), name="position-track-subscriber")
        logger.info("position_track_subscriber_started", stream=self._stream, group=self._group)

    async def stop(self) -> None:
        """Stop background consumption cleanly."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("position_track_subscriber_stopped", stream=self._stream)

    async def _consume_loop(self) -> None:
        # BUSYGROUP (the group already exists) is the expected outcome after the first start.
        with suppress(Exception):
            await self._redis.xgroup_create(self._stream, self._group, id="$", mkstream=True)

        while self._running:
            try:
                streams = await self._redis.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams={self._stream: ">"},
                    count=64,
                    block=1000,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # A Redis outage must not end the consumer.
                logger.exception("position_track_stream_read_failed", stream=self._stream)
                await asyncio.sleep(1.0)
                continue

            for _stream_name, messages in streams or []:
                for message_id, fields in messages or []:
                    try:
                        self.handle_fields(fields, now=datetime.now(UTC))
                    except Exception:
                        # One bad message is dead-lettered, never fatal to the loop.
                        _metrics.metrics.cts_messages_dead_lettered_total.labels(
                            stream=str(DeadLetterStream.POSITION_TRACKS)
                        ).inc()
                        logger.warning(
                            "position_track_message_dead_lettered",
                            stream=self._stream,
                            message_id=str(message_id),
                            exc_info=True,
                        )
                    finally:
                        with suppress(Exception):
                            await self._redis.xack(self._stream, self._group, message_id)

    def handle_fields(
        self,
        fields: dict[bytes, bytes] | dict[str, bytes],
        *,
        now: datetime,
    ) -> PositionTrackIngestResult:
        """Decode one stream entry, count it, and buffer it when valid."""
        raw_bytes: bytes | None = None
        if FIELD_POSITION_TRACK in fields:
            raw_bytes = fields[FIELD_POSITION_TRACK]  # type: ignore[index]
        elif "track" in fields:
            raw_bytes = fields["track"]  # type: ignore[index]

        if not raw_bytes:
            result = PositionTrackIngestResult.DROPPED_MALFORMED
            observation = None
        else:
            observation, result = decode_position_track(
                raw_bytes,
                site_config=self._site_config_map.get_sync(),
                now=now,
                max_track_age_s=self._max_track_age_s,
            )

        _metrics.metrics.cts_position_tracks_total.labels(result=str(result)).inc()
        if result == PositionTrackIngestResult.DROPPED_MALFORMED:
            _metrics.metrics.cts_messages_dead_lettered_total.labels(
                stream=str(DeadLetterStream.POSITION_TRACKS)
            ).inc()
            logger.warning("position_track_malformed", stream=self._stream)
            return result
        if observation is not None:
            self._buffer.add(observation)
        return result
