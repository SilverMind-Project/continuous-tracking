"""Redis Streams transport layer (proto-only).

The orchestrator consumes ``FrameReady`` proto messages from
``frames.ready`` (published by the Go ingress) and publishes
``TrackingEvent`` proto messages to ``tracking.events``. Each Redis
Streams message carries exactly one named field whose value is the raw
``Message.SerializeToString()`` output -- no codec discriminator, no
JSON, no base64.

The Redis client runs with ``decode_responses=False`` so binary proto
payloads round-trip unchanged.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as redis
from structlog import get_logger

from ..domain import Detection
from ..observability import metrics
from ..proto.continuoustracking.v1 import frame_pb2, tracking_pb2
from .codec import decode as proto_decode
from .codec import encode as proto_encode

logger = get_logger(__name__)

# Field names per stream. Each stream carries one proto type, so the
# field name doubles as a content hint for ``XRANGE`` debugging.
FIELD_FRAME = "frame"
FIELD_EVENT = "event"
FIELD_RESPONSE = "response"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportConfig:
    """Configuration for the Redis Streams transport."""

    redis_url: str = "redis://localhost:6379/0"
    consumer_group: str = "cts-orchestrator"
    consumer_name: str = "orchestrator-1"
    frames_stream: str = "frames.ready"
    events_stream: str = "tracking.events"
    responses_stream: str = "tracking.responses"
    batch_max_wait_ms: int = 100
    batch_max_size: int = 8
    xack_timeout_ms: int = 5000
    ack_ttl_seconds: int = 300


# ---------------------------------------------------------------------------
# FrameReady domain shape (mirrors frame.proto::FrameReady).  We keep a
# frozen dataclass so the rest of the codebase doesn't import the proto
# class directly.  Conversion happens at the transport boundary.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameReady:
    """Domain shape for an inbound frame from rtsp-ingress."""

    camera_id: str
    minio_key: str
    frame_index: int
    capture_time_unix_ns: int
    received_time_unix_ns: int
    width: int
    height: int
    sample_fps: float = 0.0


@dataclass
class FrameBatch:
    """A batch of frames ready for processing."""

    frames: list[FrameReady] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Main transport class
# ---------------------------------------------------------------------------


class RedisStreamsTransport:
    """Redis Streams transport for the tracking orchestrator.

    Provides:

    - :meth:`consume_frames` -- async generator yielding FrameReady messages.
    - :meth:`publish_event` -- emit a TrackingEvent.
    - :meth:`publish_response` -- emit a FrameResponse for XACK metrics.

    Usage::

        transport = RedisStreamsTransport(config)
        await transport.connect()
        async for frame in transport.consume_frames():
            ...
            await transport.publish_response(frame, success=True)
        await transport.disconnect()
    """

    def __init__(self, config: TransportConfig | None = None) -> None:
        self._config = config or TransportConfig()
        self._redis: redis.Redis | None = None
        self._group_created = False
        # id(frame) -> (Redis message ID bytes, monotonic timestamp)
        self._pending_acks: dict[int, tuple[Any, float]] = {}

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        """Connect to Redis and create the consumer group if needed."""
        if self._redis is not None:
            return

        # decode_responses=False so proto bytes round-trip unchanged.
        self._redis = redis.from_url(
            self._config.redis_url,
            decode_responses=False,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

        try:
            await self._redis.xgroup_create(
                self._config.frames_stream,
                self._config.consumer_group,
                id="0",
                mkstream=True,
            )
            self._group_created = True
            logger.info("Created Redis consumer group", group=self._config.consumer_group)
        except redis.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                self._group_created = True
                logger.info("Consumer group already exists", group=self._config.consumer_group)
            else:
                raise

        logger.info("Connected to Redis", url=self._config.redis_url)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")

    def _cleanup_stale_acks(self) -> None:
        """Evict pending-ack entries older than the configured TTL.

        Prevents unbounded memory growth when consumers fail to ack.
        """
        now = time.monotonic()
        stale = [
            fid
            for fid, (_, ts) in self._pending_acks.items()
            if now - ts > self._config.ack_ttl_seconds
        ]
        for fid in stale:
            del self._pending_acks[fid]
        if stale:
            logger.info(
                "Cleaned up stale pending acks",
                count=len(stale),
                remaining=len(self._pending_acks),
            )

    async def consume_frames(self, count: int = 1) -> AsyncIterator[FrameReady]:
        """Yield FrameReady messages via XREADGROUP."""
        if self._redis is None:
            logger.warning("Cannot consume: not connected to Redis")
            return

        self._cleanup_stale_acks()

        streams = await self._redis.xreadgroup(
            self._config.consumer_group,
            self._config.consumer_name,
            {self._config.frames_stream: ">"},
            count=count,
            block=self._config.batch_max_wait_ms,
        )
        if not streams:
            return

        for _stream_name, messages in streams:
            for message_id, fields in messages:
                frame = self._deserialize_frame(fields)
                if frame is None:
                    continue
                self._pending_acks[id(frame)] = (message_id, time.monotonic())
                metrics.metrics.frames_consumed_total.labels(camera_id=frame.camera_id).inc()
                yield frame

    async def ack_frame(self, frame: FrameReady) -> None:
        """Acknowledge processing of a FrameReady message."""
        if self._redis is None:
            return

        entry = self._pending_acks.pop(id(frame), None)
        if entry is None:
            logger.warning("Cannot ACK: no message ID on frame", camera_id=frame.camera_id)
            return

        await self._redis.xack(
            self._config.frames_stream,
            self._config.consumer_group,
            entry[0],
        )

    async def publish_event(
        self,
        camera_id: str,
        event_time: datetime,
        frame_index: int,
        detection_count: int,
        detections: list[Detection] | None = None,
        minio_key: str = "",
        room_name: str = "",
        identities: dict[str, tuple[str, float]] | None = None,
    ) -> str:
        """Publish a ``TrackingEvent`` proto to ``tracking.events``.

        Args:
            camera_id: the camera that produced this event.
            event_time: wall-clock time of the event.
            frame_index: the source frame index.
            detection_count: number of detections in this event (kept for
                back-compat with internal counters; defaults to len(detections)).
            detections: per-person detections to embed in the proto.
            minio_key: MinIO key of the frame for downstream review.
            room_name: resolved room name for the camera (currently
                propagated separately by :class:`KeyframeSampler`; reserved
                for inclusion when a future proto revision lifts it).
            identities: mapping ``global_track_id -> (identity_id, confidence)``
                for detections that resolved to a committed identity. Each
                entry becomes an ``IdentityRevision`` sub-message.

        Returns:
            The Redis message ID of the published event (decoded).
        """
        del detection_count  # derived from len(detections); kept for back-compat callers
        if self._redis is None:
            logger.error("Cannot publish: not connected to Redis")
            return ""

        event_id = str(uuid.uuid4())
        event_pb = _build_tracking_event_pb(
            camera_id=camera_id,
            event_id=event_id,
            event_time_ns=int(event_time.timestamp() * 1e9),
            frame_index=frame_index,
            minio_key=minio_key,
            room_name=room_name,
            detections=detections or [],
            identities=identities or {},
        )

        message_id_bytes = await self._redis.xadd(
            self._config.events_stream,
            proto_encode(event_pb, field=FIELD_EVENT),  # type: ignore[arg-type]
            maxlen=10000,
            approximate=True,
        )

        metrics.metrics.tracking_events_published_total.labels(camera_id=camera_id).inc()
        message_id = (
            message_id_bytes.decode("ascii")
            if isinstance(message_id_bytes, bytes)
            else str(message_id_bytes)
        )
        logger.debug("Published tracking event", event_id=event_id, message_id=message_id)
        return message_id

    async def publish_response(
        self,
        frame: FrameReady,
        success: bool,
        detection_count: int = 0,
        error_code: str = "",
        processing_latency_us: int = 0,
    ) -> str:
        """Publish a ``FrameResponse`` proto to ``tracking.responses``."""
        if self._redis is None:
            return ""

        response = frame_pb2.FrameResponse(
            camera_id=frame.camera_id,
            frame_index=frame.frame_index,
            success=success,
            error_code=error_code,
            detection_count=detection_count,
            processing_latency_us=processing_latency_us,
            completed_time_unix_ns=int(datetime.now(UTC).timestamp() * 1e9),
        )

        message_id_bytes = await self._redis.xadd(
            self._config.responses_stream,
            proto_encode(response, field=FIELD_RESPONSE),  # type: ignore[arg-type]
            maxlen=10000,
            approximate=True,
        )
        return (
            message_id_bytes.decode("ascii")
            if isinstance(message_id_bytes, bytes)
            else str(message_id_bytes)
        )

    def _deserialize_frame(self, fields: dict[Any, Any]) -> FrameReady | None:
        """Decode a ``FrameReady`` proto from a Redis-Streams field dict."""
        try:
            message = proto_decode(fields, frame_pb2.FrameReady, field=FIELD_FRAME)
        except Exception as exc:  # proto parse / lookup can raise non-ValueError types
            logger.error("Failed to deserialize FrameReady", error=str(exc))
            return None
        return FrameReady(
            camera_id=message.camera_id,
            minio_key=message.minio_key,
            frame_index=int(message.frame_index),
            capture_time_unix_ns=int(message.capture_time_unix_ns),
            received_time_unix_ns=int(message.received_time_unix_ns),
            width=int(message.width),
            height=int(message.height),
            sample_fps=float(message.sample_fps),
        )

    async def stream_length(self, stream: str | None = None) -> int:
        if self._redis is None:
            return 0
        info = await self._redis.xinfo_stream(stream or self._config.frames_stream)
        return int(info.get(b"length") or info.get("length") or 0)

    async def pending_count(self) -> int:
        if self._redis is None:
            return 0
        consumers = await self._redis.xinfo_consumers(
            self._config.frames_stream, self._config.consumer_group
        )
        if not consumers:
            return 0
        first = consumers[0]
        return int(first.get(b"pending") or first.get("pending") or 0)


# ---------------------------------------------------------------------------
# Proto build helpers
# ---------------------------------------------------------------------------


def _build_tracking_event_pb(
    *,
    camera_id: str,
    event_id: str,
    event_time_ns: int,
    frame_index: int,
    minio_key: str,
    room_name: str,
    detections: list[Detection],
    identities: dict[str, tuple[str, float]],
) -> tracking_pb2.TrackingEvent:
    """Build a TrackingEvent proto from domain types.

    The ``embedding`` field on Detection is intentionally not populated:
    the gallery owns canonical per-person embeddings; shipping a 768-float
    array per detection per frame would 10x the wire payload with no
    consumer.
    """
    event = tracking_pb2.TrackingEvent(
        camera_id=camera_id,
        event_time_unix_ns=event_time_ns,
        room_name=room_name,
        event_id=event_id,
    )
    event.frame_ref.minio_key = minio_key
    event.frame_ref.frame_index = frame_index

    for det in detections:
        d = event.detections.add(
            detection_id=det.detection_id,
            confidence=det.confidence,
            tracklet_id=det.tracklet_id or "",
            global_track_id=det.global_track_id or "",
        )
        d.bbox.x_min = det.bbox.x_min
        d.bbox.y_min = det.bbox.y_min
        d.bbox.x_max = det.bbox.x_max
        d.bbox.y_max = det.bbox.y_max
        d.floor_point.x_mm = det.floor_point.x_mm
        d.floor_point.y_mm = det.floor_point.y_mm
        d.floor_point.calibrated = det.floor_point.calibrated

    # Per-detection identity decisions are folded into the per-event
    # IdentityRevision repeated field. Stream-level revision fields
    # (revision_id, tracklet_ids, ...) are unset here -- those carry
    # meaning only on the standalone tracking.revisions stream.
    for global_track_id, (identity_id, confidence) in identities.items():
        if not global_track_id or not identity_id:
            continue
        revision = event.identity_revisions.add(
            global_track_id=global_track_id,
            map_identity_id=identity_id,
        )
        revision.candidates.add(identity_id=identity_id, probability=float(confidence))

    return event
