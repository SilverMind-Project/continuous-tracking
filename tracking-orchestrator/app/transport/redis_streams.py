"""Redis Streams transport layer.

Provides a durable, replay-capable message transport using Redis Streams.
The tracking-orchestrator consumes FrameReady messages from the
`frames.ready` stream and publishes TrackingEvent results to the
`tracking.events` stream.

All messages are protobuf-serialized (via the proto-generated Python
bindings) and stored in Redis Streams with consumer groups for
at-least-once delivery.
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
from ..inference.triton_client import TritonClientProtocol

logger = get_logger(__name__)

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
    max_retries: int = 3


# ---------------------------------------------------------------------------
# FrameReady message (proto-generated, but we define the shape here for
# type safety without requiring proto compilation in the base path).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameReady:
    """Deserialized FrameReady message from Redis Streams.

    This mirrors proto/continuoustracking/v1/frame.proto::FrameReady.
    In production, use the proto-generated Python class directly.
    """

    camera_id: str
    minio_key: str
    frame_index: int
    capture_time_unix_ns: int
    received_time_unix_ns: int
    width: int
    height: int
    sample_fps: float = 0.0


# ---------------------------------------------------------------------------
# FrameBatch: collects frames for batched processing
# ---------------------------------------------------------------------------


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
    - consume_frames(): async generator yielding FrameReady messages
    - publish_event(): publish a tracking event to the events stream
    - publish_response(): publish a FrameResponse for XACK

    Usage::

        transport = RedisStreamsTransport(config)
        await transport.connect()

        async for frame in transport.consume_frames():
            # Process frame...
            await transport.publish_response(frame, success=True)

        await transport.disconnect()
    """

    def __init__(self, config: TransportConfig | None = None) -> None:
        self._config = config or TransportConfig()
        self._redis: redis.Redis | None = None
        self._group_created = False
        # Maps id(frame) → (Redis message ID, monotonic timestamp) for pending XACK
        self._pending_acks: dict[int, tuple[Any, float]] = {}

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        """Connect to Redis and create the consumer group if needed."""
        if self._redis is not None:
            return

        self._redis = redis.from_url(
            self._config.redis_url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

        # Create consumer group (ignores error if it already exists)
        try:
            await self._redis.xgroup_create(
                self._config.frames_stream,
                self._config.consumer_group,
                id="0",  # Start from beginning
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
        """Remove pending ack entries older than the configured TTL.

        Entries that have not been ACKed within the TTL window are
        discarded to prevent unbounded memory growth.
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
        """Consume FrameReady messages from the frames.ready stream.

        Uses XREADGROUP with consumer group for at-least-once delivery.
        Messages are XACKed after the caller processes them.

        Args:
            count: maximum number of messages to read per call.

        Yields:
            FrameReady messages deserialized from Redis Streams.
        """
        if self._redis is None:
            logger.warning("Cannot consume: not connected to Redis")
            return

        self._cleanup_stale_acks()

        # XREADGROUP with block
        streams = await self._redis.xreadgroup(
            self._config.consumer_group,
            self._config.consumer_name,
            {self._config.frames_stream: ">"},  # Only new messages
            count=count,
            block=self._config.batch_max_wait_ms,
        )

        if not streams:
            return

        for _stream_name, messages in streams:
            for message_id, fields in messages:
                frame = self._deserialize_frame(fields)
                if frame is not None:
                    # Store the message ID keyed by object identity for later XACK
                    self._pending_acks[id(frame)] = (message_id, time.monotonic())
                    yield frame

    async def ack_frame(self, frame: FrameReady) -> None:
        """Acknowledge processing of a FrameReady message.

        This tells Redis that the message has been successfully processed
        and can be removed from the pending entries list.

        Args:
            frame: the FrameReady message to acknowledge.
        """
        if self._redis is None:
            return

        entry = self._pending_acks.pop(id(frame), None)
        if entry is None:
            logger.warning("Cannot ACK: no message ID on frame", camera_id=frame.camera_id)
            return

        message_id = entry[0]
        await self._redis.xack(
            self._config.frames_stream,
            self._config.consumer_group,
            message_id,
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
        """Publish a tracking event to the tracking.events stream.

        Args:
            camera_id: the camera that produced this event.
            event_time: wall-clock time of the event.
            frame_index: the frame index.
            detection_count: number of detections in this event.
            detections: optional detection details for the event payload.
            minio_key: optional MinIO key of the frame for downstream review.
            room_name: optional resolved room name for the camera.
            identities: mapping ``global_track_id -> (identity_id, confidence)``
                for detections that resolved to a committed identity. Missing
                entries are treated as UNKNOWN.

        Returns:
            The Redis message ID of the published event.
        """
        if self._redis is None:
            logger.error("Cannot publish: not connected to Redis")
            return ""

        event_id = str(uuid.uuid4())
        event_time_ns = int(event_time.timestamp() * 1e9)

        payload: dict[str, str] = {
            "event_id": event_id,
            "camera_id": camera_id,
            "event_time_unix_ns": str(event_time_ns),
            "frame_index": str(frame_index),
            "detection_count": str(detection_count),
            "minio_key": minio_key,
            "room_name": room_name,
        }

        id_map = identities or {}
        if detections:
            for i, det in enumerate(detections):
                prefix = f"detection.{i}"
                payload[f"{prefix}.id"] = det.detection_id
                payload[f"{prefix}.bbox_xmin"] = str(det.bbox.x_min)
                payload[f"{prefix}.bbox_ymin"] = str(det.bbox.y_min)
                payload[f"{prefix}.bbox_xmax"] = str(det.bbox.x_max)
                payload[f"{prefix}.bbox_ymax"] = str(det.bbox.y_max)
                payload[f"{prefix}.confidence"] = str(det.confidence)
                payload[f"{prefix}.tracklet_id"] = det.tracklet_id or ""
                payload[f"{prefix}.global_track_id"] = det.global_track_id or ""
                payload[f"{prefix}.floor_x_mm"] = str(det.floor_point.x_mm)
                payload[f"{prefix}.floor_y_mm"] = str(det.floor_point.y_mm)
                id_entry = id_map.get(det.global_track_id)
                if id_entry is not None:
                    identity_id, identity_conf = id_entry
                    payload[f"{prefix}.identity_id"] = identity_id
                    payload[f"{prefix}.identity_confidence"] = f"{identity_conf:.6f}"
                else:
                    payload[f"{prefix}.identity_id"] = ""
                    payload[f"{prefix}.identity_confidence"] = "0"

        message_id = await self._redis.xadd(
            self._config.events_stream,
            payload,  # type: ignore[arg-type]
            maxlen=10000,  # Auto-trim old entries
            approximate=True,
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
        """Publish a FrameResponse for a processed FrameReady message.

        This is used by rtsp-ingress for acknowledgment and metrics.

        Args:
            frame: the original FrameReady message.
            success: whether processing succeeded.
            detection_count: number of detections produced.
            error_code: error code if not success.
            processing_latency_us: processing latency in microseconds.

        Returns:
            The Redis message ID of the published response.
        """
        if self._redis is None:
            return ""

        completed_time_ns = int(datetime.now(UTC).timestamp() * 1e9)

        payload: dict[str, str] = {
            "camera_id": frame.camera_id,
            "frame_index": str(frame.frame_index),
            "success": str(int(success)),
            "error_code": error_code,
            "detection_count": str(detection_count),
            "processing_latency_us": str(processing_latency_us),
            "completed_time_unix_ns": str(completed_time_ns),
        }

        message_id = await self._redis.xadd(
            self._config.responses_stream,
            payload,  # type: ignore[arg-type]
            maxlen=10000,
            approximate=True,
        )

        return message_id

    def _deserialize_frame(self, fields: dict[str, str]) -> FrameReady | None:
        """Deserialize a Redis Streams fields dict into a FrameReady message."""
        try:
            return FrameReady(
                camera_id=fields.get("camera_id", ""),
                minio_key=fields.get("minio_key", ""),
                frame_index=int(fields.get("frame_index", "0")),
                capture_time_unix_ns=int(fields.get("capture_time_unix_ns", "0")),
                received_time_unix_ns=int(fields.get("received_time_unix_ns", "0")),
                width=int(fields.get("width", "0")),
                height=int(fields.get("height", "0")),
                sample_fps=float(fields.get("sample_fps", "0.0")),
            )
        except (ValueError, KeyError) as exc:
            logger.error("Failed to deserialize FrameReady", error=str(exc), fields=fields)
            return None

    async def stream_length(self, stream: str | None = None) -> int:
        """Return the number of entries in a stream."""
        if self._redis is None:
            return 0

        stream_name = stream or self._config.frames_stream
        info = await self._redis.xinfo_stream(stream_name)
        return int(info.get("length", 0))

    async def pending_count(self) -> int:
        """Return the number of pending messages for this consumer group."""
        if self._redis is None:
            return 0

        # XINFO CONSUMERS returns consumer info including pending count
        stream = self._config.frames_stream
        group = self._config.consumer_group
        consumers = await self._redis.xinfo_consumers(stream, group)
        if not consumers:
            return 0

        return int(consumers[0].get("pending", 0))


# ---------------------------------------------------------------------------
# Frame processor: the main processing loop that uses transport + inference
# ---------------------------------------------------------------------------


async def process_frame(
    frame: FrameReady,
    detector: TritonClientProtocol,
    tracker: Any,  # PerCameraTracker
    transport: RedisStreamsTransport,
    detector_confidence: float = 0.25,
) -> tuple[int, int]:
    """Process a single FrameReady message through the detection + tracking pipeline.

    This is the core M4 processing loop:
    1. Fetch the frame JPEG from MinIO (via aiobotocore).
    2. Run YOLO11m detection via Triton.
    3. Run per-camera tracking (BoT-SORT).
    4. Publish results.

    Args:
        frame: the FrameReady message from Redis Streams.
        detector: the Triton client for person detection.
        tracker: the per-camera tracker instance.
        transport: the Redis Streams transport.
        detector_confidence: minimum confidence for detections.

    Returns:
        (detection_count, processing_latency_us)
    """
    import time

    start = time.monotonic()

    # TODO: fetch frame JPEG from MinIO via aiobotocore
    # image = await minio.fetch_jpeg(frame.minio_key)

    # For now, return zero detections (skeleton mode)
    # In production, this would be:
    #   detections = await detector.detect(image, confidence=detector_confidence)
    #   local_tracks = tracker.update(frame.camera_id, detections, frame.frame_index)

    detection_count = 0

    latency_us = int((time.monotonic() - start) * 1e6)

    # Publish response for XACK
    await transport.publish_response(
        frame=frame,
        success=True,
        detection_count=detection_count,
        processing_latency_us=latency_us,
    )

    return detection_count, latency_us
