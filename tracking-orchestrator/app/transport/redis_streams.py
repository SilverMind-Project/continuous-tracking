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
from ..inference.schemas import PoseResult
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
    revisions_stream: str = "tracking.revisions"
    signals_stream: str = "tracking.signals"
    scene_samples_stream: str = "scene.samples"
    presence_stream: str = "tracking.presence"
    dwell_stream: str = "tracking.dwell"


# Re-export proto FrameReady so callers use the wire type directly without
# a redundant conversion layer.  The proto class is constructed identically
# (all fields are optional kwargs) and accessed via the same attribute names.
FrameReady = frame_pb2.FrameReady


@dataclass
class FrameBatch:
    """A batch of frames ready for processing."""

    frames: list[frame_pb2.FrameReady] = field(default_factory=list)
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
                id="$",
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
        """Yield FrameReady proto messages via XREADGROUP."""
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
                try:
                    frame = proto_decode(fields, frame_pb2.FrameReady, field=FIELD_FRAME)
                except Exception:
                    logger.exception("Failed to deserialize FrameReady")
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
        *,
        detections: list[Detection] | None = None,
        minio_key: str = "",
        room_name: str = "",
        identities: dict[str, tuple[str, float]] | None = None,
        frame_width: int = 0,
        frame_height: int = 0,
        capture_time_unix_ns: int = 0,
        detection_count: int = 0,
        pose_results: dict[str, PoseResult] | None = None,
        trail_by_ph: dict[str, list[tuple[float, float]]] | None = None,
        evidence_by_ph: dict[str, tuple[float, float, bool]] | None = None,
        det_posture: dict[str, str] | None = None,
        identity_snapshots: list[dict[str, object]] | None = None,
    ) -> str:
        """Publish a ``TrackingEvent`` proto to ``tracking.events``.

        Args:
            camera_id: the camera that produced this event.
            event_time: wall-clock time of the event.
            frame_index: the source frame index.
            detections: per-person detections to embed in the proto.
            minio_key: MinIO key of the frame for downstream review.
            room_name: resolved room name for the camera.
            identities: mapping ``ph_id -> (identity_id, confidence)``
                for detections that resolved to a committed identity. Each
                entry becomes an ``IdentityRevision`` sub-message.
            frame_width: source frame pixel width.
            frame_height: source frame pixel height.
            capture_time_unix_ns: source frame capture timestamp in unix ns.
            identity_snapshots: list of identity snapshot dicts for field 8.

        Returns:
            The Redis message ID of the published event (decoded).
        """
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
            frame_width=frame_width,
            frame_height=frame_height,
            capture_time_unix_ns=capture_time_unix_ns,
            pose_results=pose_results,
            trail_by_ph=trail_by_ph,
            evidence_by_ph=evidence_by_ph,
            det_posture=det_posture,
            identity_snapshots=identity_snapshots or [],
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
        outcome = "success" if success else (error_code or "processing_error")
        metrics.metrics.tracking_responses_published_total.labels(outcome=outcome).inc()
        return (
            message_id_bytes.decode("ascii")
            if isinstance(message_id_bytes, bytes)
            else str(message_id_bytes)
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
    frame_width: int = 0,
    frame_height: int = 0,
    capture_time_unix_ns: int = 0,
    pose_results: dict[str, PoseResult] | None = None,
    trail_by_ph: dict[str, list[tuple[float, float]]] | None = None,
    evidence_by_ph: dict[str, tuple[float, float, bool]] | None = None,
    det_posture: dict[str, str] | None = None,
    identity_snapshots: list[dict[str, object]] | None = None,
) -> tracking_pb2.TrackingEvent:
    """Build a TrackingEvent proto from domain types.

    The ``embedding`` field on Detection is intentionally not populated:
    the gallery owns canonical per-person embeddings; shipping a 768-float
    array per detection per frame would 10x the wire payload with no
    consumer.

    Optional enrichment kwargs:
        pose_results: detection_id → PoseResult (17 COCO keypoints).
        trail_by_ph: ph_id → list of (x, y) normalised foot-points.
        evidence_by_ph: ph_id → (top_prob, top2_prob, face_anchor_used).
        det_posture: detection_id → posture string (standing|sitting|walking|lying|unknown).
        identity_snapshots: list of dicts with identity snapshot fields.
    """
    event = tracking_pb2.TrackingEvent(
        camera_id=camera_id,
        event_time_unix_ns=event_time_ns,
        room_name=room_name,
        event_id=event_id,
    )
    event.frame_ref.minio_key = minio_key
    event.frame_ref.frame_index = frame_index
    event.frame_ref.width = frame_width
    event.frame_ref.height = frame_height
    event.frame_ref.capture_time_unix_ns = capture_time_unix_ns

    for det in detections:
        d = event.detections.add(
            detection_id=det.detection_id,
            confidence=det.confidence,
            ph_id=det.ph_id or "",
        )
        d.bbox.x_min = det.bbox.x_min
        d.bbox.y_min = det.bbox.y_min
        d.bbox.x_max = det.bbox.x_max
        d.bbox.y_max = det.bbox.y_max
        d.floor_point.x_mm = det.floor_point.x_mm
        d.floor_point.y_mm = det.floor_point.y_mm
        d.floor_point.calibrated = det.floor_point.calibrated

        # Floor position in metres (non-zero only when homography is calibrated).
        if det.floor_point.calibrated:
            d.floor_x = det.floor_point.x_mm / 1000.0
            d.floor_y = det.floor_point.y_mm / 1000.0

        # Pose keypoints (normalised within bbox crop).
        if pose_results and det.detection_id in pose_results:
            pr = pose_results[det.detection_id]
            for kp in pr.keypoints:
                d.pose_keypoints.add(x=kp.x, y=kp.y, score=kp.score)

        # Classified posture.
        if det_posture and det.detection_id in det_posture:
            d.posture = det_posture[det.detection_id]

        # Historical trail for this PH.
        if trail_by_ph and det.ph_id and det.ph_id in trail_by_ph:
            for tx, ty in trail_by_ph[det.ph_id]:
                d.trail.add(x=tx, y=ty)

        # Posterior evidence from identity resolver.
        if evidence_by_ph and det.ph_id and det.ph_id in evidence_by_ph:
            top_prob, top2_prob, face_anchor_used = evidence_by_ph[det.ph_id]
            d.evidence.top_prob = top_prob
            d.evidence.top2_prob = top2_prob
            d.evidence.face_anchor_used = face_anchor_used

    # per-detection identity revisions use ph_id.
    for ph_id, (identity_id, confidence) in identities.items():
        if not ph_id or not identity_id:
            continue
        revision = event.identity_revisions.add(
            ph_id=ph_id,
            map_identity_id=identity_id,
        )
        revision.candidates.add(identity_id=identity_id, probability=float(confidence))

    # Identity snapshots (field 8) — canonical per-frame identity display.
    if identity_snapshots:
        for snap in identity_snapshots:
            s = event.identity_snapshots.add()
            s.ph_id = str(snap.get("ph_id", ""))
            s.identity_id = str(snap.get("identity_id", "") or "")
            s.top_probability = float(snap.get("top_probability", 0.0) or 0.0)  # type: ignore[arg-type]
            s.second_probability = float(snap.get("second_probability", 0.0) or 0.0)  # type: ignore[arg-type]
            s.posterior_entropy = float(snap.get("posterior_entropy", 0.0) or 0.0)  # type: ignore[arg-type]
            s.direct_face_evidence = bool(snap.get("direct_face_evidence", False))
            s.evidence_json = str(snap.get("evidence_json", ""))
            s.mean_quality = float(snap.get("mean_quality", 0.0) or 0.0)  # type: ignore[arg-type]

    return event
