"""SceneSamplesPublisher: emits TaggedKeyframe messages to the scene.samples stream.

The scene.samples Redis Stream is consumed by the scene-worker (CTSSceneWorker)
in the cognitive-companion service, which runs VLM analysis on tagged keyframes.

Each message carries the minimal fields needed for the scene worker to fetch
the frame from MinIO and run inference:
- keyframe_id, tracklet_id, global_track_id, camera_id
- minio_key: path in MinIO to the JPEG frame
- captured_at: ISO 8601 timestamp
- tag_reason: 'periodic' | 'identity_changed' | 'hazard' | 'dwell_start'
- annotations: JSON-encoded dict with bbox, person_id, posture, confidence
"""

from __future__ import annotations

import json

import redis.asyncio as redis
from structlog import get_logger

from ..domain import TaggedKeyframe

logger = get_logger(__name__)

SCENE_SAMPLES_STREAM = "scene.samples"
SCENE_CONSUMER_GROUP = "scene-worker"
DEFAULT_MAXLEN = 20000


class SceneSamplesPublisher:
    """Publishes TaggedKeyframe messages to the scene.samples Redis Stream.

    Usage::

        publisher = SceneSamplesPublisher(redis_url="redis://localhost:6379/0")
        await publisher.connect()
        await publisher.publish(keyframe)
        await publisher.disconnect()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        stream: str = SCENE_SAMPLES_STREAM,
        maxlen: int = DEFAULT_MAXLEN,
    ) -> None:
        self._redis_url = redis_url
        self._stream = stream
        self._maxlen = maxlen
        self._redis: redis.Redis | None = None

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        """Connect to Redis and create the consumer group if needed."""
        if self._redis is not None:
            return

        self._redis = redis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

        try:
            await self._redis.xgroup_create(
                self._stream,
                SCENE_CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info("Created scene.samples consumer group", stream=self._stream)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            logger.info("scene.samples consumer group already exists", stream=self._stream)

        logger.info("Connected to Redis for scene.samples", url=self._redis_url)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis (scene.samples)")

    async def publish(self, keyframe: TaggedKeyframe) -> str:
        """Publish one TaggedKeyframe to the scene.samples stream.

        Returns the Redis message ID, or "" if not connected.
        """
        if self._redis is None:
            logger.error("Cannot publish scene sample: not connected")
            return ""

        payload: dict[str, str] = {
            "keyframe_id": keyframe.keyframe_id,
            "tracklet_id": keyframe.tracklet_id,
            "global_track_id": keyframe.global_track_id,
            "camera_id": keyframe.camera_id,
            "minio_key": keyframe.minio_key,
            "captured_at": keyframe.captured_at.isoformat(),
            "tag_reason": keyframe.tag_reason,
            "annotations": json.dumps(keyframe.annotations),
            "expires_at": keyframe.expires_at.isoformat(),
        }

        message_id = str(
            await self._redis.xadd(
                self._stream,
                payload,  # type: ignore[arg-type]
                maxlen=self._maxlen,
                approximate=True,
            )
        )

        logger.debug(
            "Published scene sample",
            keyframe_id=keyframe.keyframe_id,
            camera_id=keyframe.camera_id,
            tag_reason=keyframe.tag_reason,
            message_id=message_id,
        )
        return message_id
