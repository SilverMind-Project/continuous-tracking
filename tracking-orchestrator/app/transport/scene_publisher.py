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

from structlog import get_logger

from ..domain import TaggedKeyframe
from .base_publisher import BasePublisher

logger = get_logger(__name__)


class SceneSamplesPublisher(BasePublisher):
    """Publishes TaggedKeyframe messages to the scene.samples Redis Stream.

    Usage::

        publisher = SceneSamplesPublisher(redis_url="redis://localhost:6379/0")
        await publisher.connect()
        await publisher.publish(keyframe)
        await publisher.disconnect()
    """

    _stream_name = "scene.samples"
    _default_maxlen = 20000

    async def publish(self, keyframe: TaggedKeyframe) -> str:
        """Publish one TaggedKeyframe to the scene.samples stream.

        Returns the Redis message ID, or "" if not connected.
        """
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

        message_id = await self._xadd(payload)

        logger.debug(
            "Published scene sample",
            keyframe_id=keyframe.keyframe_id,
            camera_id=keyframe.camera_id,
            tag_reason=keyframe.tag_reason,
            message_id=message_id,
        )
        return message_id
