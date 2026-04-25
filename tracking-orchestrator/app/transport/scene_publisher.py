"""SceneSamplesPublisher: emits SceneSample proto messages to scene.samples.

The scene.samples Redis Stream is consumed by the scene worker
(``CTSSceneWorker`` in cognitive-companion), which fetches the JPEG from
MinIO and runs scene analysis on tagged frames.

Wire format: each Redis Streams message is a single field ``sample``
carrying the raw protobuf body of a
``continuoustracking.v1.SceneSample``.
"""

from __future__ import annotations

import json

from structlog import get_logger

from ..domain import TaggedKeyframe
from ..observability import metrics
from ..proto.continuoustracking.v1 import scene_pb2
from .base_publisher import BasePublisher

logger = get_logger(__name__)

FIELD = b"sample"


_TAG_REASON_TO_PROTO: dict[str, int] = {
    "periodic": scene_pb2.TAG_REASON_PERIODIC,
    "identity_changed": scene_pb2.TAG_REASON_IDENTITY_CHANGED,
    "hazard": scene_pb2.TAG_REASON_HAZARD,
    "dwell_start": scene_pb2.TAG_REASON_DWELL_START,
    "fall": scene_pb2.TAG_REASON_FALL,
    "dementia_signal": scene_pb2.TAG_REASON_DEMENTIA_SIGNAL,
}


class SceneSamplesPublisher(BasePublisher):
    """Publishes SceneSample proto messages to ``scene.samples``."""

    _stream_name = "scene.samples"
    _default_maxlen = 20000

    async def publish(self, keyframe: TaggedKeyframe) -> str:
        """Publish one TaggedKeyframe as a SceneSample proto."""
        message = _to_proto(keyframe)
        message_id = await self._xadd({FIELD: message.SerializeToString()})

        metrics.metrics.scene_samples_published_total.labels(reason=keyframe.tag_reason).inc()

        logger.debug(
            "Published scene sample",
            keyframe_id=keyframe.keyframe_id,
            camera_id=keyframe.camera_id,
            tag_reason=keyframe.tag_reason,
            message_id=message_id,
        )
        return message_id


def _to_proto(keyframe: TaggedKeyframe) -> scene_pb2.SceneSample:
    pb = scene_pb2.SceneSample()
    pb.keyframe_id = keyframe.keyframe_id
    pb.tracklet_id = keyframe.tracklet_id
    pb.global_track_id = keyframe.global_track_id
    pb.camera_id = keyframe.camera_id
    pb.minio_key = keyframe.minio_key
    pb.captured_at_unix_ns = int(keyframe.captured_at.timestamp() * 1e9)
    # Proto enum values are plain ints at runtime; the generated stub types
    # the attribute as the enum class so mypy rejects direct assignment.
    setattr(  # noqa: B010
        pb,
        "tag_reason",
        _TAG_REASON_TO_PROTO.get(keyframe.tag_reason, scene_pb2.TAG_REASON_UNSPECIFIED),
    )
    pb.annotations_json = json.dumps(keyframe.annotations, default=str)
    pb.expires_at_unix_ns = int(keyframe.expires_at.timestamp() * 1e9)
    return pb
