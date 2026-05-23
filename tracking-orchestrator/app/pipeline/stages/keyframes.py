"""Keyframe stage: samples keyframes and publishes to scene.samples."""

from __future__ import annotations

from ...domain import TaggedKeyframe
from ...sampling.keyframe_sampler import KeyframeSampler
from ...transport.scene_publisher import SceneSamplesPublisher
from ..frame_context import FrameContext
from .base import FrameStage


class KeyframeStage(FrameStage):
    name = "keyframes"

    def __init__(
        self,
        keyframe_sampler: KeyframeSampler | None = None,
        scene_publisher: SceneSamplesPublisher | None = None,
    ) -> None:
        self._keyframe_sampler = keyframe_sampler
        self._scene_publisher = scene_publisher

    async def run(self, ctx: FrameContext) -> None:
        if not self._keyframe_sampler or not ctx.active_tracklets:
            return

        sample_time = ctx.event_time
        revised_gt_ids = {rev.global_track_id for rev in ctx.new_revisions}
        for tracklet in ctx.active_tracklets:
            gt_id = next(
                (
                    gt.global_track_id
                    for gt in ctx.active_global_tracks
                    if tracklet.tracklet_id in gt.tracklet_ids
                ),
                tracklet.tracklet_id,
            )
            identity_id = ctx.committed_ids.get(gt_id, "") or ""
            annotations: dict[str, object] = {
                "tracklet_id": tracklet.tracklet_id,
                "camera_id": tracklet.camera_id,
                "identity_id": identity_id or "",
            }
            if tracklet.last_bbox is not None:
                annotations["bbox"] = {
                    "x_min": tracklet.last_bbox.x_min,
                    "y_min": tracklet.last_bbox.y_min,
                    "x_max": tracklet.last_bbox.x_max,
                    "y_max": tracklet.last_bbox.y_max,
                }

            bbox_data: tuple[float, float, float, float] | None = None
            if tracklet.last_bbox is not None:
                bbox_data = (
                    float(tracklet.last_bbox.x_min),
                    float(tracklet.last_bbox.y_min),
                    float(tracklet.last_bbox.x_max),
                    float(tracklet.last_bbox.y_max),
                )

            sampled: TaggedKeyframe | None
            if gt_id in revised_gt_ids:
                sampled = await self._keyframe_sampler.trigger_sample(
                    tracklet_id=tracklet.tracklet_id,
                    global_track_id=gt_id,
                    camera_id=tracklet.camera_id,
                    minio_key=ctx.frame.minio_key,
                    captured_at=sample_time,
                    annotations=annotations,
                    tag_reason="identity_changed",
                    detection_bbox=bbox_data,
                    detection_confidence=1.0,
                    detection_frame_width=ctx.effective_width,
                    detection_frame_height=ctx.effective_height,
                    detection_identity_id=identity_id or None,
                )
            else:
                sampled = await self._keyframe_sampler.maybe_sample(
                    tracklet_id=tracklet.tracklet_id,
                    global_track_id=gt_id,
                    camera_id=tracklet.camera_id,
                    minio_key=ctx.frame.minio_key,
                    captured_at=sample_time,
                    annotations=annotations,
                    detection_bbox=bbox_data,
                    detection_confidence=1.0,
                    detection_frame_width=ctx.effective_width,
                    detection_frame_height=ctx.effective_height,
                    detection_identity_id=identity_id or None,
                )
            if sampled is not None and self._scene_publisher:
                await self._scene_publisher.publish(sampled)
