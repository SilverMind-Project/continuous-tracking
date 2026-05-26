"""Keyframe stage: samples keyframes and publishes to scene.samples.

M3 fix: only include bbox annotations for tracklets that had a detection
this frame. Stale tracklets (alive via grace window but no current detection)
are excluded, preventing bboxes drawn against empty space.
"""

from __future__ import annotations

from ...domain import TaggedKeyframe
from ...observability import metrics as _metrics
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
        min_keyframe_detection_confidence: float = 0.5,
    ) -> None:
        self._keyframe_sampler = keyframe_sampler
        self._scene_publisher = scene_publisher
        self._min_det_conf = min_keyframe_detection_confidence

    async def run(self, ctx: FrameContext) -> None:
        if not self._keyframe_sampler or not ctx.active_tracklets:
            return

        # Build a lookup of tracklet_id → detection for current-frame detections.
        det_by_tracklet: dict[str, object] = {}
        for det in ctx.domain_detections:
            if det.tracklet_id:
                det_by_tracklet[det.tracklet_id] = det

        sample_time = ctx.event_time
        revised_gt_ids = {rev.global_track_id for rev in ctx.new_revisions}
        for tracklet in ctx.active_tracklets:
            detection = det_by_tracklet.get(tracklet.tracklet_id)
            if detection is None:
                # M3: Tracklet is alive via grace window but no detection this
                # frame. Skip bbox annotation entirely — do not draw a stale
                # box against the current image.
                continue

            # M3: gate on detection confidence.
            det_conf = float(getattr(detection, "confidence", 1.0))
            if det_conf < self._min_det_conf:
                _metrics.metrics.keyframe_dropped_low_confidence_total.inc()
                continue

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

            # Use the detection bbox (current-frame), NOT tracklet.last_bbox.
            det_bbox = getattr(detection, "bbox", None)
            if det_bbox is not None:
                annotations["bbox"] = {
                    "x_min": float(det_bbox.x_min),
                    "y_min": float(det_bbox.y_min),
                    "x_max": float(det_bbox.x_max),
                    "y_max": float(det_bbox.y_max),
                }
                bbox_data: tuple[float, float, float, float] | None = (
                    float(det_bbox.x_min),
                    float(det_bbox.y_min),
                    float(det_bbox.x_max),
                    float(det_bbox.y_max),
                )
            else:
                bbox_data = None

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
                    detection_confidence=det_conf,
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
                    detection_confidence=det_conf,
                    detection_frame_width=ctx.effective_width,
                    detection_frame_height=ctx.effective_height,
                    detection_identity_id=identity_id or None,
                )
            if sampled is not None and self._scene_publisher:
                await self._scene_publisher.publish(sampled)
