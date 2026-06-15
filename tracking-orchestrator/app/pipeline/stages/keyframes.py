"""Keyframe stage: samples keyframes and publishes to scene.samples.

Rewired to consume WorldFrameSnapshot instead of active_tracklets.
Each open PH with a detection this frame is a candidate for periodic
or identity-change-triggered keyframe sampling.
"""

from __future__ import annotations

from ...domain import TaggedKeyframe
from ...observability import metrics as _metrics
from ...sampling.keyframe_sampler import FrameBbox, KeyframeSampler
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
        if not self._keyframe_sampler or not ctx.world_snapshots:
            return

        revised_ph_ids = {rev.ph_id for rev in ctx.new_revisions}
        sample_time = ctx.event_time

        det_by_ph: dict[str, object] = {d.ph_id: d for d in ctx.domain_detections if d.ph_id}

        # A keyframe image contains every person visible in the frame, so each
        # sampled keyframe carries one bbox per detection -- not just the bbox
        # of the PH that triggered the sample. Build this set once per frame.
        identity_by_ph: dict[str, str] = {
            snap.ph_id: (snap.identity_id or "")
            for snap in ctx.world_snapshots
            if snap.camera_id == ctx.frame.camera_id
        }
        frame_bboxes: list[FrameBbox] = []
        for det in det_by_ph.values():
            det_bbox = getattr(det, "bbox", None)
            if det_bbox is None:
                continue
            ph_id = getattr(det, "ph_id", "")
            frame_bboxes.append(
                FrameBbox(
                    ph_id=ph_id,
                    bbox=(
                        float(det_bbox.x_min),
                        float(det_bbox.y_min),
                        float(det_bbox.x_max),
                        float(det_bbox.y_max),
                    ),
                    confidence=float(getattr(det, "confidence", 1.0)),
                    identity_id=identity_by_ph.get(ph_id) or None,
                )
            )

        for snap in ctx.world_snapshots:
            if snap.camera_id != ctx.frame.camera_id:
                continue
            detection = det_by_ph.get(snap.ph_id)
            if detection is None:
                continue

            det_conf = float(getattr(detection, "confidence", 1.0))
            if det_conf < self._min_det_conf:
                _metrics.metrics.keyframe_dropped_low_confidence_total.inc()
                continue

            identity_id = snap.identity_id or ""
            annotations: dict[str, object] = {
                "ph_id": snap.ph_id,
                "camera_id": snap.camera_id,
                "identity_id": identity_id,
                "detection_confidence": det_conf,
                "frame_width": ctx.effective_width,
                "frame_height": ctx.effective_height,
            }

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
            if snap.ph_id in revised_ph_ids:
                sampled = await self._keyframe_sampler.trigger_sample(
                    ph_id=snap.ph_id,
                    camera_id=snap.camera_id,
                    minio_key=ctx.frame.minio_key,
                    captured_at=sample_time,
                    annotations=annotations,
                    tag_reason="identity_changed",
                    detection_bbox=bbox_data,
                    detection_confidence=det_conf,
                    detection_frame_width=ctx.effective_width,
                    detection_frame_height=ctx.effective_height,
                    detection_identity_id=identity_id or None,
                    frame_bboxes=frame_bboxes,
                )
            else:
                sampled = await self._keyframe_sampler.maybe_sample(
                    ph_id=snap.ph_id,
                    camera_id=snap.camera_id,
                    minio_key=ctx.frame.minio_key,
                    captured_at=sample_time,
                    annotations=annotations,
                    detection_bbox=bbox_data,
                    detection_confidence=det_conf,
                    detection_frame_width=ctx.effective_width,
                    detection_frame_height=ctx.effective_height,
                    detection_identity_id=identity_id or None,
                    frame_bboxes=frame_bboxes,
                )
            if sampled is not None and self._scene_publisher:
                await self._scene_publisher.publish(sampled)
