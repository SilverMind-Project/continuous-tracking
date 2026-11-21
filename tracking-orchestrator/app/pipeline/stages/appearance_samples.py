"""Pipeline stage that captures labeled body-appearance samples after world tracking.

Runs in ``post_world_stages`` ahead of ``PublishStage`` across all three execution
paths (direct, per-camera batch, cross-camera batch) so training sample capture is
never gated by the live UI publish throttle.
"""

from __future__ import annotations

from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.base import FrameStage
from app.services.appearance_sample_recorder import AppearanceSampleRecorder


class AppearanceSampleRecorderStage(FrameStage):
    """Records eligible body embeddings into ``appearance_training_samples``."""

    name = "appearance_sample_recorder"

    def __init__(self, recorder: AppearanceSampleRecorder) -> None:
        self._recorder = recorder

    async def run(self, ctx: FrameContext) -> None:
        await self._recorder.record_frame(
            detections=ctx.domain_detections,
            det_to_ph=ctx.det_to_ph,
            orientation_by_detection=ctx.orientation_by_detection,
            decisions=ctx.outcome_decisions,
            committed_ids=ctx.committed_ids,
            wearable_evidence_by_ph=ctx.wearable_evidence_by_ph,
            face_anchors=ctx.face_anchors,
            shadow_face_anchors=ctx.shadow_face_anchors,
        )
