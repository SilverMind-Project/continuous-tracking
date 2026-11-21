"""Hands each tracker round's image-only evaluations to the evaluation service.

Runs once per round (the evaluations ride the primary context only, like wearable segment
transitions) ahead of ``PublishStage``, so its accounting never sits behind the publish
throttle. Every round is passed on, including rounds without evaluations, because the service
flushes on capture time.
"""

from __future__ import annotations

from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.base import FrameStage
from app.services.image_only_evaluator import ImageOnlyEvaluationSink


class ImageOnlyEvaluationStage(FrameStage):
    """Records sampled image-only evaluations of wearable-attested PHs."""

    name = "image_only_evaluation"

    def __init__(self, sink: ImageOnlyEvaluationSink) -> None:
        self._sink = sink

    async def run(self, ctx: FrameContext) -> None:
        await self._sink.record_round(ctx.image_only_evaluations, captured_at=ctx.capture_time)
