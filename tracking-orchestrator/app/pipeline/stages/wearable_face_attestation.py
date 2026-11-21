"""Hands each context's emitted faces of wearable-attested PHs to the attestation service.

Runs after world tracking (it needs ``det_to_ph`` and the round's wearable evidence) and ahead
of ``PublishStage``, on every context of a round: each context carries its own detections and
faces, while the evidence is per PH and present on every context.
"""

from __future__ import annotations

from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.base import FrameStage
from app.services.wearable_face_attestation import WearableFaceAttestationService


class WearableFaceAttestationStage(FrameStage):
    """Offers emitted faces of attested PHs for operator-gated sensor attestation."""

    name = "wearable_face_attestation"

    def __init__(self, service: WearableFaceAttestationService) -> None:
        self._service = service

    async def run(self, ctx: FrameContext) -> None:
        # Called on every round, evidence or not: a batch opened while an association was
        # attested must still go out after the association ends, before person-id's hold expires.
        await self._service.offer_round(
            evidence_by_ph=ctx.wearable_evidence_by_ph,
            det_to_ph=ctx.det_to_ph,
            emitted_faces=ctx.face_anchors,
            shadow_faces=ctx.shadow_face_anchors,
            captured_at=ctx.capture_time,
        )
