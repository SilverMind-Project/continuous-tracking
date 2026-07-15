"""Provenance-persist stage: durably writes identity decisions every round.

Runs ahead of ``PublishStage``'s throttle so a decision is never dropped
because the frame that carried it was rate-limited for the UI-facing
Redis publish. See ``codebase-hardening-m02-provenance-persistence-decoupling.md``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from structlog import get_logger

from ...domain import IdentityProvenanceDecision
from ...storage.base import IdentityDecisionRepositoryProtocol
from ..frame_context import FrameContext
from .base import FrameStage

logger = get_logger(__name__)


class ProvenancePersistStage(FrameStage):
    name = "provenance_persist"

    def __init__(self, identity_provenance_repo: IdentityDecisionRepositoryProtocol) -> None:
        self._identity_provenance_repo = identity_provenance_repo
        # In-flight provenance writes; holds references so fire-and-forget save
        # tasks are not garbage-collected before they complete.
        self._pending_saves: set[asyncio.Task[None]] = set()

    async def run(self, ctx: FrameContext) -> None:
        if not ctx.outcome_decisions:
            return

        for decision in ctx.outcome_decisions:
            # Persist at identity change points: initial commits, swaps, and
            # conflict-to-unknown transitions all set ``revises_previous``.
            # Held rounds carry no new provenance; the keyframe read model
            # resolves a held identity from its last persisted decision.
            if not decision.revises_previous:
                continue

            _top_id, top_prob = decision.posterior.top_identity()
            top_probs = sorted(decision.posterior.distribution.values(), reverse=True)
            top2_prob = top_probs[1] if len(top_probs) > 1 else 0.0

            last_independent_at: datetime | None = None
            if decision.last_independent_evidence_at_unix_ns:
                last_independent_at = datetime.fromtimestamp(
                    decision.last_independent_evidence_at_unix_ns / 1e9, tz=UTC
                )

            # IdentityProvenanceDecision is a frozen dataclass; build every
            # field in the constructor rather than mutating after creation.
            prov = IdentityProvenanceDecision(
                decision_id=decision.decision_id or "unknown",
                ph_id=decision.ph_id,
                captured_at=ctx.capture_time,
                authority=decision.authority,
                decision_source=decision.decision_source,
                diagnostics=decision.evidence or {},
                inferred_identity_id=decision.inferred_identity_id,
                effective_identity_id=decision.effective_identity_id,
                conflict_kind=decision.conflict,
                top_probability=top_prob,
                second_probability=top2_prob,
                posterior_entropy=decision.posterior.entropy(),
                last_independent_evidence_at=last_independent_at,
                config_hash=decision.config_hash,
                model_set_version=decision.model_set_version,
            )

            # Keep a strong reference until the write completes; a bare
            # create_task may be garbage-collected mid-flight, silently
            # dropping the decision write.
            task = asyncio.create_task(self._identity_provenance_repo.save(prov))
            self._pending_saves.add(task)
            task.add_done_callback(self._pending_saves.discard)

        if ctx.new_revisions:
            logger.info(
                "Identity revisions emitted",
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
                revision_count=len(ctx.new_revisions),
            )

    async def aclose(self) -> None:
        """Await in-flight provenance saves so shutdown does not abandon them."""
        if self._pending_saves:
            await asyncio.gather(*self._pending_saves, return_exceptions=True)
