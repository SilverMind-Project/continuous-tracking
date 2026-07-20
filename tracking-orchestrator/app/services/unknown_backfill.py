"""UnknownBackfillService: retroactive attribution of Unknown PH history.

When a previously Unknown Person Hypothesis receives its first calibrated
direct-face commit, this service records an inferred revision range covering
the PH's Unknown history, relabels the identity-NULL ``person_trajectories``
and ``room_dwells`` rows for that window, and publishes one
``inferred_backfill`` :class:`~app.domain.IdentityRevision` for
cognitive-companion to project (identity-continuity M05).

Safety invariant (normative): this service only ever fills NULL identity. It
never changes a non-NULL identity value, and it never crosses an operator
range, an earlier decision naming a different identity, or the configured cap.

Implements identity-continuity M04 (decision D5). See
``identity-continuity-m04-unknown-backfill-cts.md`` for the full design and
the 2026-07-20 dated corrections (in particular: initial commits already
persist a provenance decision row today, so this service does not write one).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from structlog import get_logger

from ..domain import (
    IdentityDecision,
    IdentityEvidence,
    IdentityRevision,
    IdentityRevisionJob,
    ProjectionAck,
)
from ..observability import metrics as _metrics
from ..storage.base import IdentityDecisionRepositoryProtocol, PHRepositoryProtocol
from ..tracking.identity.types import IdentityAuthority
from .identity_correction_service import CorrectionConflictError, IdentityCorrectionService
from .identity_rewriter import IdentityRewriter

logger = get_logger(__name__)

CTS_INTERNAL = "cts_internal"
CC_PROJECTION = "cc"

# Revision schema version stamped on published inferred_backfill revisions and
# on the cts_internal projection ack.
BACKFILL_SCHEMA_VERSION = "1"

# One backfill attempt per (ph_id, identity_id) per this many seconds. A PH
# commits from Unknown once in the ordinary case; a second Unknown-to-X
# transition on the same PH after a demotion is possible (contradiction,
# operator "Set to Unknown", etc.), so the limiter is keyed on the pair, not
# just the PH, with a short TTL rather than a per-process single-shot.
_RATE_LIMIT_TTL_S = 600.0


@dataclass(frozen=True)
class BackfillConfig:
    """Tunables for :class:`UnknownBackfillService`.

    Constructed in the composition root from the corresponding
    ``ResolverConfig`` fields (``enable_unknown_backfill``, ``backfill_shadow``,
    ``backfill_max_range_s``) rather than read from settings.yaml directly; see
    the M04 milestone doc's 2026-07-20 dated correction for why the raw keys
    live on ``ResolverConfig`` instead of a standalone settings section.
    """

    enabled: bool = False
    shadow: bool = True
    max_range_s: float = 14400.0


@dataclass(frozen=True)
class BackfillRange:
    """A computed, already-clipped backfill window."""

    start: datetime
    end: datetime
    clip_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SkipReason:
    """Why a candidate commit did not produce a backfill."""

    reason: str


def _trigger_matches(decision: IdentityDecision) -> bool:
    """Trigger predicate (M04 design): first calibrated direct-face commit.

    - ``previous_identity_id is None``: the PH had no identity before this
      decision (a first commit, not a change or a demotion).
    - ``identity_id`` is a real identity: not ``None`` and not the UNKNOWN
      sentinel.
    - ``authority == DIRECT_FACE``: only the calibrated ArcFace-authority
      commit path qualifies. Posterior commits do not backfill in v1: the
      backfilled label inherits the full trust of the segment, so only the
      strongest evidence class may write history.
    """
    if decision.previous_identity_id is not None:
        return False
    if not decision.identity_id or decision.identity_id == "UNKNOWN":
        return False
    return decision.authority == str(IdentityAuthority.DIRECT_FACE)


class UnknownBackfillService:
    """Owns the Unknown-segment backfill trigger, range computation, and writes.

    Constructor-injected; ``RevisionsStage`` calls :meth:`process` once per
    frame with the frame's outcome decisions and stays otherwise unaware of
    backfill logic. ``process`` takes plain typed arguments rather than a
    ``FrameContext`` to avoid a ``services -> pipeline`` import that would
    invert the ``core -> domain -> storage -> services -> transport ->
    routers`` layering (pipeline stages import services, never the reverse).
    """

    def __init__(
        self,
        *,
        ph_repo: PHRepositoryProtocol,
        identity_decision_repo: IdentityDecisionRepositoryProtocol,
        correction_service: IdentityCorrectionService,
        identity_rewriter: IdentityRewriter,
        revision_publisher: object | None,
        config: BackfillConfig,
    ) -> None:
        self._ph_repo = ph_repo
        self._decision_repo = identity_decision_repo
        self._correction_service = correction_service
        self._rewriter = identity_rewriter
        self._publisher = revision_publisher
        self._cfg = config
        # (ph_id, identity_id) -> last attempt monotonic time.
        self._rate_limit: dict[tuple[str, str], float] = {}

    async def process(
        self,
        *,
        outcome_decisions: list[IdentityDecision],
        ph_born_at_by_id: dict[str, datetime],
        event_time: datetime,
    ) -> None:
        """Scan a frame's outcome decisions for qualifying backfill triggers."""
        if not self._cfg.enabled:
            return

        for decision in outcome_decisions:
            if not _trigger_matches(decision):
                continue
            await self._handle_candidate(decision, ph_born_at_by_id, event_time)

    async def _handle_candidate(
        self,
        decision: IdentityDecision,
        ph_born_at_by_id: dict[str, datetime],
        commit_time: datetime,
    ) -> None:
        ph_id = decision.ph_id
        identity_id = decision.identity_id
        assert identity_id is not None  # narrowed by _trigger_matches

        if not self._check_rate_limit(ph_id, identity_id):
            return

        born_at = ph_born_at_by_id.get(ph_id)
        if born_at is None:
            ph = await self._ph_repo.get(ph_id)
            if ph is None:
                self._record_skip("no_birth_time")
                return
            born_at = ph.born_at

        range_or_skip = await self._compute_range(ph_id, identity_id, born_at, commit_time)
        if isinstance(range_or_skip, SkipReason):
            self._record_skip(range_or_skip.reason)
            return

        backfill_range = range_or_skip
        for reason in backfill_range.clip_reasons:
            _metrics.metrics.identity_backfill_clip_total.labels(reason=reason).inc()
        range_seconds = (backfill_range.end - backfill_range.start).total_seconds()
        _metrics.metrics.identity_backfill_range_seconds.observe(range_seconds)

        if self._cfg.shadow:
            _metrics.metrics.identity_backfill_total.labels(mode="shadow", outcome="shadow").inc()
            logger.info(
                "unknown_backfill_shadow",
                ph_id=ph_id,
                identity_id=identity_id,
                range_start=backfill_range.start.isoformat(),
                range_end=backfill_range.end.isoformat(),
                clip_reasons=list(backfill_range.clip_reasons),
            )
            return

        await self._apply(ph_id, identity_id, backfill_range)

    def _check_rate_limit(self, ph_id: str, identity_id: str) -> bool:
        key = (ph_id, identity_id)
        now = time.monotonic()
        last = self._rate_limit.get(key)
        if last is not None and (now - last) < _RATE_LIMIT_TTL_S:
            return False
        self._rate_limit[key] = now
        # Opportunistic cleanup so the dict does not grow unbounded across a
        # long process lifetime.
        stale = [k for k, ts in self._rate_limit.items() if (now - ts) >= _RATE_LIMIT_TTL_S]
        for k in stale:
            self._rate_limit.pop(k, None)
        return True

    async def _compute_range(
        self,
        ph_id: str,
        new_identity_id: str,
        born_at: datetime,
        commit_time: datetime,
    ) -> BackfillRange | SkipReason:
        start = born_at
        clip_reasons: list[str] = []

        # 2. Clip forward past the end of the latest earlier decision on this
        # PH naming a *different* identity.
        decisions, _total = await self._decision_repo.get_by_ph_id(ph_id, limit=50, offset=0)
        prior_conflicting = [
            d
            for d in decisions
            if d.captured_at < commit_time
            and d.inferred_identity_id
            and d.inferred_identity_id != new_identity_id
        ]
        if prior_conflicting:
            latest = max(prior_conflicting, key=lambda d: d.captured_at)
            if latest.captured_at > start:
                start = latest.captured_at
                clip_reasons.append("clipped_prior_decision")

        # 3. Clip forward past the end of any overlapping operator range; skip
        # entirely if an operator range owns the live edge (commit_time).
        correction_repo = self._correction_service.correction_repo
        operator_ranges = await correction_repo.operator_ranges_overlapping(
            ph_id, start, commit_time
        )
        if operator_ranges:
            if any(r.range_start <= commit_time <= r.range_end for r in operator_ranges):
                return SkipReason("operator_owns_live_edge")
            latest_end = max(r.range_end for r in operator_ranges)
            if latest_end > start:
                start = latest_end
                clip_reasons.append("clipped_operator_range")

        # 4. Cap the maximum span.
        cap_start = commit_time - timedelta(seconds=self._cfg.max_range_s)
        if cap_start > start:
            start = cap_start
            clip_reasons.append("clipped_cap")

        # 5. Empty range check.
        if start >= commit_time:
            return SkipReason("empty_range")

        return BackfillRange(start=start, end=commit_time, clip_reasons=tuple(clip_reasons))

    async def _apply(self, ph_id: str, new_identity_id: str, r: BackfillRange) -> None:
        revision_id = str(uuid.uuid4())

        try:
            await self._correction_service.record_inferred_range(
                ph_id=ph_id,
                revision_id=revision_id,
                effective_identity_id=new_identity_id,
                start=r.start,
                end=r.end,
            )
        except CorrectionConflictError:
            _metrics.metrics.identity_backfill_total.labels(
                mode="enabled", outcome="operator_conflict"
            ).inc()
            return

        await self._rewriter.backfill_null_rows(
            revision_id,
            ph_id,
            new_identity_id,
            r.start,
            r.end,
        )

        correction_repo = self._correction_service.correction_repo
        job = IdentityRevisionJob(
            job_id=str(uuid.uuid4()),
            revision_id=revision_id,
            status="applying",
            required_projections=(CTS_INTERNAL, CC_PROJECTION),
        )
        await correction_repo.save_job(job)
        await correction_repo.record_ack(
            ProjectionAck(
                revision_id=revision_id,
                consumer=CTS_INTERNAL,
                schema_version=BACKFILL_SCHEMA_VERSION,
                status="acked",
                applied_at=datetime.now(UTC),
            )
        )
        await correction_repo.complete_job_if_ready(revision_id)

        revision = IdentityRevision(
            revision_id=revision_id,
            ph_id=ph_id,
            previous_identity_id=None,
            new_identity_id=new_identity_id,
            actor="system",
            reason="unknown_backfill",
            applied_at=r.end,
            rewritten_rows=0,
            evidence=IdentityEvidence(evidence_sources=["direct_face"]),
            revision_kind="inferred_backfill",
            range_start=r.start,
            range_end=r.end,
            range_authority="inferred",
            revision_range_id=revision_id,
            required_projections=(CC_PROJECTION,),
            revision_schema_version=BACKFILL_SCHEMA_VERSION,
        )
        if self._publisher is not None:
            await self._publisher.publish_many([revision])  # type: ignore[attr-defined]

        _metrics.metrics.identity_backfill_total.labels(mode="enabled", outcome="applied").inc()
        logger.info(
            "unknown_backfill_applied",
            ph_id=ph_id,
            identity_id=new_identity_id,
            revision_id=revision_id,
            range_start=r.start.isoformat(),
            range_end=r.end.isoformat(),
        )

    def _record_skip(self, reason: str) -> None:
        _metrics.metrics.identity_backfill_total.labels(
            mode="shadow" if self._cfg.shadow else "enabled",
            outcome=f"skipped_{reason}",
        ).inc()
