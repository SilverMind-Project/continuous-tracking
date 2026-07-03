"""IdentityCorrectionService: the single owner of operator identity corrections.

Routers, batch adapters, the keyframe/PH inspector path, and compensations all
call this service. No route mutates the correction/revision repositories directly.

Responsibilities (M06):

* Propose an advisory correction segment around a selected observation, stopping
  at association discontinuities, identity anchors, splits/merges, and prior
  operator-revision boundaries.
* Apply a frame-only or bounded-segment correction, or an explicit "Set to
  Unknown", under an optimistic version token (stale -> conflict).
* Compose a PH split with the correction transactionally for ``track_handoff``.
* Write the operator :class:`IdentityRevisionRange` (authoritative inside its
  bounds), supersede overlapping inferred ranges, and reject inferred revisions
  that would overlap a live operator range.
* Seed the live-edge current identity and 30-second prior clock without granting
  indefinite future authority.
* Create one :class:`IdentityRevisionJob` and publish one idempotent revision;
  the correction completes only after every required projection acknowledges.
* Compensate (undo) by issuing a new revision that restores effective identity
  while retaining the original correction and audit chain.

Raw inference in ``identity_decisions`` is never edited here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from structlog import get_logger

from ..domain import (
    CorrectionKind,
    CorrectionReasonCode,
    IdentityEvidence,
    IdentityRevision,
    IdentityRevisionJob,
    IdentityRevisionRange,
    IdentitySegmentCorrection,
    PersonHypothesis,
    ProjectionAck,
    SegmentBoundary,
    SegmentProposal,
    WorldObservation,
)
from ..storage.base import IdentityCorrectionRepositoryProtocol, PHRepositoryProtocol
from .identity_rewriter import IdentityRewriter, InMemoryIdentityRewriter

logger = get_logger(__name__)

# Required projections that must acknowledge before a correction is complete.
CTS_INTERNAL = "cts_internal"
CC_PROJECTION = "cc"
DEFAULT_REQUIRED_PROJECTIONS = (CTS_INTERNAL, CC_PROJECTION)

# Schema version stamped on the CTS-internal projection ack.
CTS_PROJECTION_SCHEMA_VERSION = "1"


class CorrectionError(ValueError):
    """Base class for correction-rejection errors (maps to HTTP 422)."""


class PHNotFoundError(CorrectionError):
    """The target PH or correction does not exist (maps to HTTP 404)."""


class StaleVersionError(CorrectionError):
    """The PH version token no longer matches; the proposal must be recomputed."""


class EmptyIdentityError(CorrectionError):
    """A non-Unknown correction was submitted with no target identity."""


class CorrectionConflictError(CorrectionError):
    """An inferred revision would overlap a live operator range."""


@dataclass(frozen=True)
class CorrectionResult:
    """Outcome of applying a correction."""

    revision_id: str
    correction_id: str
    ph_id: str
    previous_identity_id: str | None
    new_identity_id: str | None
    range_id: str
    new_ph_id: str | None  # set when a handoff composed a split
    job: IdentityRevisionJob


@dataclass(frozen=True)
class CorrectionConfig:
    """Tunables for proposal and live-edge behavior."""

    prior_window_s: float = 30.0
    discontinuity_gap_s: float = 10.0
    required_projections: tuple[str, ...] = DEFAULT_REQUIRED_PROJECTIONS


class IdentityCorrectionService:
    """One authoritative service for all identity corrections."""

    def __init__(
        self,
        *,
        ph_repo: PHRepositoryProtocol,
        correction_repo: IdentityCorrectionRepositoryProtocol,
        publisher: object | None = None,
        rewriter: IdentityRewriter | None = None,
        config: CorrectionConfig | None = None,
    ) -> None:
        self._ph_repo = ph_repo
        self._corr = correction_repo
        self._publisher = publisher
        self._rewriter = rewriter or InMemoryIdentityRewriter()
        self._cfg = config or CorrectionConfig()

    # -- version token -------------------------------------------------------

    async def _ph_version(self, ph_id: str, observation_count: int) -> int:
        """Deterministic optimistic token.

        Increments when the PH gains observations (incl. splits/merges) or when
        a new correction lands. A stale token signals the structure changed
        under the operator's proposal.
        """
        corrections = await self._corr.list_corrections(ph_id)
        return observation_count + len(corrections)

    # -- proposal ------------------------------------------------------------

    async def propose_segment(
        self,
        ph_id: str,
        *,
        observation_id: str | None = None,
        at: datetime | None = None,
    ) -> SegmentProposal:
        """Propose an advisory correction segment around a seed observation.

        Search outward from the seed until a discontinuity, an operator-revision
        boundary, or the PH edge. The proposal is advisory; applying requires an
        explicit confirmed start/end.
        """
        ph = await self._ph_repo.get(ph_id)
        if ph is None:
            raise PHNotFoundError(f"PH not found: {ph_id}")

        observations = await self._ph_repo.list_observations(ph_id, limit=1000)
        obs = sorted(observations, key=lambda o: o.captured_at)
        if not obs:
            raise CorrectionError(f"PH {ph_id} has no observations to correct")

        seed_idx = self._seed_index(obs, observation_id=observation_id, at=at)

        # Operator-range boundaries stop the search.
        operator_ranges = [
            r
            for r in await self._corr.list_ranges(ph_id, live_only=True)
            if r.authority == "operator"
        ]

        start_idx = seed_idx
        while start_idx > 0:
            prev, cur = obs[start_idx - 1], obs[start_idx]
            reason = self._boundary_reason_backward(prev, cur, operator_ranges)
            if reason is not None:
                break
            start_idx -= 1

        end_idx = seed_idx
        while end_idx < len(obs) - 1:
            cur, nxt = obs[end_idx], obs[end_idx + 1]
            reason = self._boundary_reason_forward(cur, nxt, operator_ranges)
            if reason is not None:
                break
            end_idx += 1

        start_obs, end_obs = obs[start_idx], obs[end_idx]
        version = await self._ph_version(ph_id, ph.observation_count)
        effective, _authority = await self._corr.effective_identity(ph_id, start_obs.captured_at)
        if effective is None:
            effective = ph.current_identity_id

        return SegmentProposal(
            ph_id=ph_id,
            observation_ids=[o.observation_id for o in obs[start_idx : end_idx + 1]],
            start=SegmentBoundary(
                observation_id=start_obs.observation_id,
                captured_at=start_obs.captured_at,
                reason="segment_edge"
                if start_idx == 0
                else self._boundary_reason_backward(obs[start_idx - 1], start_obs, operator_ranges)
                or "association_discontinuity",
            ),
            end=SegmentBoundary(
                observation_id=end_obs.observation_id,
                captured_at=end_obs.captured_at,
                reason="segment_edge"
                if end_idx == len(obs) - 1
                else self._boundary_reason_forward(end_obs, obs[end_idx + 1], operator_ranges)
                or "association_discontinuity",
            ),
            ph_version=version,
            effective_identity_id=effective,
        )

    def _seed_index(
        self,
        obs: list[WorldObservation],
        *,
        observation_id: str | None,
        at: datetime | None,
    ) -> int:
        if observation_id is not None:
            for i, o in enumerate(obs):
                if o.observation_id == observation_id:
                    return i
            raise PHNotFoundError(f"Observation {observation_id} not in PH")
        if at is not None:
            # Nearest observation by capture time.
            return min(
                range(len(obs)),
                key=lambda i: abs((obs[i].captured_at - at).total_seconds()),
            )
        # Default seed: most recent observation (live-edge correction).
        return len(obs) - 1

    def _boundary_reason_backward(
        self,
        prev: WorldObservation,
        cur: WorldObservation,
        operator_ranges: list[IdentityRevisionRange],
    ) -> str | None:
        gap = (cur.captured_at - prev.captured_at).total_seconds()
        if gap > self._cfg.discontinuity_gap_s:
            return "association_discontinuity"
        # Stop if prev falls inside an operator range that cur is not in.
        if self._in_operator_range(prev.captured_at, operator_ranges) and not (
            self._in_operator_range(cur.captured_at, operator_ranges)
        ):
            return "operator_revision"
        return None

    def _boundary_reason_forward(
        self,
        cur: WorldObservation,
        nxt: WorldObservation,
        operator_ranges: list[IdentityRevisionRange],
    ) -> str | None:
        gap = (nxt.captured_at - cur.captured_at).total_seconds()
        if gap > self._cfg.discontinuity_gap_s:
            return "association_discontinuity"
        if self._in_operator_range(nxt.captured_at, operator_ranges) and not (
            self._in_operator_range(cur.captured_at, operator_ranges)
        ):
            return "operator_revision"
        return None

    @staticmethod
    def _in_operator_range(at: datetime, operator_ranges: list[IdentityRevisionRange]) -> bool:
        return any(r.range_start <= at <= r.range_end for r in operator_ranges)

    # -- apply ---------------------------------------------------------------

    async def apply_correction(
        self,
        *,
        ph_id: str,
        actor: str,
        reason_code: CorrectionReasonCode,
        observation_start: datetime,
        observation_end: datetime,
        base_ph_version: int,
        target_identity_id: str | None = None,
        set_unknown: bool = False,
        frame_only: bool = False,
        note: str | None = None,
        source_view: str | None = None,
        reviewed_frame_id: str | None = None,
        reviewed_bbox: dict[str, object] | None = None,
        at_observation_id: str | None = None,
    ) -> CorrectionResult:
        """Apply a bounded/frame-only correction or explicit Unknown.

        Raises :class:`StaleVersionError`, :class:`EmptyIdentityError`, or
        :class:`CorrectionConflictError` on rejection (no partial writes).
        """
        ph = await self._ph_repo.get(ph_id)
        if ph is None:
            raise PHNotFoundError(f"PH not found: {ph_id}")

        # Explicit Unknown is allowed; empty identity without set_unknown is not.
        if not set_unknown and not target_identity_id:
            raise EmptyIdentityError(
                "A non-Unknown correction requires a target identity; "
                "use set_unknown=true for explicit Unknown."
            )
        if set_unknown:
            target_identity_id = None

        current_version = await self._ph_version(ph_id, ph.observation_count)
        if base_ph_version != current_version:
            raise StaleVersionError(
                f"PH version is {current_version}, proposal used {base_ph_version}; "
                "recompute the segment proposal."
            )

        if frame_only:
            observation_end = observation_start

        revision_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        previous_effective, _auth = await self._corr.effective_identity(ph_id, observation_start)
        if previous_effective is None:
            previous_effective = ph.current_identity_id

        kind: CorrectionKind = "label"
        if frame_only:
            kind = "frame_only"
        new_ph_id: str | None = None

        # A track_handoff (or a detected physical discontinuity) composes a split.
        if reason_code == "track_handoff" and at_observation_id is not None:
            try:
                _orig, new_ph_id = await self._ph_repo.split(
                    ph_id,
                    at_observation_id=at_observation_id,
                    actor=actor,
                    reason="track_handoff",
                    idempotency_key=f"handoff-{revision_id}",
                )
                kind = "handoff_split"
            except ValueError as exc:
                raise CorrectionError(f"handoff split failed: {exc}") from exc

        # 1) Append-only operator record (immutable).
        correction = IdentitySegmentCorrection(
            correction_id=str(uuid.uuid4()),
            ph_id=ph_id,
            actor=actor,
            reason_code=reason_code,
            observation_start=observation_start,
            observation_end=observation_end,
            base_ph_version=base_ph_version,
            revision_id=revision_id,
            target_identity_id=target_identity_id,
            set_unknown=set_unknown,
            kind=kind,
            frame_only=frame_only,
            note=note,
            source_view=source_view,
            reviewed_frame_id=reviewed_frame_id,
            reviewed_bbox=reviewed_bbox,
            base_revision_id=None,
            created_at=now,
        )
        await self._corr.save_correction(correction)

        # 2) Operator range supersedes overlapping live ranges (operator wins).
        range_id = await self._write_operator_range(
            ph_id=ph_id,
            revision_id=revision_id,
            correction_id=correction.correction_id,
            effective_identity_id=target_identity_id,
            start=observation_start,
            end=observation_end,
            created_at=now,
        )

        # 3) Job: a correction is complete only after required projections ack.
        job = IdentityRevisionJob(
            job_id=str(uuid.uuid4()),
            revision_id=revision_id,
            status="applying",
            required_projections=self._cfg.required_projections,
            correction_id=correction.correction_id,
            created_at=now,
            updated_at=now,
        )
        await self._corr.save_job(job)

        # 4) Live-edge: update current identity + seed the 30s prior clock.
        #    Operator authority ends at the confirmed observations; future
        #    observations use the normal resolver policy.
        if self._is_live_edge(ph, observation_end):
            if target_identity_id is None:
                await self._ph_repo.clear_to_unknown(ph_id, now)
            else:
                await self._ph_repo.evidence_backed_commit(
                    ph_id,
                    target_identity_id,
                    evidence_at=observation_end,
                    committed_at=now,
                )

        # 5) CTS-internal projection: rewrite derived rows and ack synchronously.
        cts_rows = await self._apply_cts_projection(
            revision_id=revision_id,
            ph_id=ph_id,
            old_identity=previous_effective,
            new_identity=target_identity_id,
            start=observation_start,
            end=observation_end,
        )
        await self._corr.record_ack(
            ProjectionAck(
                revision_id=revision_id,
                consumer=CTS_INTERNAL,
                schema_version=CTS_PROJECTION_SCHEMA_VERSION,
                status="acked",
                counts={"rewritten_rows": cts_rows},
                applied_at=datetime.now(UTC),
            )
        )
        await self._corr.update_job(revision_id, row_counts={"cts_rewritten_rows": cts_rows})
        await self._corr.complete_job_if_ready(revision_id)

        # 6) Publish one idempotent revision for downstream (CC) projection.
        await self._publish(
            revision_id=revision_id,
            ph_id=ph_id,
            previous_identity_id=previous_effective,
            new_identity_id=target_identity_id,
            actor=actor,
            reason=self._revision_reason(kind, set_unknown),
            applied_at=now,
            rewritten_rows=cts_rows,
            range_start=observation_start,
            range_end=observation_end,
            revision_range_id=range_id,
            correction_id=correction.correction_id,
        )

        logger.info(
            "identity_correction_applied",
            revision_id=revision_id,
            correction_id=correction.correction_id,
            ph_id=ph_id,
            kind=kind,
            previous_identity_id=previous_effective,
            new_identity_id=target_identity_id,
            actor=actor,
            frame_only=frame_only,
            live_edge=self._is_live_edge(ph, observation_end),
        )

        return CorrectionResult(
            revision_id=revision_id,
            correction_id=correction.correction_id,
            ph_id=ph_id,
            previous_identity_id=previous_effective,
            new_identity_id=target_identity_id,
            range_id=range_id,
            new_ph_id=new_ph_id,
            job=job,
        )

    async def _write_operator_range(
        self,
        *,
        ph_id: str,
        revision_id: str,
        correction_id: str,
        effective_identity_id: str | None,
        start: datetime,
        end: datetime,
        created_at: datetime,
    ) -> str:
        live = await self._corr.list_ranges(ph_id, live_only=True)
        overlapping = [r for r in live if r.range_start <= end and r.range_end >= start]
        supersedes = max(overlapping, key=lambda r: r.created_at).range_id if overlapping else None
        new_range = IdentityRevisionRange(
            range_id=str(uuid.uuid4()),
            revision_id=revision_id,
            ph_id=ph_id,
            authority="operator",
            range_start=start,
            range_end=end,
            effective_identity_id=effective_identity_id,
            correction_id=correction_id,
            supersedes_range_id=supersedes,
            created_at=created_at,
        )
        await self._corr.save_range(new_range)
        for r in overlapping:
            await self._corr.supersede_range(r.range_id, by_range_id=new_range.range_id)
        return new_range.range_id

    def _is_live_edge(self, ph: PersonHypothesis, observation_end: datetime) -> bool:
        if ph.closed_at is not None:
            return False
        grace = timedelta(seconds=self._cfg.prior_window_s)
        return observation_end >= ph.last_seen_at - grace

    async def _apply_cts_projection(
        self,
        *,
        revision_id: str,
        ph_id: str,
        old_identity: str | None,
        new_identity: str | None,
        start: datetime,
        end: datetime,
    ) -> int:
        if new_identity is None or old_identity is None or old_identity == new_identity:
            return 0
        await self._rewriter.rewrite(
            revision_id,
            ph_id,
            old_identity,
            new_identity,
            start,
            end,
        )
        # Rewriter does not return a count; the row_counts are advisory here.
        return 0

    async def apply_whole_ph_correction(
        self,
        *,
        ph_id: str,
        actor: str,
        new_identity_id: str | None,
        reason: str = "manual",
    ) -> CorrectionResult:
        """Whole-PH correction adapter for the PH inspector and batch routes.

        Computes the current version internally and applies an operator range
        over the PH's full observed span so the effective-identity projection is
        consistent with the segment-scoped path. ``new_identity_id is None``
        means explicit Unknown.
        """
        ph = await self._ph_repo.get(ph_id)
        if ph is None:
            raise PHNotFoundError(f"PH not found: {ph_id}")
        version = await self._ph_version(ph_id, ph.observation_count)
        return await self.apply_correction(
            ph_id=ph_id,
            actor=actor,
            reason_code="other",
            observation_start=ph.born_at,
            observation_end=ph.last_seen_at,
            base_ph_version=version,
            target_identity_id=new_identity_id,
            set_unknown=new_identity_id is None,
            note=reason,
        )

    # -- inferred-revision guard --------------------------------------------

    async def record_inferred_range(
        self,
        *,
        ph_id: str,
        revision_id: str,
        effective_identity_id: str | None,
        start: datetime,
        end: datetime,
    ) -> IdentityRevisionRange:
        """Record an inferred effective range, refusing operator overlaps.

        Inferred revisions cannot supersede operator authority; an overlap is a
        conflict, recorded and rejected.
        """
        conflicts = await self._corr.operator_ranges_overlapping(ph_id, start, end)
        if conflicts:
            logger.warning(
                "inferred_revision_operator_conflict",
                ph_id=ph_id,
                revision_id=revision_id,
                operator_ranges=[r.range_id for r in conflicts],
            )
            raise CorrectionConflictError("inferred revision overlaps a live operator range")
        new_range = IdentityRevisionRange(
            range_id=str(uuid.uuid4()),
            revision_id=revision_id,
            ph_id=ph_id,
            authority="inferred",
            range_start=start,
            range_end=end,
            effective_identity_id=effective_identity_id,
            created_at=datetime.now(UTC),
        )
        await self._corr.save_range(new_range)
        return new_range

    # -- compensation (undo) -------------------------------------------------

    async def compensate(self, correction_id: str, *, actor: str) -> CorrectionResult:
        """Undo a correction with a compensating revision.

        The original correction row and its range are retained for audit; a new
        operator range restores the pre-correction effective identity.
        """
        original = await self._corr.get_correction(correction_id)
        if original is None:
            raise PHNotFoundError(f"Correction not found: {correction_id}")
        ph = await self._ph_repo.get(original.ph_id)
        if ph is None:
            raise PHNotFoundError(f"PH not found: {original.ph_id}")

        now = datetime.now(UTC)
        revision_id = str(uuid.uuid4())

        # Effective identity just before the original correction's window: read
        # from any underlying (inferred or earlier operator) range, then fall
        # back to the live PH identity.
        restore_to, _auth = await self._corr.effective_identity(
            original.ph_id, original.observation_start - timedelta(microseconds=1)
        )

        version = await self._ph_version(original.ph_id, ph.observation_count)
        compensation = IdentitySegmentCorrection(
            correction_id=str(uuid.uuid4()),
            ph_id=original.ph_id,
            actor=actor,
            reason_code="other",
            observation_start=original.observation_start,
            observation_end=original.observation_end,
            base_ph_version=version,
            revision_id=revision_id,
            target_identity_id=restore_to,
            set_unknown=restore_to is None,
            kind="compensation",
            note=f"compensates {correction_id}",
            base_revision_id=original.revision_id,
            compensates_correction_id=correction_id,
            created_at=now,
        )
        await self._corr.save_correction(compensation)

        range_id = await self._write_operator_range(
            ph_id=original.ph_id,
            revision_id=revision_id,
            correction_id=compensation.correction_id,
            effective_identity_id=restore_to,
            start=original.observation_start,
            end=original.observation_end,
            created_at=now,
        )

        job = IdentityRevisionJob(
            job_id=str(uuid.uuid4()),
            revision_id=revision_id,
            status="applying",
            required_projections=self._cfg.required_projections,
            correction_id=compensation.correction_id,
            created_at=now,
            updated_at=now,
        )
        await self._corr.save_job(job)

        await self._corr.record_ack(
            ProjectionAck(
                revision_id=revision_id,
                consumer=CTS_INTERNAL,
                schema_version=CTS_PROJECTION_SCHEMA_VERSION,
                status="acked",
                applied_at=now,
            )
        )
        await self._corr.complete_job_if_ready(revision_id)

        await self._publish(
            revision_id=revision_id,
            ph_id=original.ph_id,
            previous_identity_id=original.target_identity_id,
            new_identity_id=restore_to,
            actor=actor,
            reason="operator_compensation",
            applied_at=now,
            rewritten_rows=0,
            range_start=original.observation_start,
            range_end=original.observation_end,
            revision_range_id=range_id,
            correction_id=compensation.correction_id,
        )

        return CorrectionResult(
            revision_id=revision_id,
            correction_id=compensation.correction_id,
            ph_id=original.ph_id,
            previous_identity_id=original.target_identity_id,
            new_identity_id=restore_to,
            range_id=range_id,
            new_ph_id=None,
            job=job,
        )

    # -- job status ----------------------------------------------------------

    async def get_job(self, revision_id: str) -> IdentityRevisionJob | None:
        """Return the projection job for a revision, or None if unknown.

        The admin UI polls this after applying a correction: ``apply`` returns
        immediately with ``status="applying"`` and completion is asynchronous
        (revision -> CC rewriter -> projection ack). A terminal ``completed`` /
        ``failed`` status here is the honest signal that the correction is done.
        """
        return await self._corr.get_job(revision_id)

    # -- projection ack intake ----------------------------------------------

    async def record_projection_ack(self, ack: ProjectionAck) -> bool:
        """Record a downstream projection ack and complete the job if ready."""
        await self._corr.record_ack(ack)
        if ack.status == "failed":
            await self._corr.update_job(
                ack.revision_id,
                status="failed",
                last_error=f"{ack.consumer} projection failed",
            )
            return False
        return await self._corr.complete_job_if_ready(ack.revision_id)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _revision_reason(kind: CorrectionKind, set_unknown: bool) -> str:
        if kind == "handoff_split":
            return "operator_handoff"
        if set_unknown:
            return "operator_unknown"
        if kind == "frame_only":
            return "operator_frame_correction"
        return "operator_correction"

    async def _publish(
        self,
        *,
        revision_id: str,
        ph_id: str,
        previous_identity_id: str | None,
        new_identity_id: str | None,
        actor: str,
        reason: str,
        applied_at: datetime,
        rewritten_rows: int,
        range_start: datetime | None = None,
        range_end: datetime | None = None,
        revision_range_id: str = "",
        correction_id: str = "",
    ) -> None:
        revision = IdentityRevision(
            revision_id=revision_id,
            ph_id=ph_id,
            previous_identity_id=previous_identity_id,
            new_identity_id=new_identity_id,
            actor=actor,
            reason=reason,
            applied_at=applied_at,
            rewritten_rows=rewritten_rows,
            evidence=IdentityEvidence(evidence_sources=["operator"]),
            revision_kind=reason,
            range_start=range_start,
            range_end=range_end,
            range_authority="operator",
            revision_range_id=revision_range_id,
            correction_id=correction_id,
            required_projections=tuple(self._cfg.required_projections),
            revision_schema_version=CTS_PROJECTION_SCHEMA_VERSION,
        )
        # Persist to the audit feed (ph_revisions) so the revisions feed and CC
        # rewriter see operator corrections, independent of publisher health.
        try:
            await self._ph_repo.record_revision(revision, kind="manual_correct")
        except Exception as exc:  # noqa: BLE001 - audit write is best-effort
            logger.warning(
                "correction_revision_record_failed",
                revision_id=revision_id,
                error=str(exc),
            )

        publisher = self._publisher
        if publisher is None or not getattr(publisher, "is_connected", False):
            return
        try:
            await publisher.publish(revision)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - publish is best-effort
            logger.warning("correction_publish_failed", revision_id=revision_id, error=str(exc))
