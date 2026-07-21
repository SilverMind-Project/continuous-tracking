"""Tests for UnknownBackfillService (identity-continuity M04).

All InMemory; no database or Redis. Covers the trigger predicate, range
clipping (prior-decision, operator-range, cap), operator-live-edge skip,
shadow mode (no writes), enabled mode (range + rewrite + publish), the
operator-conflict race, and the per-(ph_id, identity_id) rate limit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    IdentityDecision,
    IdentityProvenanceDecision,
    IdentityRevisionRange,
    PersonHypothesis,
    PosteriorDist,
)
from app.services.identity_correction_service import IdentityCorrectionService
from app.services.identity_rewriter import IdentityRewriter
from app.services.unknown_backfill import BackfillConfig, UnknownBackfillService
from app.storage.base import InMemoryIdentityDecisionRepository, InMemoryPHRepository
from app.storage.corrections import InMemoryIdentityCorrectionRepository

T0 = datetime(2026, 7, 20, 8, 0, 0, tzinfo=UTC)


class SpyRewriter(IdentityRewriter):
    """Records rewrite/backfill_null_rows calls; a real InMemory-shaped fake."""

    def __init__(self) -> None:
        self.rewrite_calls: list[tuple[str, str, str | None, str, datetime, datetime]] = []
        self.backfill_calls: list[tuple[str, str, str, datetime, datetime]] = []

    async def rewrite(
        self,
        revision_id: str,
        ph_id: str,
        old_identity_id: str | None,
        new_identity_id: str,
        applies_from: datetime,
        applies_to: datetime,
    ) -> None:
        self.rewrite_calls.append(
            (revision_id, ph_id, old_identity_id, new_identity_id, applies_from, applies_to)
        )

    async def backfill_null_rows(
        self,
        revision_id: str,
        ph_id: str,
        new_identity_id: str,
        applies_from: datetime,
        applies_to: datetime,
    ) -> None:
        self.backfill_calls.append((revision_id, ph_id, new_identity_id, applies_from, applies_to))


class SpyPublisher:
    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish_many(self, revisions: list[object]) -> list[str]:
        self.published.extend(revisions)
        return [f"msg-{i}" for i in range(len(revisions))]


def _ph(ph_id: str, born_at: datetime) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=born_at,
        last_seen_at=born_at,
        last_seen_camera="cam-1",
        observation_count=0,
        current_identity_id=None,
        active_cameras=frozenset({"cam-1"}),
    )


def _decision(
    ph_id: str,
    identity_id: str | None,
    *,
    previous_identity_id: str | None,
    authority: str = "direct_face",
) -> IdentityDecision:
    return IdentityDecision(
        ph_id=ph_id,
        identity_id=identity_id,
        posterior=PosteriorDist(distribution={identity_id: 1.0} if identity_id else {}),
        revises_previous=True,
        previous_identity_id=previous_identity_id,
        authority=authority,
        decision_id="decision-1",
    )


_Harness = tuple[
    UnknownBackfillService,
    InMemoryPHRepository,
    InMemoryIdentityCorrectionRepository,
    InMemoryIdentityDecisionRepository,
    SpyRewriter,
    SpyPublisher,
]


def _harness(
    *, enabled: bool = True, shadow: bool = False, max_range_s: float = 14400.0
) -> _Harness:
    ph_repo = InMemoryPHRepository()
    corr_repo = InMemoryIdentityCorrectionRepository()
    decision_repo = InMemoryIdentityDecisionRepository()
    rewriter = SpyRewriter()
    publisher = SpyPublisher()
    correction_service = IdentityCorrectionService(
        ph_repo=ph_repo,
        correction_repo=corr_repo,
    )
    svc = UnknownBackfillService(
        ph_repo=ph_repo,
        identity_decision_repo=decision_repo,
        correction_service=correction_service,
        identity_rewriter=rewriter,
        revision_publisher=publisher,
        config=BackfillConfig(enabled=enabled, shadow=shadow, max_range_s=max_range_s),
    )
    return svc, ph_repo, corr_repo, decision_repo, rewriter, publisher


# -- trigger predicate -------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_only_on_direct_face_first_commit() -> None:
    svc, ph_repo, _corr, _decisions, rewriter, publisher = _harness()
    await ph_repo.save(_ph("ph-1", T0))

    # Posterior-authority first commit: must not backfill.
    posterior_decision = _decision(
        "ph-1", "alice", previous_identity_id=None, authority="posterior"
    )
    await svc.process(
        outcome_decisions=[posterior_decision],
        ph_born_at_by_id={"ph-1": T0},
        event_time=T0 + timedelta(hours=3),
    )
    assert rewriter.backfill_calls == []
    assert publisher.published == []

    # Identity-to-identity change (not a first commit): must not backfill.
    change_decision = _decision("ph-1", "bob", previous_identity_id="alice")
    await svc.process(
        outcome_decisions=[change_decision],
        ph_born_at_by_id={"ph-1": T0},
        event_time=T0 + timedelta(hours=3),
    )
    assert rewriter.backfill_calls == []
    assert publisher.published == []


# -- range clipping -----------------------------------------------------------


@pytest.mark.asyncio
async def test_range_clips_cap_prior_decision_operator_range() -> None:
    svc, ph_repo, corr_repo, decision_repo, rewriter, _publisher = _harness(max_range_s=3600.0)
    born_at = T0
    commit_time = T0 + timedelta(hours=6)
    await ph_repo.save(_ph("ph-1", born_at))

    # A prior decision at T0+1h named a different identity ("carol"); the
    # backfill for "alice" must not reach before that boundary.
    prior_decision_time = T0 + timedelta(hours=1)
    await decision_repo.save(
        IdentityProvenanceDecision(
            decision_id="prior-1",
            ph_id="ph-1",
            captured_at=prior_decision_time,
            authority="direct_face",
            decision_source="arcface_authority",
            diagnostics={},
            inferred_identity_id="carol",
        )
    )

    # An operator range from T0+2h to T0+3h ends before commit_time and does
    # not own the live edge, so it clips forward but does not skip.
    await corr_repo.save_range(
        IdentityRevisionRange(
            range_id="op-range-1",
            revision_id="op-rev-1",
            ph_id="ph-1",
            authority="operator",
            range_start=T0 + timedelta(hours=2),
            range_end=T0 + timedelta(hours=3),
            effective_identity_id="carol",
        )
    )

    decision = _decision("ph-1", "alice", previous_identity_id=None)
    await svc.process(
        outcome_decisions=[decision],
        ph_born_at_by_id={"ph-1": born_at},
        event_time=commit_time,
    )

    assert len(rewriter.backfill_calls) == 1
    _rev_id, ph_id, new_id, applies_from, applies_to = rewriter.backfill_calls[0]
    assert ph_id == "ph-1"
    assert new_id == "alice"
    # Cap (3600s = 1h before commit_time) is tighter than the operator-range
    # clip (T0+3h) and the prior-decision clip (T0+1h), so the cap wins:
    # start = commit_time - 1h = T0 + 5h.
    assert applies_from == commit_time - timedelta(hours=1)
    assert applies_to == commit_time


@pytest.mark.asyncio
async def test_operator_live_edge_skips() -> None:
    svc, ph_repo, corr_repo, _decisions, rewriter, publisher = _harness()
    born_at = T0
    commit_time = T0 + timedelta(hours=2)
    await ph_repo.save(_ph("ph-1", born_at))

    # Operator range covers the live edge (commit_time itself).
    await corr_repo.save_range(
        IdentityRevisionRange(
            range_id="op-range-1",
            revision_id="op-rev-1",
            ph_id="ph-1",
            authority="operator",
            range_start=T0,
            range_end=commit_time + timedelta(hours=1),
            effective_identity_id="carol",
        )
    )

    decision = _decision("ph-1", "alice", previous_identity_id=None)
    await svc.process(
        outcome_decisions=[decision],
        ph_born_at_by_id={"ph-1": born_at},
        event_time=commit_time,
    )

    assert rewriter.backfill_calls == []
    assert publisher.published == []


# -- shadow / enabled ---------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_mode_writes_nothing_but_counts() -> None:
    svc, ph_repo, corr_repo, _decisions, rewriter, publisher = _harness(shadow=True)
    born_at = T0
    commit_time = T0 + timedelta(hours=3)
    await ph_repo.save(_ph("ph-1", born_at))

    decision = _decision("ph-1", "alice", previous_identity_id=None)
    await svc.process(
        outcome_decisions=[decision],
        ph_born_at_by_id={"ph-1": born_at},
        event_time=commit_time,
    )

    assert rewriter.backfill_calls == []
    assert publisher.published == []
    assert await corr_repo.list_ranges("ph-1", live_only=False) == []


@pytest.mark.asyncio
async def test_enabled_records_range_relabels_and_publishes() -> None:
    svc, ph_repo, corr_repo, _decisions, rewriter, publisher = _harness()
    born_at = T0
    commit_time = T0 + timedelta(hours=3)
    await ph_repo.save(_ph("ph-1", born_at))

    decision = _decision("ph-1", "alice", previous_identity_id=None)
    await svc.process(
        outcome_decisions=[decision],
        ph_born_at_by_id={"ph-1": born_at},
        event_time=commit_time,
    )

    ranges = await corr_repo.list_ranges("ph-1", live_only=False)
    assert len(ranges) == 1
    assert ranges[0].authority == "inferred"
    assert ranges[0].range_start == born_at
    assert ranges[0].range_end == commit_time
    assert ranges[0].effective_identity_id == "alice"

    assert len(rewriter.backfill_calls) == 1
    assert len(publisher.published) == 1
    revision = publisher.published[0]
    assert revision.revision_kind == "inferred_backfill"
    assert revision.previous_identity_id is None
    assert revision.new_identity_id == "alice"
    assert revision.range_start == born_at
    assert revision.range_end == commit_time
    assert revision.required_projections == ("cc",)

    # The job requires both cts_internal and cc acks; only cts_internal is
    # acked synchronously here (acks cc once CC projects the segment), so
    # the job stays "applying" until then.
    job = await corr_repo.get_job(revision.revision_id)
    assert job is not None
    assert job.status == "applying"
    acks = await corr_repo.list_acks(revision.revision_id)
    assert {a.consumer for a in acks} == {"cts_internal"}


@pytest.mark.asyncio
async def test_operator_conflict_race_aborts_cleanly() -> None:
    """An operator range lands between the clip check and the write."""
    svc, ph_repo, corr_repo, _decisions, rewriter, publisher = _harness()
    born_at = T0
    commit_time = T0 + timedelta(hours=3)
    await ph_repo.save(_ph("ph-1", born_at))

    real_overlapping = corr_repo.operator_ranges_overlapping
    call_count = {"n": 0}

    async def racing_overlapping(ph_id: str, start: datetime, end: datetime):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call (the service's own clip check) sees nothing.
            return []
        return await real_overlapping(ph_id, start, end)

    # Simulate an operator range appearing after the clip check but before
    # record_inferred_range's own overlap check, by inserting it once the
    # service's clip-check has already run.
    async def patched(ph_id: str, start: datetime, end: datetime):
        result = await racing_overlapping(ph_id, start, end)
        if call_count["n"] == 1:
            await corr_repo.save_range(
                IdentityRevisionRange(
                    range_id="racer",
                    revision_id="racer-rev",
                    ph_id="ph-1",
                    authority="operator",
                    range_start=born_at,
                    range_end=commit_time,
                    effective_identity_id="carol",
                )
            )
        return result

    corr_repo.operator_ranges_overlapping = patched  # type: ignore[method-assign]

    decision = _decision("ph-1", "alice", previous_identity_id=None)
    await svc.process(
        outcome_decisions=[decision],
        ph_born_at_by_id={"ph-1": born_at},
        event_time=commit_time,
    )

    assert rewriter.backfill_calls == []
    assert publisher.published == []
    inferred_ranges = [
        r for r in await corr_repo.list_ranges("ph-1", live_only=False) if r.authority == "inferred"
    ]
    assert inferred_ranges == []


@pytest.mark.asyncio
async def test_rate_limit_one_per_ph_identity() -> None:
    svc, ph_repo, _corr, _decisions, rewriter, _publisher = _harness()
    born_at = T0
    commit_time = T0 + timedelta(hours=3)
    await ph_repo.save(_ph("ph-1", born_at))

    decision = _decision("ph-1", "alice", previous_identity_id=None)
    await svc.process(
        outcome_decisions=[decision],
        ph_born_at_by_id={"ph-1": born_at},
        event_time=commit_time,
    )
    assert len(rewriter.backfill_calls) == 1

    # A second identical trigger (e.g. resolver re-emits the same decision
    # next frame) must not re-attempt within the rate-limit TTL.
    await svc.process(
        outcome_decisions=[decision],
        ph_born_at_by_id={"ph-1": born_at},
        event_time=commit_time + timedelta(seconds=1),
    )
    assert len(rewriter.backfill_calls) == 1
