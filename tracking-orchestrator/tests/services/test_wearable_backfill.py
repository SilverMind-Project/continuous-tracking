"""UnknownBackfillService's wearable trigger: trajectory-attested segments backfill NULL rows.

Reuses the InMemory harness of ``test_unknown_backfill.py``. The trigger is independent of the
face trigger's flag, fires once per segment, never for room grade, keeps every clip of the face
path, and in shadow writes nothing.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from app.domain import (
    IdentityRevisionRange,
    WearableAssociationSegment,
    WearableAssociationTransition,
    WearableAttestationGrade,
)
from app.services.identity_correction_service import IdentityCorrectionService
from app.services.unknown_backfill import BackfillConfig, UnknownBackfillService
from app.storage.base import InMemoryIdentityDecisionRepository, InMemoryPHRepository
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from tests.services.test_unknown_backfill import T0, SpyPublisher, SpyRewriter, _ph

_START = T0 + timedelta(minutes=10)
_NOW = T0 + timedelta(minutes=12)


def _harness(
    *, wearable_enabled: bool = True, shadow: bool = False
) -> tuple[
    UnknownBackfillService,
    InMemoryPHRepository,
    InMemoryIdentityCorrectionRepository,
    SpyRewriter,
    SpyPublisher,
]:
    ph_repo = InMemoryPHRepository()
    corr_repo = InMemoryIdentityCorrectionRepository()
    rewriter = SpyRewriter()
    publisher = SpyPublisher()
    svc = UnknownBackfillService(
        ph_repo=ph_repo,
        identity_decision_repo=InMemoryIdentityDecisionRepository(),
        correction_service=IdentityCorrectionService(ph_repo=ph_repo, correction_repo=corr_repo),
        identity_rewriter=rewriter,
        revision_publisher=publisher,
        # The face trigger stays off: the wearable trigger must not depend on it.
        config=BackfillConfig(enabled=False, shadow=shadow, wearable_enabled=wearable_enabled),
    )
    return svc, ph_repo, corr_repo, rewriter, publisher


def _transition(
    grade: WearableAttestationGrade = WearableAttestationGrade.TRAJECTORY,
    *,
    segment_id: str = "seg-1",
) -> WearableAssociationTransition:
    return WearableAssociationTransition(
        action="update",
        segment=WearableAssociationSegment(
            segment_id=segment_id,
            ph_id="ph-1",
            tag_id="tag-a",
            identity_id="alice",
            started_at=_START,
            ended_at=None,
            attestation_grade=grade,
            window_llr=18.0,
            margin_llr=9.0,
            matched_fixes=10,
            mean_residual_m=0.1,
        ),
    )


async def _process(
    svc: UnknownBackfillService, *transitions: WearableAssociationTransition
) -> None:
    await svc.process(
        outcome_decisions=[],
        ph_born_at_by_id={"ph-1": T0},
        event_time=_NOW + timedelta(seconds=3),  # processing time; the range ends at capture
        wearable_transitions=transitions,
        captured_at=_NOW,
    )


@pytest.mark.asyncio
async def test_trajectory_segment_backfills_null_rows_over_the_segment_once() -> None:
    svc, ph_repo, corr_repo, rewriter, publisher = _harness()
    await ph_repo.save(_ph("ph-1", T0))

    await _process(svc, _transition())
    await _process(svc, _transition())  # the same segment never backfills twice

    (call,) = rewriter.backfill_calls
    _revision_id, ph_id, identity_id, start, end = call
    assert (ph_id, identity_id, start, end) == ("ph-1", "alice", _START, _NOW)
    (rng,) = await corr_repo.list_ranges("ph-1", live_only=False)
    assert (rng.authority, rng.range_start, rng.range_end) == ("inferred", _START, _NOW)
    (revision,) = publisher.published
    assert revision.revision_kind == "inferred_backfill"
    assert revision.evidence.evidence_sources == ["wearable"]


@pytest.mark.asyncio
async def test_room_grade_and_unattested_segments_never_backfill() -> None:
    svc, ph_repo, _corr, rewriter, publisher = _harness()
    await ph_repo.save(_ph("ph-1", T0))
    await _process(
        svc,
        _transition(WearableAttestationGrade.ROOM),
        _transition(WearableAttestationGrade.NONE, segment_id="seg-2"),
    )
    assert rewriter.backfill_calls == []
    assert publisher.published == []


@pytest.mark.asyncio
async def test_disabled_and_shadow_write_nothing() -> None:
    for harness in (_harness(wearable_enabled=False), _harness(shadow=True)):
        svc, ph_repo, corr_repo, rewriter, publisher = harness
        await ph_repo.save(_ph("ph-1", T0))
        await _process(svc, _transition())
        assert rewriter.backfill_calls == []
        assert publisher.published == []
        assert await corr_repo.list_ranges("ph-1", live_only=False) == []


@pytest.mark.asyncio
async def test_operator_range_clips_the_start() -> None:
    svc, ph_repo, corr_repo, rewriter, _publisher = _harness()
    await ph_repo.save(_ph("ph-1", T0))
    operator_end = _START + timedelta(seconds=40)
    await corr_repo.save_range(
        IdentityRevisionRange(
            range_id="op-1",
            revision_id="op-rev-1",
            ph_id="ph-1",
            authority="operator",
            range_start=T0,
            range_end=operator_end,
            effective_identity_id="bob",
        )
    )
    await _process(svc, _transition())
    (call,) = rewriter.backfill_calls
    # Strictly after the operator range, so the inferred range never touches it.
    assert call[3] == operator_end + timedelta(microseconds=1)


@pytest.mark.asyncio
async def test_operator_range_over_the_live_edge_skips() -> None:
    svc, ph_repo, corr_repo, rewriter, _publisher = _harness()
    await ph_repo.save(_ph("ph-1", T0))
    await corr_repo.save_range(
        replace(
            IdentityRevisionRange(
                range_id="op-1",
                revision_id="op-rev-1",
                ph_id="ph-1",
                authority="operator",
                range_start=T0,
                range_end=_NOW + timedelta(minutes=5),
                effective_identity_id="bob",
            )
        )
    )
    await _process(svc, _transition())
    assert rewriter.backfill_calls == []
