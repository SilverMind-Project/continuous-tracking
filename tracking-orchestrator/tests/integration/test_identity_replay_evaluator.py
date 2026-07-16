"""M12 golden evaluator: swap definition and the effective-identity read path.

These run in the normal gate (no marker, no Postgres): they exercise the pure
evaluator and the operator-correction overlay through the InMemory correction
repository. The headline test is ``test_operator_correction_changes_verdict``,
which proves the evaluator reads effective identity through
``effective_identity(...)`` and not the decision column -- a column read would
keep reporting the swap after the operator fixed it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from app.domain import IdentityRevisionRange
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from tests.integration._identity_replay import (
    ARCFACE_AUTHORITY_SOURCE,
    DecisionRow,
    build_records,
    evaluate,
)

T0 = datetime(2026, 6, 23, 9, 0, 0, tzinfo=UTC)


def _t(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _arcface(ph_id: str, frame: int, identity: str) -> DecisionRow:
    """A qualifying direct-ArcFace authoritative commit."""
    return DecisionRow(
        ph_id=ph_id,
        captured_at=_t(frame),
        frame_index=frame,
        inferred_identity_id=identity,
        decision_source=ARCFACE_AUTHORITY_SOURCE,
        authority="direct_face",
        bbox=(0.0, 0.0, 1.0, 2.0),
    )


def _reid(ph_id: str, frame: int, identity: str) -> DecisionRow:
    """A non-authoritative ReID/posterior commit (wobble-eligible)."""
    return DecisionRow(
        ph_id=ph_id,
        captured_at=_t(frame),
        frame_index=frame,
        inferred_identity_id=identity,
        decision_source="reid",
        authority="",
        bbox=(0.0, 0.0, 1.0, 2.0),
    )


async def _records(rows, repo=None):
    repo = repo or InMemoryIdentityCorrectionRepository()
    return await build_records(rows, repo)


async def test_arcface_flip_is_a_swap() -> None:
    rows = [
        _arcface("ph-1", 0, "amma"),
        _arcface("ph-1", 1, "amma"),
        _arcface("ph-1", 2, "grandma"),
    ]
    report = evaluate(await _records(rows))
    assert report.swap_count == 1
    swap = report.authoritative_swaps[0]
    assert (swap.from_identity, swap.to_identity) == ("amma", "grandma")


async def test_reid_wobble_is_not_a_swap() -> None:
    # ReID/posterior identity jitter never reaches the authoritative chain.
    rows = [_reid("ph-1", 0, "amma"), _reid("ph-1", 1, "grandma"), _reid("ph-1", 2, "amma")]
    report = evaluate(await _records(rows))
    assert report.swap_count == 0
    assert report.authoritative_frames == 0


async def test_arcface_then_reid_disagreement_is_not_a_swap() -> None:
    # A sub-threshold ReID disagreement between two consistent ArcFace frames
    # must not register: only authoritative frames join the chain.
    rows = [_arcface("ph-1", 0, "amma"), _reid("ph-1", 1, "grandma"), _arcface("ph-1", 2, "amma")]
    report = evaluate(await _records(rows))
    assert report.swap_count == 0


async def test_operator_correction_changes_verdict() -> None:
    """The decisive read-path proof.

    Raw decisions show a direct-ArcFace swap amma -> grandma. With no operator
    correction the evaluator reports one authoritative swap (the release gate
    fails). After an operator correction range relabels the grandma frames back
    to amma, the evaluator -- reading through effective_identity -- reports zero
    swaps. A consumer that read the inferred decision column would still see
    grandma and keep failing the gate.
    """
    rows = [
        _arcface("ph-1", 0, "amma"),
        _arcface("ph-1", 1, "amma"),
        _arcface("ph-1", 2, "grandma"),  # bad machine flip
        _arcface("ph-1", 3, "grandma"),
    ]

    # Before correction: one authoritative swap.
    before = evaluate(await _records(rows))
    assert before.swap_count == 1

    # Operator corrects frames 2-3 back to amma.
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_range(
        IdentityRevisionRange(
            range_id=str(uuid.uuid4()),
            revision_id=str(uuid.uuid4()),
            ph_id="ph-1",
            authority="operator",
            range_start=_t(2),
            range_end=_t(3),
            effective_identity_id="amma",
        )
    )
    after_records = await build_records(rows, repo)
    after = evaluate(after_records)

    assert after.swap_count == 0
    # The overlay won: frames 2-3 now read amma with operator authority...
    corrected = [r for r in after_records if r.frame_index in (2, 3)]
    assert all(r.effective_identity_id == "amma" for r in corrected)
    assert all(r.effective_authority == "operator" for r in corrected)
    # ...while the raw inferred decision column would still say grandma.
    assert all(r.inferred_identity_id == "grandma" for r in corrected)


async def test_operator_change_is_not_a_swap_but_machine_override_is() -> None:
    # ArcFace says amma frames 0-1; operator deliberately relabels frames 2-3 to
    # grandma -> authorised, not a swap. Then ArcFace overrides the operator at
    # frame 4 back to amma -> a swap (machine contradicting an operator range).
    rows = [
        _arcface("ph-1", 0, "amma"),
        _arcface("ph-1", 1, "amma"),
        _arcface("ph-1", 2, "amma"),
        _arcface("ph-1", 3, "amma"),
        _arcface("ph-1", 4, "amma"),
    ]
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_range(
        IdentityRevisionRange(
            range_id=str(uuid.uuid4()),
            revision_id=str(uuid.uuid4()),
            ph_id="ph-1",
            authority="operator",
            range_start=_t(2),
            range_end=_t(3),
            effective_identity_id="grandma",
        )
    )
    report = evaluate(await build_records(rows, repo))
    # amma(0,1) -> grandma(2,3 operator) is authorised; grandma -> amma(4 arcface) is a swap.
    assert report.swap_count == 1
    assert report.authoritative_swaps[0].to_identity == "amma"
    assert report.authoritative_swaps[0].to_authority == "arcface"


async def test_unknown_rate_and_duration() -> None:
    # Known, then a 2 s unknown gap, then known again.
    rows = [
        _arcface("ph-1", 0, "amma"),
        DecisionRow(ph_id="ph-1", captured_at=_t(1), frame_index=1, inferred_identity_id=""),
        DecisionRow(ph_id="ph-1", captured_at=_t(2), frame_index=2, inferred_identity_id=""),
        _arcface("ph-1", 3, "amma"),
    ]
    report = evaluate(await _records(rows))
    assert report.frames == 4
    assert report.unknown_frames == 2
    assert report.unknown_rate == 0.5
    assert report.unknown_after_known == 1
    assert report.unknown_durations_s == [1.0]  # _t(2) - _t(1)


async def test_duplicate_active_identity_is_flagged() -> None:
    # Two distinct PHs hold amma at the same instant.
    rows = [_arcface("ph-1", 0, "amma"), _arcface("ph-2", 0, "amma")]
    report = evaluate(await _records(rows))
    assert report.duplicate_active_frames == 1
    assert report.distinct_phs == 2


async def test_golden_identity_accuracy_and_source_attribution() -> None:
    rows = [_arcface("ph-1", 0, "amma"), _arcface("ph-1", 1, "grandma")]
    golden = {"ph-1": {0: "amma", 1: "amma"}}  # frame 1 is wrong vs golden
    report = evaluate(await _records(rows), golden=golden)
    assert report.identity_accuracy == 0.5
    assert report.source_attribution_complete is True


async def test_inferred_range_does_not_mask_arcface_authority() -> None:
    """An inferred revision range restates inference; it must not hide a swap.

    record_inferred_range is unwired today, but if an inferred range ever covered
    an ArcFace-authority frame, treating it as non-authoritative would zero the
    gate. The frames stay ArcFace-authoritative, so the amma -> grandma flip is
    still a swap.
    """
    rows = [_arcface("ph-1", 0, "amma"), _arcface("ph-1", 1, "grandma")]
    repo = InMemoryIdentityCorrectionRepository()
    # An inferred range over both frames, restating each frame's inferred id.
    for frame, ident in ((0, "amma"), (1, "grandma")):
        await repo.save_range(
            IdentityRevisionRange(
                range_id=str(uuid.uuid4()),
                revision_id=str(uuid.uuid4()),
                ph_id="ph-1",
                authority="inferred",
                range_start=_t(frame),
                range_end=_t(frame),
                effective_identity_id=ident,
            )
        )
    records = await build_records(rows, repo)
    assert all(r.effective_authority == "arcface" for r in records)
    assert evaluate(records).swap_count == 1


async def test_fragmentation_counts_identity_split_across_phs() -> None:
    # amma handed from ph-1 to ph-2 (fragmentation); grandma stays on ph-3.
    rows = [
        _arcface("ph-1", 0, "amma"),
        _arcface("ph-2", 1, "amma"),
        _arcface("ph-3", 0, "grandma"),
    ]
    report = evaluate(await _records(rows))
    assert report.fragmented_identities == 1  # only amma is split


async def test_record_serialisation_is_deterministic() -> None:
    rows = [_arcface("ph-1", 0, "amma"), _reid("ph-1", 1, "grandma")]
    first = [r.to_jsonl() for r in await _records(rows)]
    second = [r.to_jsonl() for r in await _records(rows)]
    assert first == second  # stable, sorted-key JSON for golden-diffing
    assert '"effective_identity_id": "amma"' in first[0]
