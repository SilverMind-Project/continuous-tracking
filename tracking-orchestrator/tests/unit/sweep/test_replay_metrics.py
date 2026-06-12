"""Unit tests for replay_metrics.py.

Tests >= 8 hand-crafted scenarios with known expected values, including
a revival-linked lineage case where the same ph_id continues across close+open.

No I/O; all data is constructed inline.
"""

from __future__ import annotations

import pytest
from replay_metrics import (
    FrameRecord,
    RunMetrics,
    SweepRunResult,
    aggregate_metrics,
    score_run,
)

# ── Helpers ───────────────────────────────────────────────────────────────


def _run(frames: list[FrameRecord], fixture_name: str = "test") -> SweepRunResult:
    r = SweepRunResult(fixture_name=fixture_name)
    r.frames = frames
    return r


def _frame(
    det_to_ph: dict[str, str],
    ph_identities: dict[str, str | None] | None = None,
) -> FrameRecord:
    return FrameRecord(step=0, det_to_ph=det_to_ph, ph_identities=ph_identities or {})


def _truth(persons: list[str], mapping: dict[str, str]) -> dict:
    return {"persons": persons, "detection_truth": mapping, "events": []}


# ── Scenario 1: perfect single-person tracking ─────────────────────────────


def test_perfect_single_person() -> None:
    """One person, one PH, all frames matched. IDF1=1, contamination=0."""
    frames = [
        _frame({"det-a-0": "ph-1", "det-a-1": "ph-1"}),
        _frame({"det-a-2": "ph-1", "det-a-3": "ph-1"}),
    ]
    truth = _truth(
        ["alice"],
        {"det-a-0": "alice", "det-a-1": "alice", "det-a-2": "alice", "det-a-3": "alice"},
    )
    m = score_run(_run(frames), truth)
    assert m.identity_preservation == pytest.approx(1.0)
    assert m.phantom_rate == pytest.approx(0.0)
    assert m.fragmentation == pytest.approx(1.0)
    assert m.identity_contamination == 0
    assert m.admissible


# ── Scenario 2: fragmented tracking (two PHs for one person) ──────────────


def test_fragmented_single_person() -> None:
    """One person split into two PHs (PH closed + respawn). Fragmentation=2."""
    frames = [
        _frame({"det-0": "ph-1", "det-1": "ph-1", "det-2": "ph-1"}),
        _frame({"det-3": "ph-2", "det-4": "ph-2"}),
    ]
    truth = _truth(
        ["alice"],
        {
            "det-0": "alice",
            "det-1": "alice",
            "det-2": "alice",
            "det-3": "alice",
            "det-4": "alice",
        },
    )
    m = score_run(_run(frames), truth)
    assert m.fragmentation == pytest.approx(2.0)
    # IDF1: best PH covers 3/5 obs. TP=3, FP=0, FN=2. IDF1 = 6/(6+0+2) = 0.75
    assert m.identity_preservation == pytest.approx(0.75)
    assert m.identity_contamination == 0


# ── Scenario 3: two people, perfect assignment ────────────────────────────


def test_perfect_two_people() -> None:
    """Two people, two PHs, perfect separation. IDF1=1 per person."""
    frames = [
        _frame({"det-a-0": "ph-alice", "det-b-0": "ph-bob"}),
        _frame({"det-a-1": "ph-alice", "det-b-1": "ph-bob"}),
    ]
    truth = _truth(
        ["alice", "bob"],
        {
            "det-a-0": "alice",
            "det-b-0": "bob",
            "det-a-1": "alice",
            "det-b-1": "bob",
        },
    )
    m = score_run(_run(frames), truth)
    assert m.identity_preservation == pytest.approx(1.0)
    assert m.fragmentation == pytest.approx(1.0)
    assert m.identity_contamination == 0


# ── Scenario 4: identity swap (wrong person assigned to PH) ───────────────


def test_identity_swap_causes_contamination() -> None:
    """PH carries 'alice' identity but observations are truly 'bob'."""
    frames = [
        _frame(
            {"det-b-0": "ph-1", "det-b-1": "ph-1"},
            ph_identities={"ph-1": "alice"},  # wrong: alice committed but these are bob
        ),
    ]
    truth = _truth(["alice", "bob"], {"det-b-0": "bob", "det-b-1": "bob"})
    m = score_run(_run(frames), truth)
    assert m.identity_contamination == 2  # both observations contaminated
    assert not m.admissible


# ── Scenario 5: revival-linked lineage (same ph_id reused) ───────────────


def test_revival_linked_lineage_no_fragmentation() -> None:
    """PH closes and is revived with the same ph_id. Fragmentation=1."""
    # Phase 1: ph-1 is open (3 observations)
    frames = [
        _frame({"det-0": "ph-1", "det-1": "ph-1", "det-2": "ph-1"}),
    ]
    # Phase 2: empty (ph-1 closes)
    frames.append(_frame({}))
    # Phase 3: ph-1 revived (same id, 2 more observations)
    frames.append(_frame({"det-3": "ph-1", "det-4": "ph-1"}))

    truth = _truth(
        ["alice"],
        {
            "det-0": "alice",
            "det-1": "alice",
            "det-2": "alice",
            "det-3": "alice",
            "det-4": "alice",
        },
    )
    m = score_run(_run(frames), truth)
    assert m.fragmentation == pytest.approx(1.0)
    assert m.identity_preservation == pytest.approx(1.0)
    assert m.identity_contamination == 0


# ── Scenario 6: phantom PH (detection not in truth) ───────────────────────


def test_phantom_detection_not_in_truth() -> None:
    """Detection appears in det_to_ph but not in ground truth → phantom rate."""
    frames = [
        _frame(
            {"det-alice-0": "ph-1", "det-ghost-0": "ph-2"},
        ),
    ]
    truth = _truth(["alice"], {"det-alice-0": "alice"})
    m = score_run(_run(frames), truth)
    assert m.phantom_rate == pytest.approx(0.5)  # 1 phantom / 2 total PH-obs


# ── Scenario 7: partial identity contamination ────────────────────────────


def test_partial_contamination() -> None:
    """PH initially has no committed identity, then gets wrong one mid-replay."""
    frames = [
        _frame(
            {"det-0": "ph-1", "det-1": "ph-1"},
            ph_identities={"ph-1": None},  # no identity yet
        ),
        _frame(
            {"det-2": "ph-1"},
            ph_identities={"ph-1": "alice"},  # wrong: observations are bob
        ),
    ]
    truth = _truth(
        ["bob"],
        {"det-0": "bob", "det-1": "bob", "det-2": "bob"},
    )
    m = score_run(_run(frames), truth)
    # Frame 0: no committed identity → no contamination.
    # Frame 1: committed=alice, truth=bob → 1 contamination.
    assert m.identity_contamination == 1
    assert not m.admissible


# ── Scenario 8: empty fixture ─────────────────────────────────────────────


def test_empty_fixture_returns_defaults() -> None:
    """No observations → all metrics at ideal neutral values."""
    run = _run([_frame({}), _frame({})])
    truth = _truth([], {})
    m = score_run(run, truth)
    assert m.identity_preservation == pytest.approx(1.0)
    assert m.phantom_rate == pytest.approx(0.0)
    assert m.fragmentation == pytest.approx(1.0)
    assert m.identity_contamination == 0
    assert m.admissible


# ── Scenario 9: correct identity committed, no contamination ──────────────


def test_correct_committed_identity_no_contamination() -> None:
    """PH carries 'alice' and all its observations are truly alice."""
    frames = [
        _frame(
            {"det-0": "ph-1", "det-1": "ph-1"},
            ph_identities={"ph-1": "alice"},
        ),
        _frame(
            {"det-2": "ph-1"},
            ph_identities={"ph-1": "alice"},
        ),
    ]
    truth = _truth(
        ["alice"],
        {"det-0": "alice", "det-1": "alice", "det-2": "alice"},
    )
    m = score_run(_run(frames), truth)
    assert m.identity_contamination == 0
    assert m.admissible
    assert m.identity_preservation == pytest.approx(1.0)


# ── Scenario 10: multi-person revival one revived one new ─────────────────


def test_two_people_one_revived_one_new() -> None:
    """Alice revived (same ph_id), bob spawned new (different ph_id).
    Fragmentation: alice=1, bob=2 → avg=1.5."""
    truth = _truth(
        ["alice", "bob"],
        {
            "det-a0": "alice",
            "det-a1": "alice",
            "det-a2": "alice",  # ph-alice revived
            "det-b0": "bob",
            "det-b1": "bob",
            "det-b2": "bob",  # ph-bob2 spawned
        },
    )
    frames = [
        _frame({"det-a0": "ph-alice", "det-b0": "ph-bob"}),
        _frame({}),  # both PHs coast then close
        _frame({"det-a1": "ph-alice", "det-b1": "ph-bob2"}),  # alice revived, bob new PH
        _frame({"det-a2": "ph-alice", "det-b2": "ph-bob2"}),
    ]
    m = score_run(_run(frames), truth)
    assert m.fragmentation == pytest.approx(1.5)
    assert m.identity_contamination == 0


# ── aggregate_metrics tests ───────────────────────────────────────────────


def test_aggregate_admissible_all_clean() -> None:
    metrics = [
        RunMetrics("f1", 0.9, 0.0, 1.0, 0, 1),
        RunMetrics("f2", 0.8, 0.1, 1.2, 0, 2),
    ]
    agg = aggregate_metrics(metrics)
    assert agg["identity_contamination"] == 0.0
    assert agg["admissible"] == 1.0
    assert agg["identity_preservation"] == pytest.approx(0.85)


def test_aggregate_inadmissible_if_any_contamination() -> None:
    metrics = [
        RunMetrics("f1", 1.0, 0.0, 1.0, 0, 1),
        RunMetrics("f2", 0.9, 0.0, 1.0, 1, 1),  # contamination!
    ]
    agg = aggregate_metrics(metrics)
    assert agg["admissible"] == 0.0
    assert agg["identity_contamination"] == 1.0
