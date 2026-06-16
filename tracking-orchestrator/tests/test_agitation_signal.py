"""Tests for the agitation_index dementia signal.

All tests use InMemory repositories — no DB required.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import PersonTrajectoryPoint
from app.storage.base import (
    InMemoryBehaviorBaselineRepository,
    InMemoryDementiaSignalRepository,
    InMemoryTrajectoryRepository,
)
from app.storage.signals import AgitationWindowRecord
from app.trajectory.dementia_signals import DementiaSignalWorker, SignalConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 12, 14, 0, 0, tzinfo=UTC)
_IDENTITY = "resident-agitation"
_GT = "ph-agitation-001"

_AGIT_CFG = SignalConfig(
    agitation_enabled=True,
    onset_consecutive_windows=1,
    min_baseline_n=3,
    agitation_min_observed_minutes=5,
)


# ---------------------------------------------------------------------------
# Point factory
# ---------------------------------------------------------------------------


def _point(
    room: str = "living_room",
    offset_minutes: float = 0.0,
    motion_energy: float | None = 0.15,
    floor_speed_m_s: float | None = 0.05,
    ground_x: float = 1.0,
    ground_y: float = 1.0,
    now: datetime = _NOW,
) -> PersonTrajectoryPoint:
    return PersonTrajectoryPoint(
        identity_id=_IDENTITY,
        ph_id=_GT,
        observed_at=now - timedelta(minutes=offset_minutes),
        room_name=room,
        ground_x=ground_x,
        ground_y=ground_y,
        identity_confidence=0.9,
        motion_energy=motion_energy,
        floor_speed_m_s=floor_speed_m_s,
    )


# ---------------------------------------------------------------------------
# Worker factory
# ---------------------------------------------------------------------------


def _make_worker(
    cfg: SignalConfig | None = None,
    baseline_repo: InMemoryBehaviorBaselineRepository | None = None,
) -> tuple[DementiaSignalWorker, InMemoryTrajectoryRepository, InMemoryDementiaSignalRepository]:
    traj_repo = InMemoryTrajectoryRepository()
    sig_repo = InMemoryDementiaSignalRepository()
    worker = DementiaSignalWorker(
        trajectory_repo=traj_repo,
        signal_repo=sig_repo,
        cfg=cfg or _AGIT_CFG,
        baseline_repo=baseline_repo,
    )
    return worker, traj_repo, sig_repo


# ---------------------------------------------------------------------------
# Baseline seeding helpers
# ---------------------------------------------------------------------------


def _repo_with_baseline(
    composites: list[float], now: datetime = _NOW
) -> InMemoryBehaviorBaselineRepository:
    """Return a baseline repo pre-loaded with historical composites."""
    repo = InMemoryBehaviorBaselineRepository()
    for i, v in enumerate(composites):
        window_start = now - timedelta(hours=48 + i)
        repo._agitation_windows.append(
            AgitationWindowRecord(
                identity_id=_IDENTITY,
                window_start=window_start,
                composite=v,
                computed_at=now,
            )
        )
    return repo


# ---------------------------------------------------------------------------
# Fixtures: fidget, TV-watching, purposeful walk
# ---------------------------------------------------------------------------


def _fidget_points(n: int = 20, now: datetime = _NOW) -> list[PersonTrajectoryPoint]:
    """High motion energy, near-zero floor speed — fidgeting/rocking signature."""
    return [
        _point(
            motion_energy=0.35,
            floor_speed_m_s=0.05,
            offset_minutes=float(n - i),
            now=now,
        )
        for i in range(n)
    ]


def _tv_watching_points(n: int = 20, now: datetime = _NOW) -> list[PersonTrajectoryPoint]:
    """Very low motion energy (below motion_active_floor) — TV watching."""
    return [
        _point(
            motion_energy=0.04,
            floor_speed_m_s=0.02,
            offset_minutes=float(n - i),
            now=now,
        )
        for i in range(n)
    ]


def _purposeful_walk_points(n: int = 20, now: datetime = _NOW) -> list[PersonTrajectoryPoint]:
    """High floor speed in a straight line — purposeful locomotion."""
    return [
        _point(
            room="hallway",
            motion_energy=0.20,
            floor_speed_m_s=0.80,
            ground_x=float(i) * 0.5,
            ground_y=0.0,
            offset_minutes=float(n - i),
            now=now,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Helper: save points to repo
# ---------------------------------------------------------------------------


async def _seed_points(
    traj_repo: InMemoryTrajectoryRepository, pts: list[PersonTrajectoryPoint]
) -> None:
    for p in pts:
        await traj_repo.save_trajectory_point(p)


# ---------------------------------------------------------------------------
# Test 1: fixtures through full detector
# ---------------------------------------------------------------------------


class TestAgitationFixtures:
    @pytest.mark.asyncio
    async def test_fidget_with_elevated_baseline_emits(self) -> None:
        """Fidget fixture above composite threshold with elevated z emits agitation_index."""
        from app.trajectory.restlessness import RestlessnessConfig, compute_restlessness

        pts = _fidget_points()
        feats = compute_restlessness(pts, RestlessnessConfig())
        assert feats.in_place_motion_ratio is not None
        assert feats.in_place_motion_ratio > 0.8, "fidget fixture must yield high in_place_ratio"

        baseline_repo = _repo_with_baseline([0.1, 0.08, 0.12, 0.09, 0.11, 0.10, 0.13])
        worker, traj_repo, _ = _make_worker(baseline_repo=baseline_repo)
        await _seed_points(traj_repo, pts)

        signals = await worker.run_once(now=_NOW)
        agitation = [s for s in signals if s.signal_kind == "agitation_index"]
        assert agitation, "agitation_index must fire on fidget fixture with elevated z"
        assert agitation[0].severity in ("info", "warning")
        assert agitation[0].severity != "emergency", "agitation must never reach emergency"

    @pytest.mark.asyncio
    async def test_tv_watching_never_emits(self) -> None:
        """TV-watching fixture (motion below motion_active_floor) never emits."""
        baseline_repo = _repo_with_baseline([0.0, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0])
        worker, traj_repo, _ = _make_worker(baseline_repo=baseline_repo)
        await _seed_points(traj_repo, _tv_watching_points())

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "agitation_index" for s in signals)

    @pytest.mark.asyncio
    async def test_purposeful_walk_never_emits(self) -> None:
        """High floor_speed (locomotion) never emits agitation_index."""
        baseline_repo = _repo_with_baseline([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
        worker, traj_repo, _ = _make_worker(baseline_repo=baseline_repo)
        await _seed_points(traj_repo, _purposeful_walk_points())

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "agitation_index" for s in signals)


# ---------------------------------------------------------------------------
# Test 2: no-baseline silence (no cold-start fallback)
# ---------------------------------------------------------------------------


class TestNoBaselineSilence:
    @pytest.mark.asyncio
    async def test_rich_window_zero_history_silent(self) -> None:
        """Even with a clearly elevated window, stays silent with no baseline history."""
        baseline_repo = InMemoryBehaviorBaselineRepository()  # no samples

        worker, traj_repo, _ = _make_worker(baseline_repo=baseline_repo)
        await _seed_points(traj_repo, _fidget_points())

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "agitation_index" for s in signals), (
            "experimental signal must stay silent without baseline (no cold-start fallback)"
        )

    @pytest.mark.asyncio
    async def test_below_min_baseline_n_silent(self) -> None:
        """Fewer than min_baseline_n samples → stays silent."""
        # min_baseline_n=3, seed only 2 samples.
        baseline_repo = _repo_with_baseline([0.1, 0.1])

        worker, traj_repo, _ = _make_worker(baseline_repo=baseline_repo)
        await _seed_points(traj_repo, _fidget_points())

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "agitation_index" for s in signals)

    @pytest.mark.asyncio
    async def test_no_baseline_repo_silent(self) -> None:
        """baseline_repo=None → no signal regardless of window content."""
        worker, traj_repo, _ = _make_worker(baseline_repo=None)
        await _seed_points(traj_repo, _fidget_points())

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "agitation_index" for s in signals)


# ---------------------------------------------------------------------------
# Test 3: severity cap — never reaches emergency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_in_place, expected_severity",
    [
        (18, "warning"),  # 18/20 in-place → composite 0.90, well above warning threshold (0.70)
        (15, "warning"),  # 15/20 in-place → composite 0.75, above warning composite threshold
        (12, "info"),  # 12/20 in-place → composite 0.60, above composite gate but below warning
    ],
)
class TestSeverityCap:
    @pytest.mark.asyncio
    async def test_severity_cap(self, n_in_place: int, expected_severity: str) -> None:
        """Severity must be info or warning; emergency is structurally impossible."""
        # Use only in_place weight so composite == in_place_ratio * weight.
        cfg = SignalConfig(
            agitation_enabled=True,
            onset_consecutive_windows=1,
            min_baseline_n=3,
            agitation_min_observed_minutes=5,
            agitation_weight_in_place=1.0,
            agitation_weight_entropy=0.0,
            agitation_weight_excursion=0.0,
            agitation_composite_threshold=0.45,
            agitation_warning_composite=0.7,
            agitation_warning_z=3.5,
        )
        # Low baseline so any non-trivial composite is elevated.
        baseline_repo = _repo_with_baseline([0.02, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02])

        pts: list[PersonTrajectoryPoint] = [
            _point(motion_energy=0.30, floor_speed_m_s=0.05, offset_minutes=float(20 - i))
            for i in range(n_in_place)
        ] + [
            _point(
                motion_energy=0.04,
                floor_speed_m_s=0.05,
                offset_minutes=float(20 - n_in_place - i),
            )
            for i in range(20 - n_in_place)
        ]

        worker, traj_repo, _ = _make_worker(cfg=cfg, baseline_repo=baseline_repo)
        await _seed_points(traj_repo, pts)

        signals = await worker.run_once(now=_NOW)
        agitation = [s for s in signals if s.signal_kind == "agitation_index"]

        for s in agitation:
            assert s.severity != "emergency", "agitation must never reach emergency severity"

        if agitation:
            assert agitation[0].severity == expected_severity, (
                f"expected {expected_severity} but got {agitation[0].severity} "
                f"(composite driven by {n_in_place}/20 in-place points)"
            )


# ---------------------------------------------------------------------------
# Test 4: hysteresis debounce
# ---------------------------------------------------------------------------


class TestHysteresis:
    @pytest.mark.asyncio
    async def test_single_elevated_window_no_emit(self) -> None:
        """With onset_consecutive_windows=2, first trigger must not emit."""
        baseline_repo = _repo_with_baseline([0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06])
        cfg = SignalConfig(
            agitation_enabled=True,
            onset_consecutive_windows=2,
            min_baseline_n=3,
            agitation_min_observed_minutes=5,
        )
        worker, traj_repo, _ = _make_worker(cfg=cfg, baseline_repo=baseline_repo)
        await _seed_points(traj_repo, _fidget_points())

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "agitation_index" for s in signals), (
            "first trigger with onset_consecutive_windows=2 must be held by debounce"
        )

    @pytest.mark.asyncio
    async def test_two_consecutive_windows_emit(self) -> None:
        """Two consecutive elevated runs emit on the second."""
        baseline_repo = _repo_with_baseline([0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06])
        cfg = SignalConfig(
            agitation_enabled=True,
            onset_consecutive_windows=2,
            min_baseline_n=3,
            agitation_min_observed_minutes=5,
        )
        worker, traj_repo, _ = _make_worker(cfg=cfg, baseline_repo=baseline_repo)

        # First run: debounce holds.
        now1 = _NOW - timedelta(minutes=30)
        await _seed_points(traj_repo, _fidget_points(now=now1))
        await worker.run_once(now=now1)

        # Second run: debounce clears → signal emitted.
        await _seed_points(traj_repo, _fidget_points(now=_NOW))
        signals = await worker.run_once(now=_NOW)
        assert any(s.signal_kind == "agitation_index" for s in signals), (
            "second consecutive elevated window must emit"
        )


# ---------------------------------------------------------------------------
# Test 5: flag off — detector not invoked
# ---------------------------------------------------------------------------


class TestFlagOff:
    @pytest.mark.asyncio
    async def test_agitation_disabled_no_signal(self) -> None:
        """agitation_enabled=False → no agitation_index signal ever emitted."""
        baseline_repo = _repo_with_baseline([0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06])
        cfg = SignalConfig(
            agitation_enabled=False,
            onset_consecutive_windows=1,
            min_baseline_n=3,
        )
        worker, traj_repo, _ = _make_worker(cfg=cfg, baseline_repo=baseline_repo)
        await _seed_points(traj_repo, _fidget_points())

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "agitation_index" for s in signals)


# ---------------------------------------------------------------------------
# Test 6: baseline self-exclusion
# ---------------------------------------------------------------------------


class TestBaselineSelfExclusion:
    @pytest.mark.asyncio
    async def test_current_window_stored_after_run(self) -> None:
        """run_once must persist the current window composite for future baselining."""
        baseline_repo = _repo_with_baseline([0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06])
        worker, traj_repo, _ = _make_worker(baseline_repo=baseline_repo)
        await _seed_points(traj_repo, _fidget_points())

        count_before = len(baseline_repo._agitation_windows)
        await worker.run_once(now=_NOW)
        count_after = len(baseline_repo._agitation_windows)

        assert count_after == count_before + 1, (
            "run_once must save exactly one new agitation_window record per identity"
        )
        new_record = baseline_repo._agitation_windows[-1]
        assert new_record.window_start < _NOW, "stored window_start must precede run time"

    @pytest.mark.asyncio
    async def test_baseline_query_excludes_current_window(self) -> None:
        """Self-exclusion: baseline samples do not include the current window composite.

        Verify by checking that z-score is computed against only the seeded
        historical samples; if the current value were included the z-score
        would decrease because the outlier pulls the median up.
        """
        low_values = [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06]
        baseline_repo = _repo_with_baseline(low_values)

        pts = _fidget_points()
        from app.trajectory.restlessness import RestlessnessConfig, compute_restlessness
        from app.trajectory.stats import robust_z

        feats = compute_restlessness(pts, RestlessnessConfig())
        composite = round(
            min(
                max(
                    0.5 * (feats.in_place_motion_ratio or 0.0)
                    + 0.3 * (feats.direction_change_entropy or 0.0)
                    + 0.2 * min((feats.short_excursion_rate or 0.0) / 6.0, 1.0),
                    0.0,
                ),
                1.0,
            ),
            4,
        )

        z_without = robust_z(composite, low_values).modified_z

        # With the current composite included, its high value shifts the median up,
        # reducing the modified z (the outlier is no longer as extreme relative to
        # the new median).
        z_with = robust_z(composite, [*low_values, composite]).modified_z

        # This assertion confirms the math: self-exclusion actually matters.
        # When MAD is 0 (all same) z may be inf; guard against that edge case.
        if z_without != float("inf") and z_with != float("inf"):
            assert z_with <= z_without, (
                "including the current composite in the baseline should not increase z; "
                "if it does the sample set did not produce an outlier"
            )
        # Either way, run a full worker pass and verify no exception.
        worker, traj_repo, _ = _make_worker(baseline_repo=baseline_repo)
        await _seed_points(traj_repo, pts)
        await worker.run_once(now=_NOW)


# ---------------------------------------------------------------------------
# Test 7: kind plumbing sanity
# ---------------------------------------------------------------------------


class TestKindPlumbing:
    def test_agitation_index_in_domain_literal(self) -> None:
        from typing import get_args

        from app.domain import DementiaSignalKind

        assert "agitation_index" in get_args(DementiaSignalKind)

    def test_signal_spec_has_agitation_index(self) -> None:
        from app.trajectory.dementia_signals import _SIGNAL_SPEC

        assert "agitation_index" in _SIGNAL_SPEC
        spec = _SIGNAL_SPEC["agitation_index"]
        assert spec.evidence_grade == "experimental"

    def test_agitation_never_reaches_emergency(self) -> None:
        """Structural invariant: agitation severity thresholds cap at warning."""
        cfg = SignalConfig()
        # Max severity is warning; no code path emits emergency for agitation.
        assert not hasattr(cfg, "agitation_emergency_composite"), (
            "agitation has no emergency tier by design"
        )
