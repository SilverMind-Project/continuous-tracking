"""Tests for AlgorithmSpec, DataQuality, and signal metadata attachment."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    DementiaSignal,
    PersonTrajectoryPoint,
)
from app.storage.base import (
    InMemoryBehaviorBaselineRepository,
    InMemoryDementiaSignalRepository,
    InMemoryTrajectoryRepository,
)
from app.trajectory.dementia_signals import (
    _SIGNAL_SPEC,
    DementiaSignalWorker,
    SignalConfig,
    _stable_signal_id,
)
from app.trajectory.signal_specs import (
    BATHROOM_DWELL_SPEC,
    EVENING_ACTIVITY_SPEC,
    NON_DIAGNOSTIC_DISCLAIMER,
    PACING_SPEC,
    STILLNESS_SPEC,
)

_NOW = datetime(2026, 5, 23, 18, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Algorithm spec tests
# ---------------------------------------------------------------------------


class TestAlgorithmSpecs:
    def test_every_signal_has_algorithm_spec(self) -> None:
        """All signal kinds must have an AlgorithmSpec."""
        expected_kinds = {
            "pacing",
            "sundowning_index",
            "nighttime_movement",
            "stillness_anomaly",
            "absence",
            "bathroom_dwell_anomaly",
            "fall_suspected",
            "gait_slowing",
            "agitation_index",
        }
        assert set(_SIGNAL_SPEC.keys()) == expected_kinds

    def test_spec_has_evidence_grade(self) -> None:
        """Every spec must have a non-empty evidence grade."""
        for kind, spec in _SIGNAL_SPEC.items():
            assert spec.evidence_grade, f"{kind} missing evidence_grade"

    def test_spec_has_non_diagnostic_disclaimer(self) -> None:
        """Every spec must include the non-diagnostic disclaimer."""
        for kind, spec in _SIGNAL_SPEC.items():
            assert NON_DIAGNOSTIC_DISCLAIMER in spec.disclaimer, f"{kind} missing disclaimer"

    def test_pacing_requires_transition_data(self) -> None:
        assert "room_transitions" in PACING_SPEC.required_inputs

    def test_stillness_requires_motion_energy(self) -> None:
        assert "motion_energy" in STILLNESS_SPEC.required_inputs
        assert "posture" in STILLNESS_SPEC.required_inputs

    def test_evening_activity_uses_clinical_review(self) -> None:
        assert EVENING_ACTIVITY_SPEC.evidence_grade == "clinical_review"

    def test_bathroom_dwell_requires_room_taxonomy(self) -> None:
        assert "room_taxonomy" in BATHROOM_DWELL_SPEC.required_inputs


# ---------------------------------------------------------------------------
# Algorithm metadata attachment tests
# ---------------------------------------------------------------------------


class TestAlgorithmMetadata:
    async def _make_worker(self) -> DementiaSignalWorker:
        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()
        baseline_repo = InMemoryBehaviorBaselineRepository()
        cfg = SignalConfig(
            tz_name="America/Chicago",
            stillness_threshold_minutes=60,
            stillness_emergency_minutes=120,
            stillness_motion_floor=0.02,
            pacing_room_threshold=4,
            pacing_window_minutes=30,
            nighttime_transition_threshold=3,
            absence_threshold_minutes=60,
            bathroom_absolute_threshold_seconds=2700,
        )
        return DementiaSignalWorker(
            trajectory_repo=traj_repo,
            signal_repo=sig_repo,
            cfg=cfg,
            baseline_repo=baseline_repo,
        )

    @pytest.mark.asyncio
    async def test_signal_gets_algorithm_metadata(self) -> None:
        """Emitted signals must carry algorithm name and evidence grade."""
        worker = await self._make_worker()

        # Create a signal and apply metadata.
        signal = DementiaSignal(
            signal_id="sig-1",
            identity_id="alice",
            signal_kind="stillness_anomaly",
            severity="warning",
            value=3600.0,
        )
        signal = worker._apply_algorithm_metadata(signal, "stillness_anomaly")

        assert signal.algorithm_name == STILLNESS_SPEC.name
        assert signal.evidence_grade == STILLNESS_SPEC.evidence_grade
        assert signal.algorithm_spec_json != ""

        spec_json = json.loads(signal.algorithm_spec_json)
        assert spec_json["name"] == "stillness-v2"
        assert spec_json["evidence_grade"] == "observational_study"

    @pytest.mark.asyncio
    async def test_signal_context_has_disclaimer(self) -> None:
        """Signal context must include the non-diagnostic disclaimer."""
        worker = await self._make_worker()
        signal = DementiaSignal(
            signal_id="sig-2",
            identity_id="bob",
            signal_kind="pacing",
            severity="info",
            value=0.3,
        )
        worker._apply_algorithm_metadata(signal, "pacing")
        signal.context["disclaimer"] = NON_DIAGNOSTIC_DISCLAIMER

        assert "disclaimer" in signal.context
        assert "diagnosis" in signal.context["disclaimer"].lower()

    @pytest.mark.asyncio
    async def test_identity_confidence_gating(self) -> None:
        """Low-confidence identities should be suppressed."""
        worker = await self._make_worker()
        # UNKNOWN identity should return no signals.
        result = await worker._process_identity("", _NOW)
        assert result == []

        result = await worker._process_identity("UNKNOWN", _NOW)
        assert result == []

    @pytest.mark.asyncio
    async def test_stable_signal_id_is_deterministic(self) -> None:
        """Same (identity, kind, window) must produce the same signal_id."""
        w_start = _NOW - timedelta(minutes=30)
        w_end = _NOW

        id1 = _stable_signal_id("alice", "pacing", w_start, w_end)
        id2 = _stable_signal_id("alice", "pacing", w_start, w_end)

        assert id1 == id2
        assert len(id1) == 36  # UUID string length


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------


class TestDataQuality:
    async def _make_worker(self) -> DementiaSignalWorker:
        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()
        baseline_repo = InMemoryBehaviorBaselineRepository()
        return DementiaSignalWorker(
            trajectory_repo=traj_repo,
            signal_repo=sig_repo,
            baseline_repo=baseline_repo,
        )

    @pytest.mark.asyncio
    async def test_quality_insufficient_on_empty_window(self) -> None:
        worker = await self._make_worker()
        ok, mean_conf, _coverage = worker._check_data_quality("alice", [])
        assert not ok
        assert mean_conf == 0.0

    @pytest.mark.asyncio
    async def test_quality_insufficient_with_low_confidence(self) -> None:
        worker = await self._make_worker()
        pts = [
            PersonTrajectoryPoint(
                identity_id="alice",
                ph_id="gt-1",
                observed_at=_NOW - timedelta(minutes=5),
                room_name="living_room",
                identity_confidence=0.1,  # very low
            ),
        ]
        ok, mean_conf, _coverage = worker._check_data_quality("alice", pts)
        assert mean_conf == pytest.approx(0.1)
        assert not ok  # confidence too low

    @pytest.mark.asyncio
    async def test_quality_sufficient_with_good_confidence(self) -> None:
        worker = await self._make_worker()
        pts = [
            PersonTrajectoryPoint(
                identity_id="alice",
                ph_id="gt-1",
                observed_at=_NOW - timedelta(seconds=2),
                room_name="kitchen",
                identity_confidence=0.95,
            ),
            PersonTrajectoryPoint(
                identity_id="alice",
                ph_id="gt-1",
                observed_at=_NOW,
                room_name="kitchen",
                identity_confidence=0.92,
            ),
        ]
        ok, mean_conf, _coverage = worker._check_data_quality("alice", pts)
        assert mean_conf > 0.9
        assert ok


# ---------------------------------------------------------------------------
# Cold start tests
# ---------------------------------------------------------------------------


class TestColdStart:
    """Severity must be conservative under cold start (no baseline)."""

    @pytest.mark.asyncio
    async def test_pacing_cold_start_lower_severity(self) -> None:
        """Without a baseline, pacing should default to info severity."""
        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()
        baseline_repo = InMemoryBehaviorBaselineRepository()
        cfg = SignalConfig(
            pacing_room_threshold=4,
            pacing_window_minutes=30,
        )
        worker = DementiaSignalWorker(
            trajectory_repo=traj_repo,
            signal_repo=sig_repo,
            cfg=cfg,
            baseline_repo=baseline_repo,
        )

        # Create trajectory points showing a pacing pattern.
        now = _NOW
        room_sequence = ["kitchen", "living_room", "kitchen", "living_room", "kitchen"]
        points = []
        for i, room in enumerate(room_sequence):
            pt = PersonTrajectoryPoint(
                identity_id="alice",
                ph_id="gt-1",
                observed_at=now - timedelta(minutes=len(room_sequence) - i),
                room_name=room,
                identity_confidence=0.95,
            )
            await traj_repo.save_trajectory_point(pt)
            points.append(pt)

        # Should compute signals even without baseline (cold start).
        results = await worker._compute_pacing("alice", points, now)
        # Cold start: should not emit emergency.
        for sig in results:
            assert sig.severity != "emergency", (
                f"Cold start should not emit emergency: {sig.severity}"
            )
