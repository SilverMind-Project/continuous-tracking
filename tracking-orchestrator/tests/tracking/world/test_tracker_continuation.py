"""PH continuation publisher tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    BoundingBox,
    FloorPoint,
    PersonHypothesis,
    PHContinuationCandidate,
    WorldObservation,
)
from app.observability import metrics as metrics_pkg
from app.observability.metrics import build_metrics
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker


class _FakeContinuationPublisher:
    def __init__(self) -> None:
        self.published: list[PHContinuationCandidate] = []

    async def publish(self, candidate: PHContinuationCandidate) -> None:
        self.published.append(candidate)


def _obs(captured_at: datetime) -> WorldObservation:
    return WorldObservation(
        camera_id="cam-b",
        frame_index=10,
        captured_at=captured_at,
        floor_point=FloorPoint(x_mm=1200, y_mm=1000, calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=[1.0, 0.0],
        detection_confidence=0.95,
        detection_id="det-new",
        quality=0.8,
    )


def _closed_ph(closed_at: datetime) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id="ph-old",
        state_mean=(1.0, 1.0, 0.0, 0.0),
        state_cov=(0.0,) * 16,
        born_at=closed_at - timedelta(seconds=30),
        last_seen_at=closed_at,
        last_seen_camera="cam-a",
        observation_count=5,
        current_identity_id="alice",
        current_identity_committed_at=closed_at - timedelta(seconds=20),
        active_cameras=frozenset({"cam-a"}),
        closed_at=closed_at,
        mean_quality=0.8,
    )


@pytest.mark.asyncio
async def test_non_overlapping_handoff_publishes_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_metrics = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh_metrics)
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    now = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    await ph_repo.save(_closed_ph(now - timedelta(seconds=10)))
    publisher = _FakeContinuationPublisher()
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=obs_repo,
        config=WorldTrackerConfig(
            dedup_enabled=False,
            min_observations_to_publish=1,
            inferred_handoff_max_s=60.0,
            inferred_handoff_max_distance_m=5.0,
        ),
        continuation_publisher=publisher,
    )

    result = await tracker.step([_obs(now)], now=now)

    assert len(result.continuations) == 1
    assert publisher.published == result.continuations
    candidate = result.continuations[0]
    assert candidate.predecessor_ph_id == "ph-old"
    assert candidate.predecessor_identity_id == "alice"
    assert candidate.seconds_elapsed == pytest.approx(10.0)
    assert fresh_metrics.world_tracker_continuations_total._value.get() == 1.0
