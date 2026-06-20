"""Tracker-level M03 tests: PH-local appearance contamination guard + metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from prometheus_client import CollectorRegistry

from app.domain import BoundingBox, FloorPoint, OrientationBin, WorldObservation
from app.observability import metrics as _metrics
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

_ROOM = {"room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}


def _unit(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(8).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _obs(
    x_m: float,
    detection_id: str,
    captured_at: datetime,
    embedding: list[float],
    *,
    orientation: OrientationBin = OrientationBin.FRONT,
) -> WorldObservation:
    return WorldObservation(
        camera_id="cam-1",
        frame_index=1,
        captured_at=captured_at,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=5000, calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=embedding,
        detection_confidence=0.92,
        detection_id=detection_id,
        quality=0.8,
        orientation=orientation,
        orientation_confidence=0.9,
    )


async def _build_ph_with_prototype(tracker: WorldTracker, t0: datetime, emb: list[float]):
    """Spawn a PH and establish a FRONT prototype over three consistent frames."""
    for k in range(3):
        await tracker.step(
            [_obs(5.0 + 0.05 * k, f"d{k}", t0 + timedelta(seconds=k), emb)],
            now=t0 + timedelta(seconds=k),
            room_polygons=_ROOM,
        )


async def test_cross_person_outlier_does_not_pollute_but_state_advances() -> None:
    cfg = WorldTrackerConfig(dedup_enabled=False, min_observations_to_publish=1)
    ph_repo = InMemoryPHRepository()
    tracker = WorldTracker(
        ph_repo=ph_repo, obs_repo=InMemoryWorldObservationRepository(), config=cfg
    )
    t0 = datetime(2026, 6, 20, 9, 0, 0, tzinfo=UTC)
    person = _unit(1)
    await _build_ph_with_prototype(tracker, t0, person)

    phs = await ph_repo.list_open()
    assert len(phs) == 1
    before = phs[0]
    assert before.gallery_mean is not None
    proto_before = {p.orientation: (p.embedding, p.count) for p in before.view_prototypes}
    assert OrientationBin.FRONT in proto_before

    # Frame 4: geometrically valid but an opposite-person embedding (cosine -1).
    outlier = (-np.asarray(person, dtype=np.float32)).tolist()
    t4 = t0 + timedelta(seconds=3)
    await tracker.step(
        [_obs(5.2, "d3", t4, outlier)],
        now=t4,
        room_polygons=_ROOM,
    )
    after = (await ph_repo.list_open())[0]

    # Appearance state untouched.
    np.testing.assert_allclose(after.gallery_mean, before.gallery_mean, atol=1e-9)
    proto_after = {p.orientation: (p.embedding, p.count) for p in after.view_prototypes}
    assert proto_after == proto_before
    # But the match advanced the Kalman state and the observation count.
    assert after.observation_count == before.observation_count + 1
    assert after.state_mean[0] > before.state_mean[0]


async def test_primary_pass_emits_batch_skew_and_outcome(monkeypatch) -> None:
    registry = CollectorRegistry()
    monkeypatch.setattr(_metrics, "metrics", _metrics.build_metrics(registry))

    cfg = WorldTrackerConfig(dedup_enabled=False, min_observations_to_publish=1)
    tracker = WorldTracker(
        ph_repo=InMemoryPHRepository(), obs_repo=InMemoryWorldObservationRepository(), config=cfg
    )
    t0 = datetime(2026, 6, 20, 9, 0, 0, tzinfo=UTC)
    # captured 100 ms before the batch now → a measurable skew.
    obs = _obs(5.0, "d0", t0 - timedelta(milliseconds=100), _unit(1))
    await tracker.step([obs], now=t0, room_polygons=_ROOM)

    skew = registry.get_sample_value("cts_worldtracker_batch_skew_ms_count", {"camera_id": "cam-1"})
    assert skew == 1.0
    spawned = registry.get_sample_value(
        "cts_worldtracker_association_outcome_total", {"outcome": "unmatched_obs"}
    )
    # First frame has no PHs yet, so the observation is unmatched (then spawns).
    assert spawned == 1.0
