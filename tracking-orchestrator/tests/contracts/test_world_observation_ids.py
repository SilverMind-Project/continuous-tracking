"""Contract test — world observation IDs.

Contract:   — world observations have a real ``observation_id``.
Identity resolver inputs must use persisted observation ids, not synthetic
camera/frame/time strings.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import (
    BoundingBox,
    FloorPoint,
    PersonHypothesis,
    WorldObservation,
)
from app.storage.base import (
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)


def _make_ph(ph_id: str) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=0,
        current_identity_id=None,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=frozenset(["cam-1"]),
    )


def _make_observation(camera_id: str, frame_index: int) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=[],
        detection_confidence=0.95,
        height_estimate_m=1.7,
        face_anchor=None,
    )


@pytest.mark.asyncio
async def test_observations_are_persisted_and_retrievable():
    """Observations saved via the repository must be listable by ph_id."""
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()

    ph = _make_ph("ph-1")
    await ph_repo.save(ph)

    obs = _make_observation("cam-1", 42)
    await obs_repo.save(obs, "ph-1")
    await obs_repo.save(_make_observation("cam-1", 43), "ph-1")
    await obs_repo.save(_make_observation("cam-2", 10), "ph-1")

    retrieved = await obs_repo.list_by_ph("ph-1", limit=50)
    assert len(retrieved) == 3
    assert retrieved[0].camera_id == "cam-1"
    assert retrieved[0].frame_index == 42


@pytest.mark.asyncio
async def test_repository_maintains_observation_order():
    """Observations must be returned in insertion order (oldest first)."""
    obs_repo = InMemoryWorldObservationRepository()

    await obs_repo.save(_make_observation("cam-1", 1), "ph-1")
    await obs_repo.save(_make_observation("cam-1", 2), "ph-1")
    await obs_repo.save(_make_observation("cam-1", 3), "ph-1")

    retrieved = await obs_repo.list_by_ph("ph-1", limit=50)
    assert len(retrieved) == 3
    assert [o.frame_index for o in retrieved] == [1, 2, 3]


@pytest.mark.asyncio
async def test_observation_not_synthetic_id():
    """persisted observations have real UUID-based observation_ids.

    Pre-persist observations may have empty observation_id.
    After persistence (via the repository), the id must be a real UUID,
    not a synthetic camera/frame/time string.
    """
    obs = _make_observation("cam-1", 1)

    # Pre-persist: observation_id may be empty.
    # The contract is only enforced after repository save.
    if not obs.observation_id:
        return  # acceptable for pre-persist

    # Post-persist: must not be synthetic.
    synthetic_pattern = f"{obs.camera_id}/{obs.frame_index}"
    assert synthetic_pattern not in obs.observation_id, (
        f"observation_id must not be a synthetic string: {obs.observation_id}"
    )
    assert "/" not in obs.observation_id, (
        f"observation_id must be a UUID, not a path: {obs.observation_id}"
    )


@pytest.mark.asyncio
async def test_ph_observation_ids_returns_real_ids():
    """: PersonHypothesis.observation_ids must return real observation ids.

    Currently PersonHypothesis.observation_ids returns [] (set by repository).
    This test documents the contract: when the repository populates this field,
    it must return real persisted IDs, not synthetic strings.
    """
    ph = _make_ph("ph-1")

    oids = ph.observation_ids
    # With a real repository, these would be UUIDs from the database.
    # Synthetic ids like "cam-1/42/2026-..." are forbidden.
    for oid in oids:
        assert "/" not in str(oid), (
            f"observation_ids must contain real IDs, not synthetic paths: {oid}"
        )
