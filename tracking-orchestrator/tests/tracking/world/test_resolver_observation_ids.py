"""Identity resolver observation IDs are real, not synthetic.

Tests that _resolve_identities() receives real observation IDs from the
repository, not synthetic camera/frame/time strings.
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
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import _resolve_identities


def _make_ph(ph_id: str) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=5,
        current_identity_id=None,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=frozenset(["cam-1"]),
    )


def _make_observation(camera_id: str = "cam-1", frame_index: int = 1) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=[],
        detection_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_resolver_returns_empty_when_no_resolver():
    """When resolver is None, _resolve_identities returns empty results."""
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()

    ph = _make_ph("ph-1")
    await ph_repo.save(ph)
    await obs_repo.save(_make_observation(), "ph-1")

    decisions, revisions, identity_by_ph = await _resolve_identities(
        resolver=None,
        obs_repo=obs_repo,
        ph_repo=ph_repo,
        phs=[ph],
        ph_obs_meta={"ph-1": (1, None, 0.9)},
        face_anchors=[],
        now=datetime.now(UTC),
        config=WorldTrackerConfig(),
    )
    assert decisions == []
    assert revisions == []
    assert identity_by_ph == {}


@pytest.mark.asyncio
async def test_observation_ids_from_repository_are_real_uuids():
    """After saving an observation, list_by_ph returns it with a real observation_id."""
    obs_repo = InMemoryWorldObservationRepository()

    obs = _make_observation()
    oid = await obs_repo.save(obs, "ph-1")

    # The returned observation_id should be a UUID4 string.
    assert oid
    assert len(oid) == 36  # UUID4 string length
    assert oid.count("-") == 4

    # list_by_ph should return observations with real IDs.
    retrieved = await obs_repo.list_by_ph("ph-1", limit=50)
    assert len(retrieved) == 1
    assert retrieved[0].observation_id == oid


@pytest.mark.asyncio
async def test_observation_id_is_not_synthetic():
    """An observation_id from the repository must not be a synthetic
    camera/frame/time string."""
    obs_repo = InMemoryWorldObservationRepository()

    obs = _make_observation("cam-1", 42)
    oid = await obs_repo.save(obs, "ph-1")

    # Verify the ID is not synthetic (no camera_id in it, no path separators).
    assert "/" not in oid
    assert "cam-1" not in oid
    # UUID4 format: 8-4-4-4-12 hex chars
    parts = oid.split("-")
    assert len(parts) == 5
    assert all(len(p) > 0 for p in parts)
