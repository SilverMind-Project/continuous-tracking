"""WTR6: Merge/split validation tests for InMemoryPHRepository."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import BoundingBox, FloorPoint, PersonHypothesis, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository


def _make_ph(ph_id: str, cameras: frozenset[str] | None = None) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=5,
        current_identity_id=None,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=cameras or frozenset(["cam-1"]),
    )


def _make_obs(camera_id: str = "cam-1", frame_index: int = 1) -> WorldObservation:
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
async def test_overlap_validation_blocks_unsafe_merge():
    """Cannot merge PHs with overlapping same-camera observations."""
    repo = InMemoryPHRepository()

    ph1 = _make_ph("ph-1", cameras=frozenset(["cam-1", "cam-2"]))
    ph2 = _make_ph("ph-2", cameras=frozenset(["cam-1"]))  # overlap on cam-1

    await repo.save(ph1)
    await repo.save(ph2)

    with pytest.raises(ValueError, match="overlapping camera"):
        await repo.merge(
            source_ph_id="ph-1",
            target_ph_id="ph-2",
            actor="admin",
            reason="test",
        )


@pytest.mark.asyncio
async def test_split_at_first_observation_is_rejected():
    """Cannot split at the first observation (only one observation == first)."""
    repo = InMemoryPHRepository()

    ph = _make_ph("ph-1")
    await repo.save(ph)

    obs = _make_obs()
    obs_repo = InMemoryWorldObservationRepository()
    oid = await obs_repo.save(obs, "ph-1")
    # Inject the persisted observation into the PH repo's internal observation list.
    repo._observations["ph-1"] = await obs_repo.list_by_ph("ph-1", limit=1)

    with pytest.raises(ValueError, match="first observation"):
        await repo.split(
            ph_id="ph-1",
            at_observation_id=oid,
            actor="admin",
            reason="test",
        )


@pytest.mark.asyncio
async def test_cannot_correct_closed_ph():
    """Cannot correct a closed PH."""
    repo = InMemoryPHRepository()

    now = datetime.now(UTC)
    ph = PersonHypothesis(
        ph_id="ph-1",
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=5,
        current_identity_id=None,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=frozenset(["cam-1"]),
        closed_at=now,  # already closed
    )
    await repo.save(ph)

    with pytest.raises(ValueError, match="closed"):
        await repo.correct_identity(
            ph_id="ph-1",
            new_identity_id="alice",
            reason="test",
            actor="admin",
        )
