"""WTR10: Replay — identity revision after false commit.

When an initial identity commit is later revised, the PH must be updated
and a revision record must be generated.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import (
    BoundingBox,
    FloorPoint,
    Identity,
    PersonHypothesis,
    WorldObservation,
)
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
    InMemoryPHRepository,
    InMemoryTrackingRepository,
    InMemoryWorldObservationRepository,
)
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker


def _make_obs(
    camera_id: str, frame_index: int, fx: float, fy: float,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=int(fx * 1000), y_mm=int(fy * 1000), calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=[1.0, 0.0],
        detection_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_identity_correction_updates_ph():
    """Manual identity correction must update the PH's current_identity_id."""
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()

    now = datetime.now(UTC)
    ph = PersonHypothesis(
        ph_id="ph-1",
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=5,
        current_identity_id=None,
        active_cameras=frozenset(["cam-1"]),
    )
    await ph_repo.save(ph)

    # Apply a manual correction.
    revision = await ph_repo.correct_identity(
        ph_id="ph-1",
        new_identity_id="alice",
        reason="manual correction after false commit",
        actor="practitioner",
    )
    assert revision is not None
    assert revision.ph_id == "ph-1"
    assert revision.new_identity_id == "alice"
    assert revision.actor == "practitioner"

    # PH must now have the corrected identity.
    updated = await ph_repo.get_by_id("ph-1")
    assert updated is not None
    assert updated.current_identity_id == "alice"
    assert updated.current_identity_committed_at is not None


@pytest.mark.asyncio
async def test_revision_recorded_in_revision_list():
    """Revisions must appear in the PH's revision history."""
    ph_repo = InMemoryPHRepository()

    now = datetime.now(UTC)
    ph = PersonHypothesis(
        ph_id="ph-2",
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=5,
        current_identity_id="old_bob",
        active_cameras=frozenset(["cam-1"]),
    )
    await ph_repo.save(ph)

    await ph_repo.correct_identity(
        ph_id="ph-2",
        new_identity_id="corrected_bob",
        reason="identity disagreement resolved",
        actor="practitioner",
    )

    revisions, has_more = await ph_repo.list_revisions(ph_id="ph-2", limit=10)
    assert len(revisions) >= 1
    assert revisions[0].previous_identity_id == "old_bob"
    assert revisions[0].new_identity_id == "corrected_bob"
