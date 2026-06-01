"""Held-identity confidence replay on coasting frames.

On a frame where a committed PH receives no observation (coasting / unresolved),
the resolver produces no per-frame posterior for it, so the snapshot confidence
would default to 0.0 (a sentinel the UI renders as null). WorldTracker caches the
last positive committed confidence and replays it, so a held identity keeps a
meaningful top_probability instead of flickering to null.

Pure unit test: all-InMemory repos.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import BoundingBox, FaceAnchor, FloorPoint, Identity, WorldObservation
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.orientation import OrientationBin
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

BASE = datetime(2026, 6, 2, 9, 0, 0, tzinfo=UTC)
_EMB = tuple(0.05 * (i % 7) for i in range(768))


def _grandma_obs(k: int) -> WorldObservation:
    t = BASE + timedelta(seconds=0.5 * k)
    return WorldObservation(
        camera_id="cam01",
        frame_index=k,
        captured_at=t,
        floor_point=FloorPoint(x_mm=1000.0, y_mm=1000.0, calibrated=True),
        bbox=BoundingBox(x_min=0.4, y_min=0.3, x_max=0.5, y_max=0.7),
        embedding=_EMB,
        detection_confidence=0.9,
        height_estimate_m=1.65,
        face_anchor=FaceAnchor(
            person_id="grandma",
            confidence=0.95,
            quality=0.9,
            detection_id=f"d{k}",
            camera_id="cam01",
            captured_at=t,
            recognition_state="recognized",
            similarity=0.95,
            yaw_deg=5.0,
        ),
        detection_id=f"d{k}",
        quality=0.8,
        orientation=OrientationBin.FRONT,
        orientation_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_coasting_frame_replays_held_confidence() -> None:
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(
        Identity(identity_id="grandma", display_name="grandma", enrolled_at=BASE)
    )
    resolver = IdentityResolver(
        gallery_repo=gallery, config=ResolverConfig(enable_sticky_maintenance=True)
    )
    tracker = WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=WorldTrackerConfig(),
        identity_resolver=resolver,
    )

    # Observe grandma for several frames so the PH commits and clears the
    # min-observations publish gate.
    last_conf = 0.0
    for k in range(5):
        obs = _grandma_obs(k)
        result = await tracker.step(
            observations=[obs], now=obs.captured_at, face_anchors=[obs.face_anchor]
        )
        for snap in result.snapshots:
            if snap.identity_id == "grandma":
                last_conf = snap.identity_confidence
    assert last_conf > 0.0, "grandma should resolve with a positive confidence"

    # Coasting frame: no observations, within close grace -> PH stays open and
    # is unresolved this frame.
    coast = await tracker.step(observations=[], now=BASE + timedelta(seconds=3.0))
    grandma_snaps = [s for s in coast.snapshots if s.identity_id == "grandma"]
    assert grandma_snaps, "coasting grandma PH should still emit a snapshot"
    # The held confidence is replayed, not a sentinel 0.0.
    assert grandma_snaps[0].identity_confidence == pytest.approx(last_conf)
    assert grandma_snaps[0].identity_confidence > 0.0


@pytest.mark.asyncio
async def test_cache_evicts_ph_removed_outside_tracker() -> None:
    """Leak guard: a PH closed/merged/deleted OUTSIDE the tracker (e.g. the PH
    API) no longer appears in list_open, so it never passes through step 7. The
    per-frame intersect with the open set must still evict it, keeping the cache
    bounded to open PHs. Reads the private cache because the memory-bound
    invariant has no other observable surface."""
    ph_repo = InMemoryPHRepository()
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(
        Identity(identity_id="grandma", display_name="grandma", enrolled_at=BASE)
    )
    resolver = IdentityResolver(gallery_repo=gallery, config=ResolverConfig())
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=InMemoryWorldObservationRepository(),
        config=WorldTrackerConfig(),
        identity_resolver=resolver,
    )

    for k in range(5):
        obs = _grandma_obs(k)
        await tracker.step(observations=[obs], now=obs.captured_at, face_anchors=[obs.face_anchor])

    open_phs = await ph_repo.list_open()
    assert open_phs, "grandma PH should be open"
    ph_id = open_phs[0].ph_id
    assert ph_id in tracker._last_identity_confidence

    # Simulate an external close/merge: the PH leaves list_open without passing
    # through the tracker's own close path.
    await ph_repo.save(dataclasses.replace(open_phs[0], closed_at=BASE + timedelta(seconds=2.5)))
    assert not await ph_repo.list_open()

    # Any subsequent frame must drop the now-orphaned cache entry.
    await tracker.step(observations=[], now=BASE + timedelta(seconds=3.0))
    assert ph_id not in tracker._last_identity_confidence
    assert tracker._last_identity_confidence == {}
