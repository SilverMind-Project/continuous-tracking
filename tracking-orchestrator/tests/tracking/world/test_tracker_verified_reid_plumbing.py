"""M12: verified-ReID identity plumbing into association.

M03 built the ``reid_disagreement_cost`` path in the cost matrix and the
``associate(obs_verified_reid_identity_ids=...)`` parameter, but no WorldTracker
call site fed the per-observation verified-ReID identity (the M03->M05 handoff
gap). M12 closes it: when ``enable_reid_disagreement_cost`` is true,
``WorldTracker.step`` resolves each observation's identity from the
``operator_verified`` gallery and passes it into the primary ``associate`` call,
so a body whose verified-ReID identity disagrees with a PH's committed identity
pays the disagreement cost.

These tests prove the plumbing (the cost arithmetic itself is covered by
``tests/tracking/world/test_cost_matrix.py``):

* only ``operator_verified`` gallery entries vote;
* a sub-threshold match contributes no identity;
* the resolved ids reach ``associate`` when the flag is on, and the input stays
  ``None`` (zero extra gallery queries) when the flag is off.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from app.domain import (
    BoundingBox,
    FloorPoint,
    GalleryEmbedding,
    Identity,
    OrientationBin,
    WorldObservation,
)
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.world import tracker as tracker_module
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

_ROOM = {"room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}
NOW = datetime(2026, 6, 23, 9, 0, 0, tzinfo=UTC)


def _unit(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(8).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _obs(detection_id: str, embedding: list[float], x_m: float = 5.0) -> WorldObservation:
    return WorldObservation(
        camera_id="cam-1",
        frame_index=1,
        captured_at=NOW,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=5000, calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=embedding,
        detection_confidence=0.92,
        detection_id=detection_id,
        quality=0.8,
        orientation=OrientationBin.FRONT,
        orientation_confidence=0.9,
    )


async def _add_entry(
    gallery: InMemoryGalleryRepository,
    entry_id: str,
    identity_id: str,
    embedding: list[float],
    state: str,
) -> None:
    # search_similar(active_only=True) only returns entries whose identity is a
    # registered active identity, so register the identity before the entry.
    await gallery.upsert_identity(
        Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=NOW)
    )
    await gallery.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id=entry_id,
            identity_id=identity_id,
            embedding=embedding,
            seen_at=NOW,
            state=state,
        )
    )


def _tracker(gallery: InMemoryGalleryRepository, cfg: WorldTrackerConfig) -> WorldTracker:
    return WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=cfg,
        gallery_repo=gallery,
    )


async def test_only_operator_verified_entries_vote() -> None:
    """A pending_review gallery match must not contribute a verified-ReID id."""
    amma = _unit(1)
    grandma = _unit(2)
    gallery = InMemoryGalleryRepository()
    await _add_entry(gallery, "g1", "amma", amma, "operator_verified")
    await _add_entry(gallery, "g2", "grandma", grandma, "pending_review")
    tracker = _tracker(gallery, WorldTrackerConfig(enable_reid_disagreement_cost=True))

    resolved = await tracker._resolve_verified_reid_identities(
        [
            _obs("d-amma", amma),  # matches a verified entry
            _obs("d-grandma", grandma),  # matches only a pending entry
            _obs("d-none", []),  # no embedding
        ]
    )

    assert resolved == ["amma", None, None]


async def test_sub_threshold_match_contributes_no_identity() -> None:
    """A verified entry whose similarity is below the threshold does not vote."""
    amma = _unit(1)
    orthogonal = (-np.asarray(amma, dtype=np.float32)).tolist()  # cosine -1
    gallery = InMemoryGalleryRepository()
    await _add_entry(gallery, "g1", "amma", amma, "operator_verified")
    tracker = _tracker(gallery, WorldTrackerConfig(enable_reid_disagreement_cost=True))

    resolved = await tracker._resolve_verified_reid_identities([_obs("d1", orthogonal)])

    assert resolved == [None]


async def test_step_feeds_verified_reid_ids_to_associate_when_flag_on(monkeypatch) -> None:
    amma = _unit(1)
    gallery = InMemoryGalleryRepository()
    await _add_entry(gallery, "g1", "amma", amma, "operator_verified")
    tracker = _tracker(gallery, WorldTrackerConfig(enable_reid_disagreement_cost=True))

    captured: dict[str, object] = {}
    real_associate = tracker_module.associate

    def _spy(*args, **kwargs):
        captured["obs_verified_reid_identity_ids"] = kwargs.get("obs_verified_reid_identity_ids")
        return real_associate(*args, **kwargs)

    # First frame spawns a PH (no PH to associate against yet); the second frame
    # exercises the primary associate() call with an open PH present.
    monkeypatch.setattr(tracker_module, "associate", _spy)
    await tracker.step([_obs("d1", amma)], now=NOW, room_polygons=_ROOM)
    await tracker.step([_obs("d2", amma)], now=NOW + timedelta(seconds=1), room_polygons=_ROOM)

    assert captured["obs_verified_reid_identity_ids"] == ["amma"]


async def test_step_passes_none_when_flag_off(monkeypatch) -> None:
    amma = _unit(1)
    gallery = InMemoryGalleryRepository()
    await _add_entry(gallery, "g1", "amma", amma, "operator_verified")
    tracker = _tracker(gallery, WorldTrackerConfig(enable_reid_disagreement_cost=False))

    captured: dict[str, object] = {"set": False}
    real_associate = tracker_module.associate

    def _spy(*args, **kwargs):
        captured["set"] = True
        captured["obs_verified_reid_identity_ids"] = kwargs.get("obs_verified_reid_identity_ids")
        return real_associate(*args, **kwargs)

    monkeypatch.setattr(tracker_module, "associate", _spy)
    await tracker.step([_obs("d1", amma)], now=NOW, room_polygons=_ROOM)
    await tracker.step([_obs("d2", amma)], now=NOW + timedelta(seconds=1), room_polygons=_ROOM)

    # Flag off: the input stays None, so no per-frame gallery lookup happened.
    assert captured["set"] is True
    assert captured["obs_verified_reid_identity_ids"] is None
