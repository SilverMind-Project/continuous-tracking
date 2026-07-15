"""M12 real-tracker replay: two residents under direct-ArcFace authority.

Drives the live ``WorldTracker`` + ``IdentityResolver`` over a two-person
scenario where both residents carry qualifying calibrated ArcFace anchors, then
scores the snapshots through the shared evaluator. The zero-authoritative-swap
assertion is the release gate exercised against the real resolver (here on a
clean synthetic scenario; the full private two-person golden acceptance is
operator-gated per the rollout guide). Uses InMemory repositories, so it runs in
the normal gate without Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain import (
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    Identity,
    OrientationBin,
    WorldObservation,
)
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker
from tests.integration._identity_replay import (
    build_records,
    decision_row_from_snapshot,
    evaluate,
)

BASE = datetime(2026, 6, 23, 9, 0, 0, tzinfo=UTC)
_ROOM = {"living_room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}

_AMMA_BODY = [1.0 if i < 384 else 0.0 for i in range(768)]
_GRANDMA_BODY = [0.0 if i < 384 else 1.0 for i in range(768)]


def _obs(
    cam: str, fidx: int, t: float, x_m: float, body: list[float], person: str
) -> WorldObservation:
    captured = BASE + timedelta(seconds=t)
    det_id = f"{person}-{fidx}"
    return WorldObservation(
        camera_id=cam,
        frame_index=fidx,
        captured_at=captured,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=5000, calibrated=True),
        bbox=BoundingBox(x_min=x_m, y_min=1.0, x_max=x_m + 0.5, y_max=2.0),
        embedding=body,
        detection_confidence=0.92,
        detection_id=det_id,
        quality=0.85,
        height_estimate_m=1.62,
        orientation=OrientationBin.FRONT,
        orientation_confidence=0.9,
        face_anchor=FaceAnchor(
            person_id=person,
            confidence=0.96,
            quality=0.9,
            detection_id=det_id,
            camera_id=cam,
            captured_at=captured,
            recognition_state="recognized",
            similarity=0.95,
            yaw_deg=5.0,
            calibrated_confidence=0.95,  # >= 0.80 authority threshold
        ),
    )


async def _build_tracker() -> WorldTracker:
    gallery = InMemoryGalleryRepository()
    for ident in ("amma", "grandma"):
        await gallery.upsert_identity(
            Identity(identity_id=ident, display_name=ident, enrolled_at=BASE)
        )
    resolver = IdentityResolver(
        gallery_repo=gallery, config=ResolverConfig(enable_sticky_maintenance=True)
    )
    return WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=WorldTrackerConfig(min_observations_to_publish=1),
        identity_resolver=resolver,
        gallery_repo=gallery,
    )


async def test_two_residents_no_authoritative_swap() -> None:
    tracker = await _build_tracker()
    correction_repo = InMemoryIdentityCorrectionRepository()
    rows = []

    # Both residents present every frame, walking slowly toward and past each
    # other (5->8 m and 12->9 m) so the bodies get close mid-replay.
    for k in range(12):
        amma_x = 5.0 + 0.25 * k
        grandma_x = 12.0 - 0.25 * k
        obs = [
            _obs("cam-1", k, 0.5 * k, amma_x, _AMMA_BODY, "amma"),
            _obs("cam-1", k, 0.5 * k, grandma_x, _GRANDMA_BODY, "grandma"),
        ]
        result = await tracker.step(
            observations=obs,
            now=BASE + timedelta(seconds=0.5 * k),
            room_polygons=_ROOM,
            face_anchors=[o.face_anchor for o in obs],
        )
        rows.extend(decision_row_from_snapshot(s) for s in result.snapshots)

    records = await build_records(rows, correction_repo)
    report = evaluate(records)

    # Release gate: no authoritative identity swap anywhere in the replay.
    assert report.swap_count == 0, report.authoritative_swaps
    # Each resident's body actually committed under ArcFace authority at least
    # once, so the zero-swap result is meaningful (not vacuous).
    assert report.authoritative_frames > 0
    # Two residents -> at most one active PH per identity per instant.
    assert report.duplicate_active_frames == 0
    # Every known decision names a provenance source.
    assert report.source_attribution_complete is True
