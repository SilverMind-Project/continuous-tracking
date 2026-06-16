"""Cross-camera revival step-level guardrails.

The pure selector (select_revival_candidate) is unit-tested in
tests/unit/tracking/world/test_revival.py (accept/reject, low-appearance,
implausible transit, face conflict). These tests prove the end-to-end behaviour
through WorldTracker.step with enable_cross_camera_revival:

- Positive: cross_camera_handoff.bin links the camera-B segment to the camera-A
  identity once cross-camera revival is on.
- Clinical guardrail: a stranger appearing on a second camera after a resident
  handoff must NEVER inherit the resident's identity via cross-camera revival.

Marked @pytest.mark.integration; CI selects this marker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.integration._replay import _ROOM_POLYGONS, FIXTURES_DIR, load_fixture

BASE_TIME = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)


async def _replay_with_cross_camera_revival(
    db_pool: Any, fixture_name: str, *, identity_ids: list[str]
) -> Any:
    """Replay a fixture with PH revival and cross-camera revival enabled."""
    from app.domain import Identity
    from app.storage.base import InMemoryGalleryRepository
    from app.storage.postgres.ph_repo import (
        PostgresPHRepository,
        PostgresWorldObservationRepository,
    )
    from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
    from app.tracking.world.config import WorldTrackerConfig
    from app.tracking.world.tracker import WorldTracker

    gallery = InMemoryGalleryRepository()
    for iid in identity_ids:
        await gallery.upsert_identity(
            Identity(identity_id=iid, display_name=iid, enrolled_at=datetime.now(UTC))
        )
    resolver = IdentityResolver(
        gallery_repo=gallery, config=ResolverConfig(enable_sticky_maintenance=True)
    )
    ph_repo = PostgresPHRepository(db_pool)
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=PostgresWorldObservationRepository(db_pool),
        config=WorldTrackerConfig(
            enable_ph_revival=True,
            enable_cross_camera_revival=True,
            enable_uncalibrated_gate_relax=True,
        ),
        identity_resolver=resolver,
    )

    steps = load_fixture(FIXTURES_DIR / fixture_name)
    for i, frame_obs in enumerate(steps):
        face_anchors = [o.face_anchor for o in frame_obs if o.face_anchor is not None] or None
        await tracker.step(
            observations=frame_obs,
            now=BASE_TIME + timedelta(seconds=i * 0.5),
            room_polygons=_ROOM_POLYGONS,
            face_anchors=face_anchors,
        )
    return ph_repo


@pytest.mark.integration
class TestCrossCameraHandoff:
    """cross_camera_handoff.bin links the two cameras to one identity."""

    @pytest.mark.asyncio
    async def test_handoff_preserves_identity_across_cameras(self, db_pool: Any) -> None:
        from app.storage.postgres.ph_repo import PostgresPHRepository

        await _replay_with_cross_camera_revival(
            db_pool, "cross_camera_handoff.bin", identity_ids=["alice"]
        )
        ph_repo = PostgresPHRepository(db_pool)
        phs, _total = await ph_repo.list_active(include_transient=True)

        # The camera-B segment must carry alice's identity after the handoff is linked.
        b_phs = [ph for ph in phs if "cam-handoff-b" in (ph.active_cameras or set())]
        assert b_phs, "expected at least one PH active on camera B"
        assert any(ph.current_identity_id == "alice" for ph in b_phs), (
            "camera-B segment did not inherit alice's identity via cross-camera revival"
        )


@pytest.mark.integration
class TestCrossCameraStrangerGuardrail:
    """A stranger on a second camera must not inherit the resident's identity."""

    @pytest.mark.asyncio
    async def test_stranger_does_not_inherit_identity_cross_camera(self, db_pool: Any) -> None:
        from app.domain import (
            BoundingBox,
            FaceAnchor,
            FloorPoint,
            Identity,
            WorldObservation,
        )
        from app.storage.base import InMemoryGalleryRepository
        from app.storage.postgres.ph_repo import (
            PostgresPHRepository,
            PostgresWorldObservationRepository,
        )
        from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
        from app.tracking.orientation import OrientationBin
        from app.tracking.world.config import WorldTrackerConfig
        from app.tracking.world.tracker import WorldTracker

        grandma_body = tuple(1.0 if i < 384 else 0.0 for i in range(768))
        stranger_body = tuple(0.0 if i < 384 else 1.0 for i in range(768))

        gallery = InMemoryGalleryRepository()
        await gallery.upsert_identity(
            Identity(identity_id="grandma", display_name="grandma", enrolled_at=datetime.now(UTC))
        )
        resolver = IdentityResolver(
            gallery_repo=gallery, config=ResolverConfig(enable_sticky_maintenance=True)
        )
        ph_repo = PostgresPHRepository(db_pool)
        tracker = WorldTracker(
            ph_repo=ph_repo,
            obs_repo=PostgresWorldObservationRepository(db_pool),
            config=WorldTrackerConfig(
                enable_ph_revival=True,
                enable_cross_camera_revival=True,
            ),
            identity_resolver=resolver,
        )

        def _obs(
            cam: str, fidx: int, t: float, body: tuple[float, ...], face: bool
        ) -> WorldObservation:
            fa = (
                FaceAnchor(
                    person_id="grandma",
                    confidence=0.95,
                    quality=0.9,
                    detection_id=f"{cam}-{fidx}",
                    camera_id=cam,
                    captured_at=BASE_TIME + timedelta(seconds=t),
                    recognition_state="recognized",
                    similarity=0.95,
                    yaw_deg=5.0,
                )
                if face
                else None
            )
            return WorldObservation(
                camera_id=cam,
                frame_index=fidx,
                captured_at=BASE_TIME + timedelta(seconds=t),
                floor_point=FloorPoint(x_mm=1000.0, y_mm=1000.0, calibrated=False),
                bbox=BoundingBox(x_min=0.4, y_min=0.3, x_max=0.5, y_max=0.7),
                embedding=body,
                detection_confidence=0.9,
                height_estimate_m=1.65,
                face_anchor=fa,
                detection_id=f"{cam}-{fidx}",
                quality=0.8,
                orientation=OrientationBin.FRONT,
                orientation_confidence=0.9,
            )

        # Grandma on camera A (face-recognized), builds prototypes + identity.
        for k in range(6):
            obs = _obs("cam-a", k, 0.5 * k, grandma_body, face=True)
            await tracker.step(
                observations=[obs], now=obs.captured_at, face_anchors=[obs.face_anchor]
            )
        # Gap: empty frames push past ph_close_grace_s so the camera-A PH closes.
        for k in range(6, 40):
            await tracker.step(observations=[], now=BASE_TIME + timedelta(seconds=0.5 * k))

        # Stranger appears on camera B with an orthogonal body and no face.
        stranger_committed_grandma = False
        for k in range(40, 50):
            obs = _obs("cam-b", k, 0.5 * k, stranger_body, face=False)
            result = await tracker.step(observations=[obs], now=obs.captured_at)
            if any(d.identity_id == "grandma" for d in result.identity_decisions):
                stranger_committed_grandma = True

        assert not stranger_committed_grandma, "stranger must not commit grandma's identity"
        b_phs = [
            ph
            for ph in (await ph_repo.list_active(include_transient=True))[0]
            if "cam-b" in (ph.active_cameras or set())
        ]
        assert b_phs, "stranger PH should exist on camera B"
        assert all(ph.current_identity_id != "grandma" for ph in b_phs), (
            "stranger PH on camera B must not carry grandma's identity"
        )
