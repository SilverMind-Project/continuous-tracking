"""Two-person synthetic replay proving height separates identically dressed residents."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.calibration.state import CalibrationState
from app.domain import (
    BoundingBox,
    CameraProfile,
    FloorPoint,
    GalleryEmbedding,
    Identity,
    IdentityHeightProfile,
    OrientationBin,
    WorldObservation,
)
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)
from app.storage.height_profiles import InMemoryIdentityHeightProfileRepository
from app.tracking.camera_pose import (
    pose_from_homography,
    standing_height_with_sigma,
)
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker
from tests.unit.tracking.test_camera_pose import _synthetic_camera
from tests.unit.tracking.test_standing_height import _project_point


async def _run_replay(*, height_mode: str) -> tuple[dict[str, list[str | None]], int]:
    """Replay 15 frames of Alice (1.52 m) and Bob (1.80 m) in identical clothing."""
    width_px, height_px = 1920, 1080
    h_i2f, rot, centre, f_px, principal = _synthetic_camera(
        centre_m=(0.0, 0.0, 2.5),
        tilt_deg=-30.0,
        yaw_deg=0.0,
        fov_deg=75.0,
        width_px=width_px,
        height_px=height_px,
    )
    pose = pose_from_homography(h_i2f, f_px, principal)

    cal = CalibrationState()
    await cal.set_homography("cam-hall", h_i2f.tolist())
    _ = CameraProfile(
        camera_id="cam-hall",
        horizontal_fov_deg=75.0,
        mounting_height_m=2.5,
        tilt_deg=-30.0,
    )

    t0 = datetime(2026, 11, 21, 10, 0, 0, tzinfo=UTC)
    shared_clothes = [0.6, 0.8, 0.0, 0.0]

    gallery = InMemoryGalleryRepository()
    identities = [
        Identity(identity_id="alice", display_name="Alice", enrolled_at=t0),
        Identity(identity_id="bob", display_name="Bob", enrolled_at=t0),
    ]
    for ident in identities:
        await gallery.upsert_identity(ident)
    for pid in ("alice", "bob"):
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"emb-{pid}",
                identity_id=pid,
                embedding=shared_clothes,
                quality=0.95,
                seen_at=t0 - timedelta(hours=1),
                face_confirmed=True,
                orientation=OrientationBin.FRONT,
                state="operator_verified",
            )
        )

    profiles = InMemoryIdentityHeightProfileRepository(
        [
            IdentityHeightProfile(
                identity_id="alice",
                height_m=1.52,
                height_sigma_m=0.03,
                source="learned",
                samples=300,
                cameras=2,
                updated_at=t0 - timedelta(hours=1),
            ),
            IdentityHeightProfile(
                identity_id="bob",
                height_m=1.80,
                height_sigma_m=0.03,
                source="learned",
                samples=300,
                cameras=2,
                updated_at=t0 - timedelta(hours=1),
            ),
        ]
    )

    resolver = IdentityResolver(
        gallery_repo=gallery,
        identities=[
            Identity(identity_id="alice", display_name="Alice", enrolled_at=t0),
            Identity(identity_id="bob", display_name="Bob", enrolled_at=t0),
        ],
        config=ResolverConfig(
            height_mode=height_mode,  # type: ignore[arg-type]
            enable_sticky_maintenance=False,
            enable_multiview_gallery=True,
        ),
        height_profile_repo=profiles,
    )

    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=obs_repo,
        config=WorldTrackerConfig(),
        identity_resolver=resolver,
    )

    history: dict[str, list[str | None]] = {"alice": [], "bob": []}
    swaps = 0

    for frame_idx in range(15):
        now = t0 + timedelta(milliseconds=500 * frame_idx)
        # Alice walks at y=-1.0 m, Bob walks at y=+1.0 m (well separated on floor).
        x_m = 2.5 + 0.15 * frame_idx
        actors = [
            ("alice", (x_m, -1.0), 1.52),
            ("bob", (x_m, +1.0), 1.80),
        ]
        obs_list: list[WorldObservation] = []
        for actor_name, foot_xy, true_h in actors:
            fu, fv = _project_point((foot_xy[0], foot_xy[1], 0.0), rot, centre, f_px, principal)
            hu, hv = _project_point((foot_xy[0], foot_xy[1], true_h), rot, centre, f_px, principal)
            est = standing_height_with_sigma(pose, foot_xy, None, (hu, hv))
            assert est is not None
            h_est, h_sigma = est
            obs_list.append(
                WorldObservation(
                    camera_id="cam-hall",
                    frame_index=frame_idx,
                    captured_at=now,
                    floor_point=FloorPoint(
                        x_mm=round(foot_xy[0] * 1000),
                        y_mm=round(foot_xy[1] * 1000),
                        calibrated=True,
                    ),
                    bbox=BoundingBox(
                        x_min=int(fu - 50),
                        y_min=int(hv),
                        x_max=int(fu + 50),
                        y_max=int(fv),
                    ),
                    embedding=shared_clothes,
                    detection_confidence=0.95,
                    detection_id=f"{actor_name}-{frame_idx}",
                    quality=0.9,
                    height_estimate_m=h_est,
                    height_sigma_m=h_sigma,
                    orientation=OrientationBin.FRONT,
                    orientation_confidence=0.95,
                    floor_residual_m=0.02,
                    footpoint_reliable=True,
                )
            )

        await tracker.step(observations=obs_list, now=now, low_band_observations=[])
        open_phs = await ph_repo.list_open()
        assert len(open_phs) == 2
        for ph in open_phs:
            true_actor = "alice" if ph.state_mean[1] < 0.0 else "bob"
            assigned = ph.current_identity_id
            history[true_actor].append(assigned)
            if assigned is not None and assigned != true_actor:
                swaps += 1

    return history, swaps


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_person_replay_height_separates_identically_dressed_residents() -> None:
    hist_off, swaps_off = await _run_replay(height_mode="off")
    hist_on, swaps_on = await _run_replay(height_mode="enabled")

    assert swaps_off == 0
    assert swaps_on == 0

    # With height off, identical clothing leaves both tracks unresolved throughout.
    assert all(x is None for x in hist_off["alice"])
    assert all(x is None for x in hist_off["bob"])

    # With height enabled, both Alice and Bob resolve accurately with zero swaps.
    first_alice = next((i for i, x in enumerate(hist_on["alice"]) if x == "alice"), None)
    first_bob = next((i for i, x in enumerate(hist_on["bob"]) if x == "bob"), None)
    assert first_alice is not None and first_alice <= 2
    assert first_bob is not None and first_bob <= 2
    assert hist_on["alice"][-1] == "alice"
    assert hist_on["bob"][-1] == "bob"
