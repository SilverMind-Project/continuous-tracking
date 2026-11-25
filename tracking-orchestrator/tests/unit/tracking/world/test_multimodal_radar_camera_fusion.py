"""Tests for multi-modal camera + device-free position-sensor tracking and fusion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from cts_contracts.wearables import FixKind, MotionState, WearState
from prometheus_client import CollectorRegistry

from app.domain import (
    AnonymousPositionObservation,
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    Identity,
    ObservationSource,
    WearableFix,
    WorldObservation,
)
from app.observability import metrics as metrics_module
from app.observability.metrics import DeviceFreeObservationOutcome
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.device_free import (
    anonymous_track_to_world_observation,
    has_visual_history,
)
from app.tracking.world.tracker import WorldTracker
from app.tracking.world.wearable_config import WearableConfig

_T0 = datetime(2026, 11, 25, 13, 0, 0, tzinfo=UTC)

_ROOMS = {
    "living": [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
    "bathroom": [(6.0, 0.0), (10.0, 0.0), (10.0, 5.0), (6.0, 5.0)],
    "bedroom": [(12.0, 0.0), (16.0, 0.0), (16.0, 5.0), (12.0, 5.0)],
}
_ROOM_NAMES = {"living": "Living Room", "bathroom": "Bathroom", "bedroom": "Bedroom"}
_CAMERA_ROOMS = {
    "cam-living": "Living Room",
    "sensor:radar-living-1": "Living Room",
    "sensor:radar-bathroom-1": "Bathroom",
    "sensor:radar-bedroom-1": "Bedroom",
}
_CAMERA_ROOM_IDS = {
    "cam-living": "living",
    "sensor:radar-living-1": "living",
    "sensor:radar-bathroom-1": "bathroom",
    "sensor:radar-bedroom-1": "bedroom",
}


def _cam_obs(
    at: datetime,
    *,
    det_id: str,
    x_m: float = 2.5,
    y_m: float = 2.5,
    cov_yy: float = 0.81,
    person_id: str | None = None,
) -> WorldObservation:
    face = (
        FaceAnchor(
            person_id=person_id,
            match_score=0.92,
            quality=0.95,
            detection_id=det_id,
            camera_id="cam-living",
            captured_at=at,
            recognition_state="recognized",
            similarity=0.92,
            yaw_deg=5.0,
            calibrated_confidence=0.95,
        )
        if person_id is not None
        else None
    )
    return WorldObservation(
        camera_id="cam-living",
        frame_index=1,
        captured_at=at,
        floor_point=FloorPoint(x_mm=round(x_m * 1000), y_mm=round(y_m * 1000), calibrated=True),
        bbox=BoundingBox(x_min=50.0, y_min=40.0, x_max=150.0, y_max=280.0),
        embedding=[1.0, 0.0, 0.0, 0.0],
        detection_confidence=0.95,
        height_estimate_m=1.70,
        height_sigma_m=0.05,
        face_anchor=face,
        detection_id=det_id,
        quality=0.85,
        floor_residual_m=0.05,
        floor_cov_random=(0.04, 0.0, 0.0, cov_yy),
        primary_score=0.85,
        source=ObservationSource.CAMERA,
    )


def _radar_obs(
    sensor_id: str,
    at: datetime,
    *,
    track_id: str = "t-1",
    x_m: float = 2.5,
    y_m: float = 2.5,
    cov_m2: tuple[float, float, float, float] = (0.04, 0.0, 0.0, 0.04),
    residual_m: float = 0.10,
) -> WorldObservation:
    return anonymous_track_to_world_observation(
        AnonymousPositionObservation(
            sensor_id=sensor_id,
            sensor_track_id=track_id,
            measured_at=at,
            x_m=x_m,
            y_m=y_m,
            cov_m2=cov_m2,
            calibration_residual_m=residual_m,
        )
    )


def _watch_fix(at: datetime, x_m: float, y_m: float, *, sigma_m: float = 0.12) -> WearableFix:
    var = sigma_m * sigma_m
    return WearableFix(
        tag_id="watch-alice",
        member_id="alice",
        measured_at=at,
        kind=FixKind.POSITION,
        x_m=x_m,
        y_m=y_m,
        cov_m2=(var, 0.0, 0.0, var),
        wear_state=WearState.ON_BODY,
        motion_state=MotionState.MOVING,
        physical_device_id="pw5-alice",
    )


async def _make_tracker(
    *,
    wearable_mode: str = "off",
    enable_quality_gate: bool = True,
) -> tuple[WorldTracker, InMemoryPHRepository]:
    gallery = InMemoryGalleryRepository()
    for name in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=name, display_name=name, enrolled_at=_T0)
        )
    wcfg = WearableConfig(mode=wearable_mode)  # type: ignore[arg-type]
    resolver = IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(
            enable_quality_gate=enable_quality_gate,
            min_quality_to_commit=0.35,
            commit_prob=0.65,
            commit_margin=0.15,
        ),
        wearable_config=wcfg,
    )
    ph_repo = InMemoryPHRepository()
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=InMemoryWorldObservationRepository(),
        config=WorldTrackerConfig(
            min_observations_to_publish=1,
            k_cal=1.0,
            primary_switch_frames=5,
        ),
        identity_resolver=resolver,
        gallery_repo=gallery,
        wearable_config=wcfg,
    )
    return tracker, ph_repo


@pytest.mark.asyncio
async def test_camera_ray_error_collapses_under_radar_round_preserving_appearance() -> None:
    tracker, ph_repo = await _make_tracker()

    cam1 = _cam_obs(_T0, det_id="c1", cov_yy=0.81)
    await tracker.step(
        observations=[cam1],
        now=_T0,
        room_polygons=_ROOMS,
        camera_room_map=_CAMERA_ROOMS,
        room_names=_ROOM_NAMES,
        camera_room_ids=_CAMERA_ROOM_IDS,
    )
    cam2 = _cam_obs(_T0 + timedelta(seconds=0.5), det_id="c2", cov_yy=0.81)
    await tracker.step(
        observations=[cam2],
        now=_T0 + timedelta(seconds=0.5),
        room_polygons=_ROOMS,
        camera_room_map=_CAMERA_ROOMS,
        room_names=_ROOM_NAMES,
        camera_room_ids=_CAMERA_ROOM_IDS,
    )
    before = (await ph_repo.list_open())[0]
    before_cov_yy = before.state_cov[5]

    radar1 = _radar_obs(
        "radar-living-1",
        _T0 + timedelta(seconds=1.0),
        x_m=2.5,
        y_m=2.5,
        cov_m2=(0.04, 0.0, 0.0, 0.04),
        residual_m=0.10,
    )
    await tracker.step(
        observations=[radar1],
        now=_T0 + timedelta(seconds=1.0),
        room_polygons=_ROOMS,
        camera_room_map=_CAMERA_ROOMS,
        room_names=_ROOM_NAMES,
        camera_room_ids=_CAMERA_ROOM_IDS,
    )
    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1
    after = open_phs[0]
    assert after.ph_id == before.ph_id
    # Radar total y-variance is 0.05; fused posterior beats both prior and radar total,
    # while remaining >= radar systematic bias floor 0.10^2 = 0.01.
    assert after.state_cov[5] < before_cov_yy
    assert after.state_cov[5] < 0.05
    assert after.state_cov[5] >= 0.01 - 1e-9
    # Appearance, height, quality, and optical primary ownership are untouched.
    assert after.gallery_mean == before.gallery_mean
    assert after.view_prototypes == before.view_prototypes
    assert after.mean_quality == pytest.approx(before.mean_quality)
    assert after.height_estimate_m == pytest.approx(before.height_estimate_m)
    assert tracker._primary_camera[after.ph_id] == "cam-living"


@pytest.mark.asyncio
async def test_hallway_camera_to_bathroom_radar_handoff_and_optical_recovery() -> None:
    tracker, ph_repo = await _make_tracker(wearable_mode="enabled")

    # Alice walks near the living-room exit wearing her watch.
    res_cam = await tracker.step(
        observations=[_cam_obs(_T0, det_id="c1", x_m=4.5, y_m=2.5, person_id="alice")],
        now=_T0,
        room_polygons=_ROOMS,
        camera_room_map=_CAMERA_ROOMS,
        room_names=_ROOM_NAMES,
        face_anchors=[_cam_obs(_T0, det_id="c1", person_id="alice").face_anchor],  # type: ignore[list-item]
        wearable_fixes=[_watch_fix(_T0, 4.5, 2.5)],
        camera_room_ids=_CAMERA_ROOM_IDS,
    )
    assert len(res_cam.updated_phs) == 1
    ph_id = res_cam.updated_phs[0].ph_id
    committed_after_cam = await ph_repo.list_open()
    assert committed_after_cam[0].current_identity_id == "alice"

    # Alice steps into the camera-free bathroom covered only by radar-bathroom-1.
    fixes = [_watch_fix(_T0, 4.5, 2.5)]
    last_res = res_cam
    for i in range(1, 6):
        ts = _T0 + timedelta(seconds=i * 0.5)
        x_m = 6.2 + i * 0.3
        fixes.append(_watch_fix(ts, x_m, 2.5))
        last_res = await tracker.step(
            observations=[_radar_obs("radar-bathroom-1", ts, x_m=x_m, y_m=2.5)],
            now=ts,
            room_polygons=_ROOMS,
            camera_room_map=_CAMERA_ROOMS,
            room_names=_ROOM_NAMES,
            wearable_fixes=fixes,
            camera_room_ids=_CAMERA_ROOM_IDS,
        )

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1
    assert open_phs[0].ph_id == ph_id
    assert open_phs[0].current_identity_id == "alice"
    assert len(last_res.snapshots) == 1
    snap = last_res.snapshots[0]
    assert snap.ph_id == ph_id
    assert snap.room_id == "bathroom"
    assert snap.camera_id == "sensor:radar-bathroom-1"
    assert snap.bbox is None

    # Returning to living-room camera immediately reclaims optical primary ownership.
    t_return = _T0 + timedelta(seconds=3.0)
    res_back = await tracker.step(
        observations=[_cam_obs(t_return, det_id="c-back", x_m=4.8, y_m=2.5, person_id="alice")],
        now=t_return,
        room_polygons=_ROOMS,
        camera_room_map=_CAMERA_ROOMS,
        room_names=_ROOM_NAMES,
        face_anchors=[_cam_obs(t_return, det_id="c-back", person_id="alice").face_anchor],  # type: ignore[list-item]
        camera_room_ids=_CAMERA_ROOM_IDS,
    )
    assert res_back.snapshots[0].camera_id == "cam-living"
    assert res_back.snapshots[0].bbox is not None


@pytest.mark.asyncio
async def test_radar_born_ph_commits_from_wearable_and_camera_room_radar_spawn_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CollectorRegistry()
    test_metrics = metrics_module.build_metrics(registry)
    monkeypatch.setattr(metrics_module, "metrics", test_metrics)

    tracker, ph_repo = await _make_tracker(wearable_mode="enabled", enable_quality_gate=True)

    # Radar in camera-covered living room cannot spawn an unmatched PH.
    await tracker.step(
        observations=[_radar_obs("radar-living-1", _T0, x_m=2.0, y_m=2.5)],
        now=_T0,
        room_polygons=_ROOMS,
        camera_room_map=_CAMERA_ROOMS,
        room_names=_ROOM_NAMES,
        camera_room_ids=_CAMERA_ROOM_IDS,
    )
    assert await ph_repo.list_open() == []
    assert test_metrics.cts_device_free_observations_total.labels(
        outcome=DeviceFreeObservationOutcome.SPAWN_SUPPRESSED
    )._value.get() == pytest.approx(1.0)

    # Radar-born PH in camera-free bedroom has mean_quality=0, no visual history, and commits alice
    # once wearable trajectory evidence meets min_fixes.
    fixes: list[WearableFix] = []
    for i in range(6):
        ts = _T0 + timedelta(seconds=(i + 1) * 0.5)
        x_m = 13.5 + i * 0.2
        fixes.append(_watch_fix(ts, x_m, 2.5))
        await tracker.step(
            observations=[_radar_obs("radar-bedroom-1", ts, track_id="t-bed", x_m=x_m, y_m=2.5)],
            now=ts,
            room_polygons=_ROOMS,
            camera_room_map=_CAMERA_ROOMS,
            room_names=_ROOM_NAMES,
            wearable_fixes=fixes,
            camera_room_ids=_CAMERA_ROOM_IDS,
        )

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1
    bedroom_ph = open_phs[0]
    assert has_visual_history(bedroom_ph) is False
    assert bedroom_ph.mean_quality == pytest.approx(0.0)
    assert bedroom_ph.current_identity_id == "alice"


@pytest.mark.asyncio
async def test_repeated_radar_updates_never_shrink_variance_below_calibration_floor() -> None:
    tracker, ph_repo = await _make_tracker()
    residual_m = 0.20
    floor_m2 = residual_m**2

    for i in range(15):
        ts = _T0 + timedelta(seconds=i * 0.5)
        obs = _radar_obs(
            "radar-bedroom-1",
            ts,
            x_m=14.0,
            y_m=2.5,
            cov_m2=(0.0025, 0.0, 0.0, 0.0025),
            residual_m=residual_m,
        )
        await tracker.step(
            observations=[obs],
            now=ts,
            room_polygons=_ROOMS,
            camera_room_map=_CAMERA_ROOMS,
            room_names=_ROOM_NAMES,
            camera_room_ids=_CAMERA_ROOM_IDS,
        )

    ph = (await ph_repo.list_open())[0]
    pos_cov = np.array(ph.state_cov, dtype=np.float64).reshape(4, 4)[0:2, 0:2]
    min_eig = float(np.linalg.eigvalsh(pos_cov)[0])
    assert min_eig >= floor_m2 - 1e-9
