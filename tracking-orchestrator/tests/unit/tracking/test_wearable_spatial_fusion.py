"""Spatial fusion of trajectory-attested wearable position fixes into PH Kalman state."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from cts_contracts import FixKind, MotionState, WearState
from prometheus_client import CollectorRegistry

from app.domain import (
    BoundingBox,
    FloorPoint,
    PersonHypothesis,
    WearableAttestationGrade,
    WearableFix,
    WorldObservation,
)
from app.observability import metrics as metrics_module
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.kalman import KalmanState
from app.tracking.world.tracker import WorldTracker, WorldTrackerResult
from app.tracking.world.wearable_config import WearableConfig
from app.tracking.world.wearable_spatial_fusion import (
    MotionModel,
    fuse_position_fix,
    select_fusion_fix,
)

T0 = datetime(2026, 11, 25, 9, 0, 0, tzinfo=UTC)
MOTION = MotionModel(process_noise_accel_m_s2=0.5, velocity_decay_s=3.0, damped=False)
ROOMS: dict[str, list[tuple[float, float]]] = {
    "1": [(0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0)],
}


def _fix(
    t: datetime,
    x_m: float,
    y_m: float,
    *,
    sigma_m: float = 0.12,
    tag_id: str = "watch-uwb",
    kind: FixKind = FixKind.POSITION,
) -> WearableFix:
    var = sigma_m * sigma_m
    return WearableFix(
        tag_id=tag_id,
        member_id="amma",
        measured_at=t,
        kind=kind,
        x_m=x_m,
        y_m=y_m,
        cov_m2=(var, 0.0, 0.0, var),
        wear_state=WearState.ON_BODY,
        motion_state=MotionState.MOVING,
        physical_device_id="watch-amma",
    )


def _state(
    t: datetime,
    *,
    x_m: float = 2.0,
    y_m: float = 2.0,
    vx: float = 0.0,
    cov_xx: float = 0.04,
    cov_yy: float = 0.81,
) -> KalmanState:
    cov = np.diag([cov_xx, cov_yy, 0.09, 0.09]).astype(np.float64)
    return KalmanState(mean=np.array([x_m, y_m, vx, 0.0]), covariance=cov, updated_at=t)


class TestSelectFusionFix:
    def test_picks_newest_eligible_fix_of_the_evidence_tag(self) -> None:
        cfg = WearableConfig(mode="shadow", spatial_fusion_mode="shadow")
        state_t = T0 + timedelta(seconds=10)
        fixes = [
            _fix(T0 + timedelta(seconds=8), 2.0, 2.0),
            _fix(T0 + timedelta(seconds=9), 2.0, 2.0),
            _fix(T0 + timedelta(seconds=9.5), 2.0, 2.0, sigma_m=0.5),  # too wide
            _fix(T0 + timedelta(seconds=9.6), 2.0, 2.0, tag_id="watch-ble"),  # other tag
            _fix(T0 + timedelta(seconds=10.5), 2.0, 2.0),  # newer than the state
            _fix(T0 + timedelta(seconds=6), 2.0, 2.0),  # beyond max_dt_s
        ]
        best = select_fusion_fix(
            fixes,
            tag_id="watch-uwb",
            state_time=state_t,
            fused_through=None,
            allow_newer=False,
            cfg=cfg,
        )
        assert best is not None
        assert best.measured_at == T0 + timedelta(seconds=9)

        # A camera-occluded PH may be advanced by the newer fix.
        newer = select_fusion_fix(
            fixes,
            tag_id="watch-uwb",
            state_time=state_t,
            fused_through=None,
            allow_newer=True,
            cfg=cfg,
        )
        assert newer is not None
        assert newer.measured_at == T0 + timedelta(seconds=10.5)

        # A fix is fused once.
        assert (
            select_fusion_fix(
                fixes,
                tag_id="watch-uwb",
                state_time=state_t,
                fused_through=T0 + timedelta(seconds=9),
                allow_newer=False,
                cfg=cfg,
            )
            is None
        )


class TestFusePositionFix:
    def test_tight_fix_collapses_oblique_viewing_ray_but_never_below_floor(self) -> None:
        cfg = WearableConfig(mode="shadow", spatial_fusion_mode="enabled")
        t = T0 + timedelta(seconds=10)
        res = fuse_position_fix(_state(t), _fix(t, 2.05, 2.3), cfg=cfg, motion=MOTION)
        assert not res.gated
        assert res.state is not None
        post = res.state.covariance
        assert post[1, 1] < 0.10
        floor = cfg.spatial_fusion_bias_floor_m**2
        assert float(np.linalg.eigvalsh(post[0:2, 0:2])[0]) >= floor - 1e-12
        assert res.state.updated_at == t

    def test_repeated_fixes_never_shrink_position_below_the_bias_floor(self) -> None:
        cfg = WearableConfig(mode="shadow", spatial_fusion_mode="enabled")
        state = _state(T0, cov_xx=0.5, cov_yy=0.5)
        for k in range(40):
            res = fuse_position_fix(
                state, _fix(T0 - timedelta(milliseconds=k), 2.0, 2.0), cfg=cfg, motion=MOTION
            )
            assert res.state is not None
            state = res.state
        floor = cfg.spatial_fusion_bias_floor_m**2
        assert float(np.linalg.eigvalsh(state.covariance[0:2, 0:2])[0]) == pytest.approx(floor)

    def test_camera_tighter_than_floor_is_not_loosened(self) -> None:
        cfg = WearableConfig(mode="shadow", spatial_fusion_mode="enabled")
        state = _state(T0, cov_xx=0.005, cov_yy=0.005)
        res = fuse_position_fix(state, _fix(T0, 2.0, 2.0), cfg=cfg, motion=MOTION)
        assert res.state is not None
        assert float(np.linalg.eigvalsh(res.state.covariance[0:2, 0:2])[0]) <= 0.005 + 1e-12

    def test_far_fix_is_gated(self) -> None:
        cfg = WearableConfig(mode="shadow", spatial_fusion_mode="enabled")
        res = fuse_position_fix(_state(T0), _fix(T0, 4.0, 2.0), cfg=cfg, motion=MOTION)
        assert res.gated
        assert res.state is None

    def test_lagged_fix_is_moved_to_state_time_along_velocity(self) -> None:
        cfg = WearableConfig(mode="shadow", spatial_fusion_mode="enabled")
        state = _state(T0 + timedelta(seconds=2), x_m=5.0, vx=1.0, cov_xx=0.04, cov_yy=0.04)
        # The person walked 2 m since the fix was taken: shifted, the fix lands on the PH.
        res = fuse_position_fix(state, _fix(T0, 3.0, 2.0), cfg=cfg, motion=MOTION)
        assert not res.gated
        assert res.shift_m < 0.01
        assert res.state is not None
        assert res.state.updated_at == state.updated_at

    def test_newer_fix_predicts_state_to_fix_time(self) -> None:
        cfg = WearableConfig(mode="shadow", spatial_fusion_mode="enabled")
        state = _state(T0, x_m=2.0, vx=1.0, cov_xx=0.04, cov_yy=0.04)
        res = fuse_position_fix(
            state, _fix(T0 + timedelta(seconds=1), 3.0, 2.0), cfg=cfg, motion=MOTION
        )
        assert res.state is not None
        assert res.state.updated_at == T0 + timedelta(seconds=1)
        assert res.state.mean[0] == pytest.approx(3.0, abs=0.05)


def test_spatial_fusion_needs_the_wearable_channel() -> None:
    with pytest.raises(ValueError, match=r"needs wearable\.mode"):
        WearableConfig(mode="off", spatial_fusion_mode="shadow")
    with pytest.raises(ValueError, match=r"spatial_fusion\.mode must"):
        WearableConfig(mode="shadow", spatial_fusion_mode="on")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# WorldTracker replays
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_metrics(monkeypatch: pytest.MonkeyPatch) -> metrics_module.Metrics:
    fresh = metrics_module.build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_module, "metrics", fresh)
    return fresh


def _obs(i: int, t: datetime, x_m: float, y_m: float) -> WorldObservation:
    # A single oblique camera: tight across the viewing ray (x), stretched along it (y).
    return WorldObservation(
        camera_id="cam-1",
        frame_index=i,
        captured_at=t,
        floor_point=FloorPoint(x_mm=round(x_m * 1000), y_mm=round(y_m * 1000), calibrated=True),
        bbox=BoundingBox(x_min=100, y_min=100, x_max=220, y_max=400),
        embedding=[1.0, 0.0, 0.0, 0.0],
        detection_confidence=0.92,
        detection_id=f"det-{i}",
        quality=0.65,
        floor_cov_random=(0.02, 0.0, 0.0, 0.81),
    )


def _tracker(mode: str) -> tuple[WorldTracker, InMemoryPHRepository]:
    wcfg = WearableConfig(
        mode="shadow",
        spatial_fusion_mode=mode,  # type: ignore[arg-type]
    )
    ph_repo = InMemoryPHRepository()
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=InMemoryWorldObservationRepository(),
        config=WorldTrackerConfig(),
        wearable_config=wcfg,
    )
    return tracker, ph_repo


def _path(second: float) -> tuple[float, float]:
    return 1.0 + 0.25 * second, 3.0


async def _walk(
    tracker: WorldTracker,
    *,
    seconds: int,
    camera_until: int | None = None,
) -> list[WorldTrackerResult]:
    """Walk along x at 4 Hz camera rounds with 1 Hz watch fixes arriving 2 s late."""
    results: list[WorldTrackerResult] = []
    fixes: list[WearableFix] = []
    frame = 0
    for quarter in range(seconds * 4 + 1):
        sec = quarter / 4.0
        t = T0 + timedelta(seconds=sec)
        x, y = _path(sec)
        if quarter % 4 == 0 and sec >= 2.0:
            fx, fy = _path(sec - 2.0)
            fixes.append(_fix(t - timedelta(seconds=2.0), fx, fy))
        observations = []
        if camera_until is None or sec <= camera_until:
            frame += 1
            # The camera's along-ray error: a constant 0.6 m bias in y.
            observations = [_obs(frame, t, x, y + 0.6)]
        results.append(
            await tracker.step(
                observations=observations,
                now=t,
                room_polygons=ROOMS,
                camera_room_map={"cam-1": "Living Room"},
                camera_room_ids={"cam-1": "1"},
                wearable_fixes=[f for f in fixes if (t - f.measured_at).total_seconds() <= 30.0],
            )
        )
    return results


async def _only_ph(ph_repo: InMemoryPHRepository) -> PersonHypothesis:
    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1
    return open_phs[0]


class TestWorldTrackerFusion:
    @pytest.mark.asyncio
    async def test_enabled_pulls_ph_onto_wearable_and_history_keeps_raw_camera(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        off, off_repo = _tracker("off")
        enabled, enabled_repo = _tracker("enabled")
        await _walk(off, seconds=16)
        results = await _walk(enabled, seconds=16)

        assert any(
            ev.attestation_grade == WearableAttestationGrade.TRAJECTORY
            for r in results
            for ev in r.wearable_evidence_by_ph.values()
        )
        ph_off = await _only_ph(off_repo)
        ph_on = await _only_ph(enabled_repo)
        _, true_y = _path(16.0)
        # Off follows the biased camera ray; enabled is pulled toward the wearable's y.
        assert abs(ph_on.state_mean[1] - true_y) < abs(ph_off.state_mean[1] - true_y) - 0.2
        assert ph_on.state_cov[5] < ph_off.state_cov[5]

        # Floor history holds the raw camera measurements, never the fused posterior.
        samples = enabled.ph_floor_history.samples_for(ph_on.ph_id)
        assert samples
        assert all(math.isclose(s.y_m, true_y + 0.6, abs_tol=1e-3) for s in samples)
        # The camera's own along-ray variance (plus its calibration floor), not the
        # wearable-tightened posterior.
        assert all(s.cov_m2[3] >= 0.81 for s in samples)

        applied = isolated_metrics.cts_wearable_spatial_fusion_total.labels(
            mode="enabled", outcome="applied"
        )._value.get()
        assert applied > 0

    @pytest.mark.asyncio
    async def test_shadow_measures_without_touching_ph_state(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        off, off_repo = _tracker("off")
        shadow, shadow_repo = _tracker("shadow")
        await _walk(off, seconds=16)
        await _walk(shadow, seconds=16)
        ph_off = await _only_ph(off_repo)
        ph_shadow = await _only_ph(shadow_repo)
        assert ph_shadow.state_mean == ph_off.state_mean
        assert ph_shadow.state_cov == ph_off.state_cov
        assert ph_shadow.last_seen_at == ph_off.last_seen_at
        # Floor history, and so every association and attestation, is unchanged too.
        off_samples = off.ph_floor_history.samples_for(ph_off.ph_id)
        shadow_samples = shadow.ph_floor_history.samples_for(ph_shadow.ph_id)
        assert [(s.captured_at, s.x_m, s.y_m, s.cov_m2) for s in shadow_samples] == [
            (s.captured_at, s.x_m, s.y_m, s.cov_m2) for s in off_samples
        ]
        measured = isolated_metrics.cts_wearable_spatial_fusion_total.labels(
            mode="shadow", outcome="applied"
        )._value.get()
        assert measured > 0

    @pytest.mark.asyncio
    async def test_off_keeps_posterior_history(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        off, off_repo = _tracker("off")
        await _walk(off, seconds=6)
        ph = await _only_ph(off_repo)
        last = off.ph_floor_history.samples_for(ph.ph_id)[-1]
        assert last.y_m == pytest.approx(ph.state_mean[1])

    @pytest.mark.asyncio
    async def test_bridge_keeps_occluded_attested_ph_open_then_closes(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        enabled, enabled_repo = _tracker("enabled")
        off, off_repo = _tracker("off")
        # The camera loses the resident at 16 s; the watch keeps reporting.
        await _walk(enabled, seconds=23, camera_until=16)
        await _walk(off, seconds=23, camera_until=16)

        # Without fusion the PH closes ph_close_grace_s (5 s) after the camera loses it.
        assert await off_repo.list_open() == []
        bridged = await _only_ph(enabled_repo)
        # The watch advanced the occluded PH past its last camera observation.
        assert bridged.last_seen_at > T0 + timedelta(seconds=16)
        assert (
            isolated_metrics.cts_wearable_spatial_fusion_total.labels(
                mode="enabled", outcome="bridged"
            )._value.get()
            > 0
        )

        # Past the bridge window (10 s after the last camera observation) it closes.
        await _walk_tail(enabled, start=24, end=29)
        assert await enabled_repo.list_open() == []


async def _walk_tail(tracker: WorldTracker, *, start: int, end: int) -> None:
    fixes: list[WearableFix] = []
    for sec in range(start, end + 1):
        t = T0 + timedelta(seconds=sec)
        fx, fy = _path(sec - 2.0)
        fixes.append(_fix(t - timedelta(seconds=2.0), fx, fy))
        await tracker.step(
            observations=[],
            now=t,
            room_polygons=ROOMS,
            camera_room_map={"cam-1": "Living Room"},
            camera_room_ids={"cam-1": "1"},
            wearable_fixes=fixes[-10:],
        )
