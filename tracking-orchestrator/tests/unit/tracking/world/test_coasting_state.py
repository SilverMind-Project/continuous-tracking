"""Unit tests for world tracker coasting state, capture-time clock, and time-skew dedup."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    BoundingBox,
    Detection,
    FloorPoint,
    PersonHypothesis,
    TransitZone,
    WorldObservation,
)
from app.observability import metrics as metrics_module
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.world_tracking import WorldTrackingStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.services.transit_zone_map import TransitZoneBinding, TransitZoneMap
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.dedup import dedup_observations
from app.tracking.world.kalman import KalmanState, predict
from app.tracking.world.revival import select_revival_candidate
from app.tracking.world.tracker import WorldTracker, _advance_unmatched_ph
from app.tracking.world.transit_detector import TransitDetector
from app.transport.redis_streams import FrameReady

_T0 = datetime(2026, 11, 13, 12, 0, 0, tzinfo=UTC)
_EMB = [1.0, 0.0, 0.0, 0.0]


def _make_obs(
    *,
    camera_id: str = "cam-1",
    x_m: float = 0.0,
    y_m: float = 0.0,
    captured_at: datetime = _T0,
    detection_id: str = "det-1",
    frame_index: int = 1,
    embedding: list[float] | None = None,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=captured_at,
        floor_point=FloorPoint(x_mm=round(x_m * 1000.0), y_mm=round(y_m * 1000.0), calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=10, x_max=110, y_max=210),
        embedding=list(_EMB if embedding is None else embedding),
        detection_confidence=0.95,
        detection_id=detection_id,
        quality=0.85,
    )


def _ks_of(ph: PersonHypothesis) -> KalmanState:
    return KalmanState(
        mean=np.array(ph.state_mean, dtype=np.float64),
        covariance=np.array(ph.state_cov, dtype=np.float64).reshape(4, 4),
        updated_at=ph.last_seen_at,
    )


@pytest.mark.asyncio
async def test_coasting_round_rate_independence_and_structural_invariant() -> None:
    """Stored PH state and in-round prediction are invariant to empty-step cadence.

    Also verifies the structural invariant: after any step, for every unobserved
    open PH, its stored state equals its state before the step.
    """
    cfg = WorldTrackerConfig(
        ph_close_grace_s=15.0,
        min_observations_to_publish=2,
        zupt_consecutive_frames=99,
    )
    stored_states: list[tuple[tuple[float, ...], tuple[float, ...], datetime]] = []
    predicted_at_end: list[tuple[np.ndarray, np.ndarray]] = []

    for n_steps in (2, 10, 40):
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()
        tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

        # Spawn at x = 0 at t = 0, then move to x = 0.5 at t = 0.5 s.
        await tracker.step([_make_obs(x_m=0.0, captured_at=_T0, detection_id="d0")], now=_T0)
        t_obs = _T0 + timedelta(seconds=0.5)
        await tracker.step(
            [_make_obs(x_m=0.5, captured_at=t_obs, detection_id="d1", frame_index=2)],
            now=t_obs,
        )
        open_before = await ph_repo.list_open()
        assert len(open_before) == 1
        baseline_ph = open_before[0]
        assert baseline_ph.state_mean[2] > 0.0

        # Run n_steps empty rounds over the next 2.0 s (up to t = 2.5 s).
        dt_step = 2.0 / n_steps
        last_result = None
        for step_idx in range(1, n_steps + 1):
            t_now = t_obs + timedelta(seconds=step_idx * dt_step)
            before_step = {p.ph_id: p for p in await ph_repo.list_open()}
            last_result = await tracker.step([], now=t_now)
            after_step = {p.ph_id: p for p in await ph_repo.list_open()}

            # Structural check: every unobserved open PH keeps its exact pre-step state.
            for pid, ph_after in after_step.items():
                ph_prev = before_step[pid]
                assert ph_after.state_mean == ph_prev.state_mean
                assert ph_after.state_cov == ph_prev.state_cov
                assert ph_after.last_seen_at == ph_prev.last_seen_at

        open_after = await ph_repo.list_open()
        assert len(open_after) == 1
        ph_final = open_after[0]
        stored_states.append((ph_final.state_mean, ph_final.state_cov, ph_final.last_seen_at))

        t_end = t_obs + timedelta(seconds=2.0)
        pred_ks = predict(
            _ks_of(ph_final),
            t_end,
            cfg.process_noise_accel_m_s2,
            cfg.velocity_decay_s,
        )
        predicted_at_end.append((pred_ks.mean, pred_ks.covariance))

        # Published snapshot for the unobserved PH carries coasting=True and predicted state.
        assert last_result is not None
        assert ph_final.ph_id in last_result.coasting_ph_ids
        assert len(last_result.snapshots) == 1
        snap = last_result.snapshots[0]
        assert snap.coasting is True
        assert abs(snap.floor_x_m - float(pred_ks.mean[0])) < 1e-9

    base_mean, base_cov, base_seen = stored_states[0]
    for mean, cov, seen in stored_states[1:]:
        assert seen == base_seen
        assert np.allclose(mean, base_mean, atol=1e-9)
        assert np.allclose(cov, base_cov, atol=1e-9)

    base_pred_mean, base_pred_cov = predicted_at_end[0]
    for pred_mean, pred_cov in predicted_at_end[1:]:
        assert np.allclose(pred_mean, base_pred_mean, atol=1e-9)
        assert np.allclose(pred_cov, base_pred_cov, atol=1e-9)


def test_advance_unmatched_ph_matches_single_predict_for_all_round_intervals() -> None:
    """Advancing an unmatched PH across any round interval matches single-step predict."""
    for step_s in (0.2, 0.5, 1.0):
        ph = PersonHypothesis(
            ph_id="p",
            state_mean=(0.0, 0.0, 1.0, 0.0),
            state_cov=(0.09, 0, 0, 0, 0, 0.09, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.25),
            born_at=_T0,
            last_seen_at=_T0,
            last_seen_camera="c1",
            observation_count=5,
        )
        t = _T0
        n_rounds = round(15.0 / step_s)
        for i in range(1, n_rounds + 1):
            t = _T0 + timedelta(seconds=i * step_s)
            pred = predict(_ks_of(ph), t, process_noise_accel_m_s2=0.5, velocity_decay_s=3.0)
            ph = _advance_unmatched_ph(ph, pred, t, ph_close_grace_s=15.0)
            elapsed = (t - _T0).total_seconds()
            if abs(elapsed - round(elapsed)) < 1e-6 and round(elapsed) in (2, 5, 10, 15):
                single = predict(
                    KalmanState(
                        mean=np.array([0.0, 0.0, 1.0, 0.0]),
                        covariance=np.diag([0.09, 0.09, 0.25, 0.25]),
                        updated_at=_T0,
                    ),
                    t,
                    0.5,
                    3.0,
                )
                assert np.allclose(pred.mean, single.mean, atol=1e-9)
                assert np.allclose(pred.covariance, single.covariance, atol=1e-9)
                assert ph.state_mean == (0.0, 0.0, 1.0, 0.0)


@pytest.mark.asyncio
async def test_geometric_gate_after_coasting_rejects_6m_jump_for_all_round_rates() -> None:
    """After 2 s unobserved, the geometric gate matches a 2 s prediction and rejects a 6 m jump."""
    cfg = WorldTrackerConfig(
        ph_close_grace_s=15.0,
        enable_ph_revival=False,
        zupt_consecutive_frames=99,
    )
    single_2s = predict(
        KalmanState(
            mean=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            covariance=np.diag([0.09, 0.09, 0.25, 0.25]).astype(np.float64),
            updated_at=_T0,
        ),
        _T0 + timedelta(seconds=2.0),
        cfg.process_noise_accel_m_s2,
        cfg.velocity_decay_s,
    )
    for n_steps in (2, 10, 40):
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()
        tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

        orig_ph = PersonHypothesis(
            ph_id="ph-steady",
            state_mean=(0.0, 0.0, 1.0, 0.0),
            state_cov=(0.09, 0, 0, 0, 0, 0.09, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.25),
            born_at=_T0,
            last_seen_at=_T0,
            last_seen_camera="cam-1",
            observation_count=5,
            gallery_mean=list(_EMB),
        )
        await ph_repo.save(orig_ph)

        dt_step = 2.0 / n_steps
        for step_idx in range(1, n_steps):
            t_now = _T0 + timedelta(seconds=step_idx * dt_step)
            await tracker.step([], now=t_now)

        # At _T0 + 2.0 s, the in-round prediction matches single_2s (sigma_x = 1.20 m),
        # so an observation 6.0 m away from the last position (6.0 m vs predicted 1.03 m)
        # exceeds gate_chi2 and is rejected for every N.
        t_final = _T0 + timedelta(seconds=2.0)
        stored_before_final = (await ph_repo.list_open())[0]
        pred_at_final = predict(
            _ks_of(stored_before_final),
            t_final,
            cfg.process_noise_accel_m_s2,
            cfg.velocity_decay_s,
        )
        assert np.allclose(pred_at_final.mean, single_2s.mean, atol=1e-9)
        assert np.allclose(pred_at_final.covariance, single_2s.covariance, atol=1e-9)

        far_obs = _make_obs(x_m=6.0, captured_at=t_final, detection_id="d-far", frame_index=99)
        result = await tracker.step([far_obs], now=t_final)

        # The 6 m jump must not associate with orig_ph; it spawns a separate PH.
        assert result.det_to_ph["d-far"] != orig_ph.ph_id
        assert orig_ph.ph_id in result.coasting_ph_ids


@pytest.mark.asyncio
async def test_closed_ph_revival_measures_distance_from_last_measured_position() -> None:
    """When an unobserved PH closes, revival distance uses its last measured position."""
    cfg = WorldTrackerConfig(
        ph_close_grace_s=5.0,
        enable_ph_revival=True,
        revive_max_age_s=30.0,
        revive_max_distance_m=1.5,
        revive_appearance_min_sim=0.55,
        zupt_consecutive_frames=99,
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    # Observe moving from x = 0.0 to x = 1.0 over 0.5 s so velocity is positive (~1.8 m/s).
    await tracker.step([_make_obs(x_m=0.0, captured_at=_T0, detection_id="d0")], now=_T0)
    t1 = _T0 + timedelta(seconds=0.5)
    await tracker.step(
        [_make_obs(x_m=1.0, captured_at=t1, detection_id="d1", frame_index=2)],
        now=t1,
    )
    open_ph = (await ph_repo.list_open())[0]
    measured_x = open_ph.state_mean[0]
    assert abs(measured_x - 1.0) < 0.2

    # Step 25 empty rounds at 0.25 s intervals (6.25 s > ph_close_grace_s = 5.0 s).
    for i in range(1, 26):
        await tracker.step([], now=t1 + timedelta(seconds=0.25 * i))

    closed_phs = await ph_repo.list_closed_since(_T0, limit=10)
    assert len(closed_phs) == 1
    closed_ph = closed_phs[0]
    assert closed_ph.state_mean[0] == measured_x

    # Observation near the last measured position (x = 1.2 m) revives the closed PH.
    t_revive = t1 + timedelta(seconds=7.0)
    near_measured = _make_obs(x_m=1.2, captured_at=t_revive, detection_id="d-near")
    assert select_revival_candidate(near_measured, [closed_ph], t_revive, cfg) is not None

    # Observation at a drifted position (x = 4.0 m, > 1.5 m from measured_x) is rejected.
    far_drifted = _make_obs(x_m=4.0, captured_at=t_revive, detection_id="d-drifted")
    assert select_revival_candidate(far_drifted, [closed_ph], t_revive, cfg) is None


@pytest.mark.asyncio
async def test_transit_detector_skips_unobserved_coasting_ph() -> None:
    """WorldTrackingStage does not emit a room transition for an unobserved coasting PH."""
    cfg = WorldTrackerConfig(ph_close_grace_s=15.0, zupt_consecutive_frames=99)
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    zone = TransitZone(
        zone_id="z-door",
        name="Hall Door",
        kind="door",
        polygon=[(1.15, -1.0), (1.25, -1.0), (1.25, 1.0), (1.15, 1.0)],
        inside_room_id="room-a",
        outside_room_id="room-b",
        direction_vec=(1.0, 0.0),
    )
    zone_map = TransitZoneMap()
    await zone_map.set_all([TransitZoneBinding(zone=zone, bound_at=_T0)])
    transit_detector = TransitDetector()

    published_events: list[object] = []

    class _RecorderPublisher:
        async def publish(self, event: object, identity_id: str | None = None) -> None:
            published_events.append((event, identity_id))

    live_config = LiveConfigHolder(
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
        transit_zone_map=zone_map,
        transit_detector=transit_detector,
        room_transition_publisher=_RecorderPublisher(),  # type: ignore[arg-type]
    )
    stage = WorldTrackingStage(tracker=tracker, live_config=live_config, config=cfg)

    def _ctx(t: datetime, x_m: float | None, idx: int) -> FrameContext:
        frame = FrameReady(
            camera_id="cam-1",
            frame_index=idx,
            capture_time_unix_ns=int(t.timestamp() * 1_000_000_000),
            received_time_unix_ns=int(t.timestamp() * 1_000_000_000),
            minio_key=f"cam-1/{idx}.jpg",
            width=1280,
            height=720,
        )
        ctx = FrameContext(frame=frame, event_time=t, capture_time=t)
        if x_m is not None:
            ctx.domain_detections = [
                Detection(
                    detection_id=f"det-{idx}",
                    camera_id="cam-1",
                    capture_time=t,
                    event_time=t,
                    bbox=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=300),
                    confidence=0.95,
                    floor_point=FloorPoint(x_mm=round(x_m * 1000.0), y_mm=0, calibrated=True),
                    embedding=list(_EMB),
                    crop_quality=0.85,
                )
            ]
        return ctx

    # Observe at x = 0.0 (t = 0) and x = 1.0 (t = 0.5 s), approaching the line at x = 1.2 m.
    await stage.run(_ctx(_T0, 0.0, 1))
    await stage.run(_ctx(_T0 + timedelta(seconds=0.5), 1.0, 2))
    assert published_events == []

    # Coast with no observation at t = 1.5 s (prediction crosses x = 1.2 m, stored does not).
    await stage.run(_ctx(_T0 + timedelta(seconds=1.5), None, 3))
    assert published_events == []

    # Actual observation at x = 1.6 m (t = 2.0 s) crosses the line and emits the transition.
    await stage.run(_ctx(_T0 + timedelta(seconds=2.0), 1.6, 4))
    assert len(published_events) == 1


@pytest.mark.asyncio
async def test_out_of_order_and_equal_time_observations_do_not_rewind_last_seen_at() -> None:
    """Older and equal-time observations update position at dt=0 and keep last_seen_at monotonic."""
    reg = CollectorRegistry()
    metrics_module.metrics = metrics_module.build_metrics(reg)

    cfg = WorldTrackerConfig(dedup_enabled=False, zupt_consecutive_frames=99)
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    t1 = _T0 + timedelta(seconds=1.0)
    await tracker.step(
        [_make_obs(camera_id="cam-a", x_m=1.0, captured_at=t1, detection_id="d1")], now=t1
    )
    ph_at_t1 = (await ph_repo.list_open())[0]
    assert ph_at_t1.last_seen_at == t1

    # Equal-time observation (captured_at == last_seen_at): updates at dt=0, no metric inc.
    await tracker.step(
        [_make_obs(camera_id="cam-b", x_m=1.2, captured_at=t1, detection_id="d2", frame_index=2)],
        now=t1,
    )
    ph_after_eq = (await ph_repo.list_open())[0]
    assert ph_after_eq.last_seen_at == t1
    assert ph_after_eq.state_mean[0] > 1.0
    eq_count = reg.get_sample_value(
        "cts_worldtracker_out_of_order_observations_total", {"camera_id": "cam-b"}
    )
    assert (eq_count or 0.0) == 0.0

    # Out-of-order observation (captured_at < last_seen_at): updates position with dt=0, keeps
    # last_seen_at at t1, and increments cts_worldtracker_out_of_order_observations_total.
    t_older = _T0 + timedelta(seconds=0.6)
    await tracker.step(
        [
            _make_obs(
                camera_id="cam-b",
                x_m=1.4,
                captured_at=t_older,
                detection_id="d3",
                frame_index=3,
            )
        ],
        now=t1,
    )
    ph_after_old = (await ph_repo.list_open())[0]
    assert ph_after_old.last_seen_at == t1
    assert ph_after_old.state_mean[0] > ph_after_eq.state_mean[0]
    oo_count = reg.get_sample_value(
        "cts_worldtracker_out_of_order_observations_total", {"camera_id": "cam-b"}
    )
    assert oo_count == 1.0


@pytest.mark.asyncio
async def test_stage_multi_camera_batch_uses_capture_time() -> None:
    """Two cameras captured 400 ms apart in one batch update at their own capture_time."""
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        dedup_max_distance_m=0.9,
        dedup_time_skew_speed_m_s=1.4,
        dedup_max_distance_ceiling_m=1.5,
        zupt_consecutive_frames=99,
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)
    live_config = LiveConfigHolder(
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
    )
    stage = WorldTrackingStage(tracker=tracker, live_config=live_config, config=cfg)

    def _make_ctx(
        cam: str, cap_t: datetime, recv_t: datetime, x_m: float, idx: int
    ) -> FrameContext:
        frame = FrameReady(
            camera_id=cam,
            frame_index=idx,
            capture_time_unix_ns=int(cap_t.timestamp() * 1_000_000_000),
            received_time_unix_ns=int(recv_t.timestamp() * 1_000_000_000),
            minio_key=f"{cam}/{idx}.jpg",
            width=1280,
            height=720,
        )
        ctx = FrameContext(frame=frame, event_time=recv_t, capture_time=cap_t)
        ctx.domain_detections = [
            Detection(
                detection_id=f"{cam}-det-{idx}",
                camera_id=cam,
                capture_time=cap_t,
                event_time=recv_t,
                bbox=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=300),
                confidence=0.95,
                floor_point=FloorPoint(x_mm=round(x_m * 1000.0), y_mm=0, calibrated=True),
                embedding=list(_EMB),
                crop_quality=0.85,
            )
        ]
        return ctx

    # Seed the PH at t0 on cam-1.
    await stage.run(_make_ctx("cam-1", _T0, _T0 + timedelta(milliseconds=400), 0.0, 1))

    # Next batch: cam-1 captured at t0 + 0.6 s (x = 0.84 m), cam-2 captured 400 ms later
    # at t0 + 1.0 s (x = 1.40 m), both received at t0 + 1.4 s.
    t_cam1 = _T0 + timedelta(milliseconds=600)
    t_cam2 = _T0 + timedelta(milliseconds=1000)
    t_recv = _T0 + timedelta(milliseconds=1400)
    ctx1 = _make_ctx("cam-1", t_cam1, t_recv, 0.84, 2)
    ctx2 = _make_ctx("cam-2", t_cam2, t_recv, 1.40, 2)
    await stage.run_many([ctx1, ctx2])

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1
    ph = open_phs[0]
    # PH last_seen_at must be the newer capture_time (t_cam2), never wall-clock event_time.
    assert ph.last_seen_at == t_cam2
    saved_obs = await obs_repo.list_by_ph(ph.ph_id, limit=10)
    assert [o.captured_at for o in saved_obs] == [_T0, t_cam1, t_cam2]


@pytest.mark.asyncio
async def test_constant_speed_walker_estimates_speed_within_5_percent() -> None:
    """A 1.4 m/s walker over 10 rounds with 400 ms processing delay is estimated within 5%.

    Pre-fix baseline (when batch_time used event_time = capture_time + 400 ms and
    step 4 updated against predicted_states[ph_idx] at batch_time):
    - At 1.0 s cadence: 1.0502 m/s (25.0% underestimate)
    - At 0.5 s cadence: 0.7361 m/s (47.4% underestimate)
    - At 0.36 s cadence: 0.6080 m/s (56.6% underestimate)
    """
    cfg = WorldTrackerConfig(zupt_consecutive_frames=99)
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)
    live_config = LiveConfigHolder(
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
    )
    stage = WorldTrackingStage(tracker=tracker, live_config=live_config, config=cfg)

    true_speed_m_s = 1.4
    delay = timedelta(milliseconds=400)
    for round_idx in range(10):
        cap_t = _T0 + timedelta(seconds=round_idx * 1.0)
        recv_t = cap_t + delay
        x_m = true_speed_m_s * round_idx * 1.0
        frame = FrameReady(
            camera_id="cam-1",
            frame_index=round_idx + 1,
            capture_time_unix_ns=int(cap_t.timestamp() * 1_000_000_000),
            received_time_unix_ns=int(recv_t.timestamp() * 1_000_000_000),
            minio_key=f"cam-1/{round_idx + 1}.jpg",
            width=1280,
            height=720,
        )
        ctx = FrameContext(frame=frame, event_time=recv_t, capture_time=cap_t)
        ctx.domain_detections = [
            Detection(
                detection_id=f"det-{round_idx + 1}",
                camera_id="cam-1",
                capture_time=cap_t,
                event_time=recv_t,
                bbox=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=300),
                confidence=0.95,
                floor_point=FloorPoint(x_mm=round(x_m * 1000.0), y_mm=0, calibrated=True),
                embedding=list(_EMB),
                crop_quality=0.85,
            )
        ]
        await stage.run(ctx)

    ph = (await ph_repo.list_open())[0]
    estimated_speed = ph.last_floor_speed_m_s
    rel_error = abs(estimated_speed - true_speed_m_s) / true_speed_m_s
    assert rel_error < 0.05, (
        f"expected speed within 5% of {true_speed_m_s}, got {estimated_speed:.4f}"
    )


def test_damped_velocity_prediction_monotonic_asymptote_and_covariance_psd() -> None:
    """With damped=True, displacement is monotonic in dt, tends to v*tau, and P is PSD."""
    state = KalmanState(
        mean=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        covariance=np.diag([0.09, 0.09, 0.25, 0.25]).astype(np.float64),
        updated_at=_T0,
    )
    tau = 3.0
    dts = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0)
    displacements: list[float] = []
    for dt in dts:
        pred = predict(
            state,
            _T0 + timedelta(seconds=dt),
            process_noise_accel_m_s2=0.5,
            velocity_decay_s=tau,
            damped=True,
        )
        displacements.append(float(pred.mean[0]))

    for prev_d, next_d in pairwise(displacements):
        assert next_d > prev_d, f"displacement not strictly monotonic: {displacements}"

    # Asymptote is v * tau = 1.0 * 3.0 = 3.0 m.
    assert abs(displacements[-1] - 3.0) < 1e-3

    # At normal frame interval dt = 0.36 s, damped and undamped differ by < 7% in displacement.
    t_frame = _T0 + timedelta(seconds=0.36)
    d_damped = float(predict(state, t_frame, 0.5, tau, damped=True).mean[0])
    d_legacy = float(predict(state, t_frame, 0.5, tau, damped=False).mean[0])
    assert abs(d_damped - d_legacy) / d_damped < 0.07

    # Covariance stays symmetric and PSD for dt in (0.1, 1, 5, 30).
    for dt in (0.1, 1.0, 5.0, 30.0):
        cov = predict(state, _T0 + timedelta(seconds=dt), 0.5, tau, damped=True).covariance
        assert np.allclose(cov, cov.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(cov)
        assert float(np.min(eigvals)) >= -1e-12


def test_dedup_time_skew_term_merges_skewed_observations_only() -> None:
    """With dedup_time_skew_speed_m_s=1.4, obs 1.2 m apart merge at 0.4 s skew, not at 0 s."""
    cfg_skew = WorldTrackerConfig(
        dedup_enabled=True,
        dedup_max_distance_m=0.9,
        dedup_time_skew_speed_m_s=1.4,
        dedup_max_distance_ceiling_m=1.5,
    )
    obs1 = _make_obs(camera_id="cam-1", x_m=0.0, y_m=0.0, captured_at=_T0, detection_id="d1")
    obs2_skewed = replace(
        _make_obs(
            camera_id="cam-2",
            x_m=1.2,
            y_m=0.0,
            captured_at=_T0 + timedelta(milliseconds=400),
            detection_id="d2",
        ),
        quality=0.60,
    )
    obs2_same_instant = replace(obs2_skewed, captured_at=_T0)

    # 0.4 s apart: gate is min(0.9 + 1.4 * 0.4, 1.5) = 1.46 m >= 1.2 m -> merges.
    deduped_skew, cluster_map_skew = dedup_observations([obs1, obs2_skewed], cfg_skew)
    assert len(deduped_skew) == 1
    assert set(cluster_map_skew[deduped_skew[0].detection_id]) == {"d1", "d2"}
    assert deduped_skew[0].captured_at == _T0 + timedelta(milliseconds=400)

    # Same instant: gate is 0.9 m < 1.2 m -> does not merge.
    deduped_same, _ = dedup_observations([obs1, obs2_same_instant], cfg_skew)
    assert len(deduped_same) == 2

    # Default (dedup_time_skew_speed_m_s = 0.0): does not merge even when 0.4 s apart.
    cfg_default = replace(cfg_skew, dedup_time_skew_speed_m_s=0.0)
    deduped_default, _ = dedup_observations([obs1, obs2_skewed], cfg_default)
    assert len(deduped_default) == 2
