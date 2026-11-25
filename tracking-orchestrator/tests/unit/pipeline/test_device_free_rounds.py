"""Pipeline tests for device-free rounds, mode firewall, serialization, and publication."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    AnonymousPositionObservation,
    FloorPoint,
    Identity,
    PositionSensor,
    WearableSiteConfig,
)
from app.observability import metrics as metrics_module
from app.observability.metrics import DeviceFreeRoundOutcome, DeviceFreeShadowOutcome
from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
)
from app.pipeline.gallery_cache import GalleryCache
from app.pipeline.stages.base import StageRunner
from app.pipeline.stages.trajectory import TrajectoryStage
from app.services.camera_room_map import (
    CameraRoomBinding,
    CameraRoomMap,
    RoomPolygonBinding,
    RoomPolygonMap,
)
from app.services.position_track_buffer import PositionTrackBuffer
from app.services.wearable_site_config_map import WearableSiteConfigMap
from app.storage.base import (
    InMemoryGaitBoutRepository,
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    InMemoryTrajectoryRepository,
    InMemoryWorldObservationRepository,
)
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.position_sensor_config import PositionSensorConfig
from app.transport.redis_streams import FrameReady, PositionOnlyDetection

_T0 = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)


async def _rooms() -> RoomPolygonMap:
    polygons = RoomPolygonMap()
    await polygons.set_all(
        [
            RoomPolygonBinding(
                room_id="living",
                room_name="Living Room",
                polygon_m=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
                bound_at=_T0,
            ),
            RoomPolygonBinding(
                room_id="bathroom",
                room_name="Bathroom",
                polygon_m=[(6.0, 0.0), (10.0, 0.0), (10.0, 5.0), (6.0, 5.0)],
                bound_at=_T0,
            ),
        ]
    )
    return polygons


def _site_config_map() -> WearableSiteConfigMap:
    return WearableSiteConfigMap(
        WearableSiteConfig(
            floor_plan_id="fp-1",
            position_sensors=(
                PositionSensor(
                    sensor_id="radar-bath-1",
                    room_id="bathroom",
                    x_mm=8000,
                    y_mm=2500,
                    yaw_deg=0.0,
                    calibration_residual_m=0.10,
                ),
            ),
            version="v1",
        )
    )


def _patch_pipeline_network(
    monkeypatch: pytest.MonkeyPatch, published: list[dict[str, object]]
) -> None:
    async def _noop(self: object) -> None:
        return None

    async def _fake_publish(
        self: object,
        camera_id: str,
        event_time: datetime,
        frame_index: int,
        **kwargs: object,
    ) -> str:
        published.append(
            {
                "camera_id": camera_id,
                "event_time": event_time,
                "frame_index": frame_index,
                **kwargs,
            }
        )
        return "1-0"

    monkeypatch.setattr("app.pipeline.frame_pipeline.RedisStreamsTransport.connect", _noop)
    monkeypatch.setattr("app.pipeline.frame_pipeline.RedisStreamsTransport.disconnect", _noop)
    monkeypatch.setattr(
        "app.pipeline.frame_pipeline.RedisStreamsTransport.publish_event",
        _fake_publish,
    )
    monkeypatch.setattr("app.pipeline.frame_pipeline.RevisionPublisher.connect", _noop)
    monkeypatch.setattr("app.pipeline.frame_pipeline.RevisionPublisher.disconnect", _noop)
    monkeypatch.setattr("app.pipeline.frame_pipeline.PHContinuationPublisher.connect", _noop)
    monkeypatch.setattr("app.pipeline.frame_pipeline.PHContinuationPublisher.disconnect", _noop)
    monkeypatch.setattr("app.pipeline.frame_pipeline.SceneSamplesPublisher.connect", _noop)
    monkeypatch.setattr("app.pipeline.frame_pipeline.SceneSamplesPublisher.disconnect", _noop)
    monkeypatch.setattr("app.pipeline.frame_pipeline.SignalPublisher.connect", _noop)
    monkeypatch.setattr("app.pipeline.frame_pipeline.SignalPublisher.disconnect", _noop)


async def _init_pipeline(
    mode: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    ph_repo: InMemoryPHRepository | None = None,
    traj_repo: InMemoryTrajectoryRepository | None = None,
    gait_bout_repo: InMemoryGaitBoutRepository | None = None,
    gallery_repo: InMemoryGalleryRepository | None = None,
    track_buf: PositionTrackBuffer | None = None,
) -> tuple[FrameProcessingPipeline, list[dict[str, object]]]:
    published: list[dict[str, object]] = []
    _patch_pipeline_network(monkeypatch, published)
    cfg = PipelineConfig(
        allow_skeleton=True,
        signals_enabled=False,
        pose_enabled=False,
        world_tracker=WorldTrackerConfig(min_observations_to_publish=1),
        position_sensor=PositionSensorConfig(
            mode=mode,  # type: ignore[arg-type]
            round_interval_s=0.05,
            min_track_age_s=1.0,
            round_max_track_age_s=5.0,
            buffer_max_age_s=10.0,
        ),
    )
    pipeline = FrameProcessingPipeline(cfg)
    await pipeline.initialize(
        PipelineDependencies(
            ph_repo=ph_repo or InMemoryPHRepository(),
            obs_repo=InMemoryWorldObservationRepository(),
            trajectory_repo=traj_repo or InMemoryTrajectoryRepository(),
            gait_bout_repo=gait_bout_repo or InMemoryGaitBoutRepository(),
            gallery_repo=gallery_repo or InMemoryGalleryRepository(),
            position_track_source=track_buf,
        )
    )
    crm = CameraRoomMap()
    await crm.set_all(
        [
            CameraRoomBinding(
                camera_id="cam-1",
                room_id="living",
                room_name="Living Room",
                bound_at=_T0,
            )
        ]
    )
    pipeline.set_camera_room_map(crm)
    pipeline._live_config.room_polygon_map = await _rooms()
    pipeline.set_wearable_site_config_map(_site_config_map())
    return pipeline, published


@pytest.mark.asyncio
async def test_mode_off_starts_no_device_free_round_task(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=16)
    pipeline, published = await _init_pipeline("off", monkeypatch=monkeypatch, track_buf=buf)
    monkeypatch.setattr(pipeline, "_consume_loop", AsyncMock(return_value=None))
    monkeypatch.setattr(pipeline, "_stream_metrics_loop", AsyncMock(return_value=None))

    created_coroutines: list[str] = []
    orig_create_task = asyncio.create_task

    def _recording_task(coro: object, **kwargs: object) -> asyncio.Task[object]:
        created_coroutines.append(getattr(getattr(coro, "cr_code", None), "co_name", ""))
        return orig_create_task(coro, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(asyncio, "create_task", _recording_task)
    await pipeline.start()
    try:
        assert "_device_free_round_loop" not in created_coroutines
        assert await pipeline.run_device_free_round_once(now=_T0) == 0
        assert published == []
    finally:
        await pipeline.stop()


@pytest.mark.asyncio
async def test_shadow_mode_evaluates_outcomes_without_touching_state_or_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CollectorRegistry()
    test_metrics = metrics_module.build_metrics(registry)
    monkeypatch.setattr(metrics_module, "metrics", test_metrics)

    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=16)
    ph_repo = InMemoryPHRepository()
    traj_repo = InMemoryTrajectoryRepository()
    pipeline, published = await _init_pipeline(
        "shadow",
        monkeypatch=monkeypatch,
        ph_repo=ph_repo,
        traj_repo=traj_repo,
        track_buf=buf,
    )

    # Confirm two tracks: bathroom (would_spawn) and living room with camera (would_drop).
    for dt_s in (0.0, 1.2):
        ts = _T0 + timedelta(seconds=dt_s)
        buf.add(
            AnonymousPositionObservation(
                sensor_id="radar-bath-1",
                sensor_track_id="t-bath",
                measured_at=ts,
                x_m=8.0,
                y_m=2.5,
                cov_m2=(0.04, 0.0, 0.0, 0.04),
                calibration_residual_m=0.10,
            )
        )
        buf.add(
            AnonymousPositionObservation(
                sensor_id="radar-bath-1",
                sensor_track_id="t-living",
                measured_at=ts,
                x_m=2.0,
                y_m=2.5,
                cov_m2=(0.04, 0.0, 0.0, 0.04),
                calibration_residual_m=0.10,
            )
        )

    processed = await pipeline.run_device_free_round_once(now=_T0 + timedelta(seconds=1.3))
    assert processed == 2
    assert await ph_repo.list_open() == []
    assert pipeline._world_tracker is not None
    assert len(pipeline._world_tracker.ph_floor_history._samples) == 0
    assert published == []
    # Buffer must NOT be consumed in shadow mode.
    peeked = buf.peek_round_tracks(
        now=_T0 + timedelta(seconds=1.3),
        max_age_s=5.0,
        min_track_age_s=1.0,
    )
    assert len(peeked) == 2
    # Second shadow tick at same timestamps does not double-count.
    assert await pipeline.run_device_free_round_once(now=_T0 + timedelta(seconds=1.3)) == 0
    assert test_metrics.cts_device_free_shadow_total.labels(
        outcome=DeviceFreeShadowOutcome.WOULD_SPAWN
    )._value.get() == pytest.approx(1.0)
    assert test_metrics.cts_device_free_shadow_total.labels(
        outcome=DeviceFreeShadowOutcome.WOULD_DROP
    )._value.get() == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_enabled_mode_runs_reduced_stages_publishes_position_only_and_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CollectorRegistry()
    test_metrics = metrics_module.build_metrics(registry)
    monkeypatch.setattr(metrics_module, "metrics", test_metrics)

    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=16)
    ph_repo = InMemoryPHRepository()
    traj_repo = InMemoryTrajectoryRepository()
    gait_bout_repo = InMemoryGaitBoutRepository()
    gallery_repo = InMemoryGalleryRepository()
    await gallery_repo.upsert_identity(
        Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0)
    )

    pipeline, published = await _init_pipeline(
        "enabled",
        monkeypatch=monkeypatch,
        ph_repo=ph_repo,
        traj_repo=traj_repo,
        gait_bout_repo=gait_bout_repo,
        gallery_repo=gallery_repo,
        track_buf=buf,
    )
    # Attach a real GalleryCache with high TTL to verify _begin_tracker_round() invalidates it.
    cache = GalleryCache(gallery_repo, max_age_s=999.0)
    pipeline._gallery_cache = cache
    assert pipeline._device_free_post_world_runner is not None
    # Verify no vision stages (pose/reid/keyframes) are in the device-free post-world runner.
    stage_names = [getattr(s, "name", "") for s in pipeline._device_free_post_world_runner._stages]
    assert "device_free_publish" in stage_names
    assert "reid_candidates" not in stage_names
    assert "keyframes" not in stage_names
    assert "posture" not in stage_names
    assert "fall_detection" not in stage_names

    # Seed cache before round to verify invalidation.
    await cache.list_gallery_entries_for_tracklets({"alice"})
    assert cache._entries_by_tracklets

    for dt_s in (0.0, 1.1):
        buf.add(
            AnonymousPositionObservation(
                sensor_id="radar-bath-1",
                sensor_track_id="t-bath",
                measured_at=_T0 + timedelta(seconds=dt_s),
                x_m=8.0,
                y_m=2.5,
                cov_m2=(0.04, 0.0, 0.0, 0.04),
                calibration_residual_m=0.10,
                speed_m_s=1.1,
            )
        )

    processed = await pipeline.run_device_free_round_once(now=_T0 + timedelta(seconds=1.2))
    assert processed == 1
    assert cache._entries_by_tracklets == {}
    assert len(await ph_repo.list_open()) == 1
    assert len(published) == 1
    ev = published[0]
    assert ev["camera_id"] == "sensor:radar-bath-1"
    assert ev["room_name"] == "Bathroom"
    assert ev.get("detections") is None
    pos_dets = ev.get("position_only_detections")
    assert isinstance(pos_dets, list) and len(pos_dets) == 1
    det = pos_dets[0]
    assert isinstance(det, PositionOnlyDetection)
    assert det.floor_point == FloorPoint(x_mm=8000, y_mm=2500, calibrated=True)
    # Trajectories are written, while device-free gait bouts remain skipped.
    trajs = await traj_repo.list_trajectory_points()
    dwells = await traj_repo.list_room_dwells()
    assert len(trajs) == 1
    assert len(dwells) == 1
    assert trajs[0].primary_camera_id == "sensor:radar-bath-1"
    assert dwells[0].room_name == "Bathroom"
    assert await gait_bout_repo.list_bouts() == []
    assert test_metrics.cts_device_free_rounds_total.labels(
        outcome=DeviceFreeRoundOutcome.COMPLETED
    )._value.get() == pytest.approx(1.0)
    await pipeline.stop()


@pytest.mark.asyncio
async def test_world_round_lock_serializes_camera_and_device_free_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buf = PositionTrackBuffer(max_age_s=10.0, track_gap_s=2.0, max_tracks=16)
    pipeline, _ = await _init_pipeline("enabled", monkeypatch=monkeypatch, track_buf=buf)
    pipeline._detector = MagicMock()
    for dt_s in (0.0, 1.1):
        buf.add(
            AnonymousPositionObservation(
                sensor_id="radar-bath-1",
                sensor_track_id="t-1",
                measured_at=_T0 + timedelta(seconds=dt_s),
                x_m=8.0,
                y_m=2.5,
                cov_m2=(0.04, 0.0, 0.0, 0.04),
                calibration_residual_m=0.10,
            )
        )

    in_flight = 0
    max_concurrent = 0

    class _FakeWorldStage:
        name = "world_tracking"

        async def run(self, _ctx: object) -> None:
            nonlocal in_flight, max_concurrent
            in_flight += 1
            max_concurrent = max(max_concurrent, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1

        async def run_device_free_round(self, *_args: object, **_kwargs: object) -> None:
            nonlocal in_flight, max_concurrent
            in_flight += 1
            max_concurrent = max(max_concurrent, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1

    class _NoopRunner(StageRunner):
        def __init__(self) -> None:
            super().__init__([])

        async def run(self, _ctx: object) -> None:  # type: ignore[override]
            return None

    fake_runner = _NoopRunner()
    pipeline._io_runner = fake_runner
    pipeline._pre_world_runner = fake_runner
    pipeline._post_world_runner = fake_runner
    pipeline._device_free_post_world_runner = fake_runner
    pipeline._world_tracking_stage = _FakeWorldStage()  # type: ignore[assignment]

    frame = FrameReady(camera_id="cam-1", frame_index=1, minio_key="k.jpg")
    await asyncio.gather(
        pipeline._process_frame(frame),
        pipeline.run_device_free_round_once(now=_T0 + timedelta(seconds=1.2)),
    )
    assert max_concurrent == 1
    assert isinstance(TrajectoryStage, type)
    await pipeline.stop()
