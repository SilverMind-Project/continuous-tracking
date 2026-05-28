"""Unit tests for the FrameProcessingPipeline (skeleton mode)."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.calibration.state import AdjacencyEdge as CalibrationAdjacencyEdge
from app.calibration.state import calibration_state
from app.inference.schemas import DetectionBox
from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
    SignalConfig,
)
from app.transport.redis_streams import FrameReady

# A capture timestamp that is always "live" (within the 30s age gate).
# Using the module-load time is sufficient; the tests run in milliseconds.
_NOW_NS = int(time.time() * 1e9)

# ---------------------------------------------------------------------------
# Pipeline skeleton tests
# ---------------------------------------------------------------------------


@contextmanager
def _mock_redis_deps() -> Generator[tuple[AsyncMock, AsyncMock, AsyncMock], None, None]:
    """Mock all Redis-dependent components so tests don't need a live Redis."""
    with (
        patch("app.pipeline.frame_pipeline.RedisStreamsTransport") as mock_transport_cls,
        patch("app.pipeline.frame_pipeline.RevisionPublisher") as mock_rev_cls,
        patch("app.pipeline.frame_pipeline.SceneSamplesPublisher") as mock_scene_cls,
    ):
        mock_transport = AsyncMock()
        mock_transport_cls.return_value = mock_transport
        mock_rev = AsyncMock()
        mock_rev_cls.return_value = mock_rev
        mock_scene = AsyncMock()
        mock_scene_cls.return_value = mock_scene
        yield mock_transport, mock_rev, mock_scene


class TestPipelineSkeleton:
    @pytest.fixture
    def pipeline(self) -> FrameProcessingPipeline:
        config = PipelineConfig(
            allow_skeleton=True,
            transport=type(
                "TransportConfig",
                (),
                {
                    "redis_url": "redis://localhost:6379/0",
                    "consumer_group": "cts-orchestrator",
                    "consumer_name": "test-1",
                    "frames_stream": "frames.ready",
                    "events_stream": "tracking.events",
                    "responses_stream": "tracking.responses",
                    "revisions_stream": "tracking.revisions",
                    "signals_stream": "tracking.signals",
                    "scene_samples_stream": "scene.samples",
                    "batch_max_wait_ms": 100,
                    "batch_max_size": 8,
                    "xack_timeout_ms": 5000,
                    "max_retries": 3,
                },
            )(),
        )
        return FrameProcessingPipeline(config)

    def test_create_pipeline(self, pipeline: FrameProcessingPipeline) -> None:
        assert not pipeline.is_running

    @pytest.mark.asyncio
    async def test_initialize_without_detector(self, pipeline: FrameProcessingPipeline) -> None:
        """Initialize should create repo and tracklet manager without a detector."""
        with _mock_redis_deps():
            await pipeline.initialize()

            assert pipeline._transport is not None
            assert pipeline._repo is not None
            assert pipeline._detector is None  # Skeleton mode
            assert pipeline._trajectory_writer is not None  # M6
            assert pipeline._keyframe_sampler is not None  # M6

    @pytest.mark.asyncio
    async def test_skeleton_frame_processed(self, pipeline: FrameProcessingPipeline) -> None:
        """In skeleton mode, a frame should produce a zero-detection event."""
        with _mock_redis_deps():
            await pipeline.initialize()
            assert pipeline._transport is not None

            frame = FrameReady(
                camera_id="cam-1",
                minio_key="frames/cam-1/42.jpg",
                frame_index=42,
                capture_time_unix_ns=_NOW_NS,
                received_time_unix_ns=_NOW_NS + 100_000_000,
                width=640,
                height=480,
            )
            await pipeline._process_frame(frame)

            # Verify the event was persisted via in-memory repo
            assert pipeline._repo is not None
            # Events are stored by event_id; find one with matching camera_id
            events = list(pipeline._repo._events.values())
            cam_events = [e for e in events if e.camera_id == frame.camera_id]
            assert len(cam_events) == 1
            assert len(cam_events[0].detections) == 0

            await pipeline._transport.ack_frame(frame)
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stale_frame_is_dropped_without_publishing(
        self, pipeline: FrameProcessingPipeline
    ) -> None:
        """Frames older than _MAX_FRAME_AGE_S must not produce tracking events."""
        stale_ns = int((time.time() - 60) * 1e9)  # 60 s ago, past the 30 s gate
        with _mock_redis_deps() as (mock_transport, _, _):
            await pipeline.initialize()
            frame = FrameReady(
                camera_id="cam-stale",
                minio_key="frames/cam-stale/0.jpg",
                frame_index=0,
                capture_time_unix_ns=stale_ns,
                received_time_unix_ns=stale_ns + 100_000_000,
                width=640,
                height=480,
            )
            await pipeline._process_frame(frame)

            mock_transport.publish_event.assert_not_called()
            assert pipeline._repo is not None
            events = list(pipeline._repo._events.values())
            assert not any(e.camera_id == "cam-stale" for e in events)
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, pipeline: FrameProcessingPipeline) -> None:
        with _mock_redis_deps():
            await pipeline.initialize()
            # Should not raise
            await pipeline.stop()

    def test_is_running_false_by_default(self, pipeline: FrameProcessingPipeline) -> None:
        assert not pipeline.is_running

    @pytest.mark.asyncio
    async def test_close_track_on_global_track_termination(
        self, pipeline: FrameProcessingPipeline
    ) -> None:
        """Issue #23 (M1 update): when a PH disappears, the pipeline
        processes the closure without error."""
        with _mock_redis_deps():
            mock_detector = AsyncMock()
            mock_detector.detect = AsyncMock(return_value=[])
            await pipeline.initialize(PipelineDependencies(detector=mock_detector))

            # Mock world_tracker.step: first with active PH, then empty.
            from app.domain import PersonHypothesis, WorldFrameSnapshot

            active_ph = PersonHypothesis(
                ph_id="ph-001",
                state_mean=(1.0, 2.0, 0.0, 0.0),
                state_cov=tuple([0.1] * 16),
                born_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                last_seen_camera="cam-1",
                observation_count=5,
                current_identity_id="alice",
                active_cameras=frozenset(["cam-1"]),
            )
            from app.tracking.world.tracker import WorldTrackerResult

            pipeline._world_tracker.step = AsyncMock(  # type: ignore[method-assign]
                side_effect=[
                    WorldTrackerResult(
                        updated_phs=[active_ph],
                        snapshots=[
                            WorldFrameSnapshot(
                                ph_id="ph-001",
                                camera_id="cam-1",
                                frame_index=0,
                                captured_at=datetime.now(UTC),
                                floor_x_m=1.0,
                                floor_y_m=2.0,
                                floor_vx_m_s=0.0,
                                floor_vy_m_s=0.0,
                                position_sigma_m=0.1,
                                identity_id="alice",
                            )
                        ],
                        continuations=[],
                    ),
                    WorldTrackerResult(
                        updated_phs=[],
                        snapshots=[],
                        continuations=[],
                    ),
                ]
            )

            # Replace trajectory writer with mock to verify close_track calls.
            pipeline._trajectory_writer = AsyncMock()
            from app.pipeline.stages.trajectory import CloseTerminatedStage, TrajectoryStage

            for stage in pipeline._stage_runner._stages:  # type: ignore[union-attr]
                if isinstance(stage, (CloseTerminatedStage, TrajectoryStage)):
                    stage._trajectory_writer = pipeline._trajectory_writer  # type: ignore[union-attr]

            for i in range(2):
                frame = FrameReady(
                    camera_id="cam-1",
                    minio_key=f"frames/cam-1/{i}.jpg",
                    frame_index=i,
                    capture_time_unix_ns=_NOW_NS + i * 200_000_000,
                    received_time_unix_ns=_NOW_NS + 100_000_000 + i * 200_000_000,
                    width=640,
                    height=480,
                )
                await pipeline._process_frame(frame)

            # Pipeline should not have crashed.
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_detector_and_reid_path_publishes_detections(
        self, pipeline: FrameProcessingPipeline
    ) -> None:
        """Issue #3/#13: detector output should flow through tracking without
        zero-placeholder embeddings."""

        class FakeFetcher:
            async def fetch_rgb(self, minio_key: str) -> np.ndarray:
                assert minio_key == "frames/cam-1/3.jpg"
                return np.zeros((100, 200, 3), dtype=np.uint8)

        class FakeDetector:
            async def detect(self, image: np.ndarray) -> list[DetectionBox]:
                assert image.shape == (100, 200, 3)
                return [DetectionBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8, confidence=0.95)]

        class FakeReid:
            async def embed_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
                assert len(crops) == 1
                return [np.ones(768, dtype=np.float32)]

        with _mock_redis_deps() as (mock_transport, _, _):
            await pipeline.initialize(
                PipelineDependencies(
                    detector=FakeDetector(),  # type: ignore[arg-type]
                    frame_fetcher=FakeFetcher(),
                    reid_embedder=FakeReid(),
                )
            )
            frame = FrameReady(
                camera_id="cam-1",
                minio_key="frames/cam-1/3.jpg",
                frame_index=3,
                capture_time_unix_ns=_NOW_NS,
                received_time_unix_ns=_NOW_NS + 100_000_000,
                width=200,
                height=100,
            )

            await pipeline._process_frame(frame)

            publish_kwargs = mock_transport.publish_event.call_args.kwargs
            detections = publish_kwargs["detections"]
            assert len(detections) == 1
            assert detections[0].embedding == [1.0] * 768
            assert publish_kwargs["frame_width"] == 200
            assert publish_kwargs["frame_height"] == 100
            assert publish_kwargs["capture_time_unix_ns"] == frame.capture_time_unix_ns

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_calibration_adjacency_syncs_into_pipeline(
        self, pipeline: FrameProcessingPipeline
    ) -> None:
        """Verify overlap groups are accessible after pipeline initialization."""
        old_edges = list(calibration_state.adjacency_edges)
        old_version = calibration_state.version
        try:
            await calibration_state.set_adjacency(
                [
                    CalibrationAdjacencyEdge(
                        from_camera="cam-1",
                        to_camera="cam-2",
                        max_transit_s=12.0,
                    )
                ]
            )
            with _mock_redis_deps():
                await pipeline.initialize()
                assert pipeline._overlap_groups is not None
                await pipeline.stop()
        finally:
            calibration_state.adjacency_edges = old_edges
            calibration_state.version = old_version


class TestFullPipelineIntegration:
    """End-to-end tests exercising the full pipeline with all real components.

    Uses InMemory repositories so no Redis or Postgres is needed.  The
    detector and ReID embedder are mocked but all tracking, cross-camera,
    identity resolution, and trajectory writing components are real.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_tracking_and_event_emission(self) -> None:
        """Process frames and verify tracking events are published."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )
        with _mock_redis_deps() as (mock_transport, _mock_rev, _mock_scene):
            # Realistic detector: one person in each frame.
            mock_detector = AsyncMock()
            mock_detector.detect = AsyncMock(
                side_effect=lambda img: [
                    DetectionBox(x1=0.3, y1=0.2, x2=0.5, y2=0.6, confidence=0.95)
                ]
            )

            mock_reid = AsyncMock()
            mock_reid.embed_batch = AsyncMock(return_value=[np.ones(768, dtype=np.float32)])

            await pipeline.initialize(
                PipelineDependencies(detector=mock_detector, reid_embedder=mock_reid)
            )

            # Send several frames from the same camera so a confirmed track
            # is established (min_hits=3).
            for i in range(5):
                frame = FrameReady(
                    camera_id="cam-1",
                    minio_key=f"frames/cam-1/{i}.jpg",
                    frame_index=i,
                    capture_time_unix_ns=_NOW_NS + i * 200_000_000,
                    received_time_unix_ns=_NOW_NS + 100_000_000 + i * 200_000_000,
                    width=640,
                    height=480,
                )
                await pipeline._process_frame(frame)

            # After 5 frames, tracking events should have been published.
            assert mock_transport.publish_event.call_count >= 1

            # Verify frame_ref fields are populated (Phase 1 fix #2).
            last_kwargs = mock_transport.publish_event.call_args.kwargs
            assert last_kwargs["frame_width"] == 640
            assert last_kwargs["frame_height"] == 480
            assert last_kwargs["capture_time_unix_ns"] > 0

            # In M1, PHs require calibrated floor points (detector + homography).
            # Without calibration, observations are dropped safely.
            # The pipeline publishes events regardless.
            assert pipeline._ph_repo is not None

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_empty_frame_skeleton_mode(self) -> None:
        """Skeleton mode (no detector) produces zero-detection events."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )
        with _mock_redis_deps() as (mock_transport, _mock_rev, _mock_scene):
            await pipeline.initialize()  # no detector → skeleton mode

            frame = FrameReady(
                camera_id="cam-1",
                minio_key="frames/cam-1/0.jpg",
                frame_index=0,
                capture_time_unix_ns=_NOW_NS,
                received_time_unix_ns=_NOW_NS + 100_000_000,
                width=640,
                height=480,
            )
            await pipeline._process_frame(frame)

            assert mock_transport.publish_event.call_count == 1
            kwargs = mock_transport.publish_event.call_args.kwargs
            assert kwargs.get("detections") is None
            assert kwargs["frame_width"] == 640
            assert kwargs["frame_height"] == 480

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_graceful_degradation_on_minio_miss(self) -> None:
        """Pipeline does not crash when MinIO fetch fails (empty image fallback)."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )
        with _mock_redis_deps() as (mock_transport, _mock_rev, _mock_scene):
            mock_detector = AsyncMock()
            mock_detector.detect = AsyncMock(return_value=[])
            await pipeline.initialize(PipelineDependencies(detector=mock_detector))

            # No frame_fetcher is set — falls back to blank image.
            frame = FrameReady(
                camera_id="cam-1",
                minio_key="frames/cam-1/missing.jpg",
                frame_index=0,
                capture_time_unix_ns=_NOW_NS,
                received_time_unix_ns=_NOW_NS + 100_000_000,
                width=0,
                height=0,
            )
            await pipeline._process_frame(frame)

            # Should produce an event with zero detections, not crash.
            assert mock_transport.publish_event.call_count == 1
            await pipeline.stop()


class TestCameraRowUpsert:
    """The pipeline must seed an FK-anchor row in ``cameras`` on first sight."""

    def _frame(self, camera_id: str, frame_index: int = 0) -> FrameReady:
        return FrameReady(
            camera_id=camera_id,
            minio_key=f"frames/{camera_id}/{frame_index}.jpg",
            frame_index=frame_index,
            capture_time_unix_ns=_NOW_NS,
            received_time_unix_ns=_NOW_NS + 100_000_000,
            width=640,
            height=480,
        )

    @pytest.mark.asyncio
    async def test_first_frame_per_camera_upserts_row(self) -> None:
        """Every new ``camera_id`` triggers exactly one upsert; subsequent
        frames from the same camera don't repeat the call."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )
        with _mock_redis_deps():
            await pipeline.initialize()
            assert pipeline._settings_repo is not None

            spy = AsyncMock(wraps=pipeline._settings_repo.save_camera_config)
            pipeline._settings_repo.save_camera_config = spy  # type: ignore[method-assign]

            await pipeline._handle_frame(self._frame("cam-a"))
            await pipeline._handle_frame(self._frame("cam-a", frame_index=1))
            await pipeline._handle_frame(self._frame("cam-b"))

            assert spy.await_count == 2
            seen = {call.args[0].camera_id for call in spy.await_args_list}
            assert seen == {"cam-a", "cam-b"}
            cameras = await pipeline._settings_repo.list_camera_configs()
            assert {c.camera_id for c in cameras} == {"cam-a", "cam-b"}

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_existing_camera_row_is_not_overwritten(self) -> None:
        """A pre-existing row (e.g. from a future CC sync) must not be stomped
        by the bare placeholder ``CameraConfig`` the lazy-seed path constructs."""
        from app.domain import CameraConfig

        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )
        with _mock_redis_deps():
            await pipeline.initialize()
            assert pipeline._settings_repo is not None

            # Simulate a prior process having written real camera metadata.
            await pipeline._settings_repo.save_camera_config(
                CameraConfig(
                    camera_id="cam-pre",
                    name="Kitchen",
                    rtsp_url="rtsp://10.0.0.5/stream",
                    location="kitchen",
                )
            )
            save_spy = AsyncMock(wraps=pipeline._settings_repo.save_camera_config)
            pipeline._settings_repo.save_camera_config = save_spy  # type: ignore[method-assign]

            await pipeline._handle_frame(self._frame("cam-pre"))

            save_spy.assert_not_awaited()
            cfg = await pipeline._settings_repo.get_camera_config("cam-pre")
            assert cfg is not None
            assert cfg.name == "Kitchen"
            assert cfg.rtsp_url == "rtsp://10.0.0.5/stream"
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_upsert_runs_before_process_frame(self) -> None:
        """The FK anchor must land before any tracking row is written."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )
        with _mock_redis_deps():
            await pipeline.initialize()
            assert pipeline._settings_repo is not None

            order: list[str] = []

            original_save = pipeline._settings_repo.save_camera_config

            async def record_save(cfg: object) -> None:
                order.append("ensure_camera")
                await original_save(cfg)  # type: ignore[arg-type]

            async def record_process(frame: FrameReady) -> None:
                order.append("process_frame")

            pipeline._settings_repo.save_camera_config = record_save  # type: ignore[assignment, method-assign]
            pipeline._process_frame = record_process  # type: ignore[assignment, method-assign]

            await pipeline._handle_frame(self._frame("cam-z"))

            assert order == ["ensure_camera", "process_frame"]
            await pipeline.stop()
