"""Unit tests for the FrameProcessingPipeline (skeleton mode)."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.calibration.state import AdjacencyEdge as CalibrationAdjacencyEdge
from app.calibration.state import calibration_state
from app.inference.schemas import DetectionBox
from app.pipeline.frame_pipeline import FrameProcessingPipeline, PipelineConfig
from app.transport.redis_streams import FrameReady

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
                capture_time_unix_ns=1700000000000000000,
                received_time_unix_ns=1700000000100000000,
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
        """Issue #23: when a global track disappears from the active list,
        the trajectory writer must have close_track called to close the open
        dwell and prevent unbounded memory growth."""
        with _mock_redis_deps():
            # Mock detector so the full pipeline (not skeleton) runs.
            mock_detector = AsyncMock()
            mock_detector.detect = AsyncMock(return_value=[])
            await pipeline.initialize(detector=mock_detector)

            # Mock tracklet manager to return an active tracklet so the M5
            # (cross-camera + identity) block is entered even with empty
            # detections.
            pipeline._tracklet_manager.get_active_tracklets = (  # type: ignore[method-assign,union-attr]
                lambda: [
                    type(
                        "Tracklet",
                        (),
                        {
                            "tracklet_id": "tl-1",
                            "camera_id": "cam-1",
                            "detection_ids": ["det-1"],
                            "started_at": None,
                            "ended_at": None,
                            "state": "active",
                        },
                    )()
                ]
            )

            # The cross-camera associator returns an active global track on
            # the first call and nothing on the second (tracklet terminated).
            mock_associate = AsyncMock(
                side_effect=[
                    [
                        type(
                            "GlobalTrack",
                            (),
                            {
                                "global_track_id": "gt-001",
                                "camera_ids": ["cam-1"],
                                "tracklet_ids": ["tl-1"],
                                "started_at": None,
                                "last_seen_at": None,
                                "current_identity_id": None,
                                "state": "active",
                            },
                        )(),
                    ],
                    [],  # tracklet terminated — global track no longer active
                ]
            )
            pipeline._cross_camera.associate = mock_associate  # type: ignore[method-assign,union-attr]

            # Mock identity resolver so it returns decisions with an identity.
            from app.domain import (
                IdentityDecision,
                PosteriorDist,
                ResolveOutcome,
            )

            mock_resolver = AsyncMock()
            outcome = ResolveOutcome()
            outcome.decisions = [
                IdentityDecision(
                    global_track_id="gt-001",
                    identity_id="alice",
                    posterior=PosteriorDist(distribution={"alice": 0.95}),
                    revises_previous=False,
                )
            ]
            mock_resolver.resolve = AsyncMock(return_value=outcome)
            pipeline._identity_resolver = mock_resolver

            # Replace the real trajectory writer with a mock so we can verify
            # close_track is called for terminated global tracks.
            pipeline._trajectory_writer = AsyncMock()

            frame = FrameReady(
                camera_id="cam-1",
                minio_key="frames/cam-1/1.jpg",
                frame_index=1,
                capture_time_unix_ns=1700000000000000000,
                received_time_unix_ns=1700000000100000000,
                width=640,
                height=480,
            )

            # Frame 1: global track is active — writes a trajectory point.
            await pipeline._process_frame(frame)

            # Frame 2: global track is gone — should trigger close_track.
            frame2 = FrameReady(
                camera_id="cam-1",
                minio_key="frames/cam-1/2.jpg",
                frame_index=2,
                capture_time_unix_ns=1700000001000000000,
                received_time_unix_ns=1700000001100000000,
                width=640,
                height=480,
            )
            await pipeline._process_frame(frame2)

            # Verify close_track was called for the terminated global track.
            assert pipeline._trajectory_writer is not None
            calls = pipeline._trajectory_writer.close_track.call_args_list
            assert len(calls) == 1
            assert calls[0].args[0] == "gt-001"

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
                detector=FakeDetector(),  # type: ignore[arg-type]
                frame_fetcher=FakeFetcher(),
                reid_embedder=FakeReid(),
            )
            frame = FrameReady(
                camera_id="cam-1",
                minio_key="frames/cam-1/3.jpg",
                frame_index=3,
                capture_time_unix_ns=1700000000000000000,
                received_time_unix_ns=1700000000100000000,
                width=200,
                height=100,
            )

            await pipeline._process_frame(frame)

            publish_kwargs = mock_transport.publish_event.call_args.kwargs
            detections = publish_kwargs["detections"]
            assert len(detections) == 1
            assert detections[0].embedding == [1.0] * 768
            assert publish_kwargs["detection_count"] == 1

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_calibration_adjacency_syncs_into_pipeline(
        self, pipeline: FrameProcessingPipeline
    ) -> None:
        """Issue #30: operator-pushed adjacency should affect the pipeline graph."""
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
                pipeline._sync_adjacency()
                assert pipeline._adjacency is not None
                assert pipeline._adjacency.get_max_transition("cam-1", "cam-2") == 12.0
                await pipeline.stop()
        finally:
            calibration_state.adjacency_edges = old_edges
            calibration_state.version = old_version
