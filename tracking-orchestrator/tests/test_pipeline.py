"""Unit tests for the FrameProcessingPipeline (skeleton mode)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline.frame_pipeline import FrameProcessingPipeline, PipelineConfig
from app.transport.redis_streams import FrameReady

# ---------------------------------------------------------------------------
# Pipeline skeleton tests
# ---------------------------------------------------------------------------


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
        with (
            patch("app.pipeline.frame_pipeline.RedisStreamsTransport") as mock_transport_cls,
            patch("app.pipeline.frame_pipeline.RevisionPublisher") as mock_rev_cls,
        ):
            mock_transport = AsyncMock()
            mock_transport_cls.return_value = mock_transport
            mock_rev = AsyncMock()
            mock_rev_cls.return_value = mock_rev

            await pipeline.initialize()

            assert pipeline._transport is not None
            assert pipeline._repo is not None
            assert pipeline._detector is None  # Skeleton mode

    @pytest.mark.asyncio
    async def test_skeleton_frame_processed(self, pipeline: FrameProcessingPipeline) -> None:
        """In skeleton mode, a frame should produce a zero-detection event."""
        with (
            patch("app.pipeline.frame_pipeline.RedisStreamsTransport") as mock_transport_cls,
            patch("app.pipeline.frame_pipeline.RevisionPublisher") as mock_rev_cls,
        ):
            mock_transport = AsyncMock()
            mock_transport_cls.return_value = mock_transport
            mock_rev = AsyncMock()
            mock_rev_cls.return_value = mock_rev

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
        with (
            patch("app.pipeline.frame_pipeline.RedisStreamsTransport") as mock_transport_cls,
            patch("app.pipeline.frame_pipeline.RevisionPublisher") as mock_rev_cls,
        ):
            mock_transport = AsyncMock()
            mock_transport_cls.return_value = mock_transport
            mock_rev = AsyncMock()
            mock_rev_cls.return_value = mock_rev

            await pipeline.initialize()
            # Should not raise
            await pipeline.stop()

    def test_is_running_false_by_default(self, pipeline: FrameProcessingPipeline) -> None:
        assert not pipeline.is_running
