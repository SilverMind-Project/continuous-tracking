"""Invariant 2: tracker.update must be called on every frame, even with
zero detections, so BoT-SORT ages lost tracklets toward close_grace_frames."""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.inference.schemas import DetectionBox
from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
    SignalConfig,
)
from app.transport.redis_streams import FrameReady

_NOW_NS = int(time.time() * 1e9)


@contextmanager
def _mock_redis_deps():
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


class FakeFetcher:
    async def fetch_rgb(self, minio_key: str) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class FakeReid:
    async def embed_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        return [np.zeros(768, dtype=np.float32) for _ in crops]


class TestEmptyFrameHandling:
    @pytest.mark.asyncio
    async def test_tracker_update_called_on_empty_frame(self) -> None:
        """tracker.update must be called even when the detector returns no detections."""

        class EmptyDetector:
            async def detect(self, image: np.ndarray) -> list[DetectionBox]:
                return []

        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )

        with _mock_redis_deps():
            await pipeline.initialize(
                PipelineDependencies(
                    detector=EmptyDetector(),  # type: ignore[arg-type]
                    frame_fetcher=FakeFetcher(),
                    reid_embedder=FakeReid(),
                )
            )

            # Spy on tracker.update to record calls.
            calls: list[list] = []
            original_update = pipeline._tracker.update

            def spy_update(*args, **kwargs):
                calls.append(args[1] if len(args) > 1 else kwargs.get("detections", []))
                return original_update(*args, **kwargs)

            assert pipeline._tracker is not None
            pipeline._tracker.update = spy_update  # type: ignore[method-assign]

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

            assert len(calls) == 1, f"tracker.update should be called once, got {len(calls)}"
            assert calls[0] == [], (
                f"tracker.update should be called with empty detections list, got {calls[0]}"
            )

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_tracklet_ages_out_after_empty_frames(self) -> None:
        """A tracklet must transition to closed state after close_grace_frames
        empty frames, confirming the aging mechanism works through empty frames."""

        detection_count = 0

        class DetectorWithCutoff:
            async def detect(self, image: np.ndarray) -> list[DetectionBox]:
                nonlocal detection_count
                detection_count += 1
                if detection_count <= 5:
                    return [
                        DetectionBox(
                            x1=0.3,
                            y1=0.2,
                            x2=0.5,
                            y2=0.6,
                            confidence=0.95,
                        )
                    ]
                return []

        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )

        with _mock_redis_deps():
            await pipeline.initialize(
                PipelineDependencies(
                    detector=DetectorWithCutoff(),  # type: ignore[arg-type]
                    frame_fetcher=FakeFetcher(),
                    reid_embedder=FakeReid(),
                )
            )

            # After 5 frames with a detection, a confirmed tracklet exists.
            # close_grace_frames defaults to 15; send 18 more empty frames
            # so lost_count exceeds the threshold.
            for i in range(23):  # 5 detection + 18 empty = 23 total
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

            assert pipeline._tracklet_manager is not None
            active = pipeline._tracklet_manager.get_active_tracklets()
            assert len(active) == 0, (
                f"Expected 0 active tracklets after {23} frames "
                f"(5 with detection + 18 empty > close_grace_frames=15), "
                f"got {len(active)}"
            )

            await pipeline.stop()
