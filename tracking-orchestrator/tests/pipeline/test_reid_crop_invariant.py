"""Invariant 1: ReID embeddings must be computed from YOLO-bbox crops,
not full frames. Gallery distances collapse to noise otherwise."""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.inference.schemas import DetectionBox
from app.pipeline.frame_pipeline import FrameProcessingPipeline, PipelineConfig
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


class FakeDetector:
    async def detect(self, image: np.ndarray) -> list[DetectionBox]:
        # One person at (100, 100, 200, 200) in pixel coords → normalised.
        return [
            DetectionBox(
                x1=100 / 640,
                y1=100 / 480,
                x2=200 / 640,
                y2=200 / 480,
                confidence=0.95,
            )
        ]


class TestReidCropInvariant:
    @pytest.mark.asyncio
    async def test_embed_batch_receives_crop_not_full_frame(self) -> None:
        captured: list[np.ndarray] = []

        async def fake_embed(crops: list[np.ndarray]) -> list[np.ndarray]:
            captured.extend(crops)
            return [np.zeros(768, dtype=np.float32) for _ in crops]

        class FakeReid:
            async def embed_batch(
                self, crops: list[np.ndarray]
            ) -> list[np.ndarray]:
                return await fake_embed(crops)

        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signal_enabled=False)
        )

        with _mock_redis_deps():
            await pipeline.initialize(
                detector=FakeDetector(),  # type: ignore[arg-type]
                frame_fetcher=FakeFetcher(),
                reid_embedder=FakeReid(),
            )

            frame = FrameReady(
                camera_id="cam-1",
                minio_key="frames/cam-1/1.jpg",
                frame_index=1,
                capture_time_unix_ns=_NOW_NS,
                received_time_unix_ns=_NOW_NS + 100_000_000,
                width=640,
                height=480,
            )
            await pipeline._process_frame(frame)

            assert len(captured) == 1
            assert captured[0].shape == (100, 100, 3), (
                f"Expected (100, 100, 3) crop, got {captured[0].shape}"
            )
            assert captured[0].shape != (480, 640, 3), (
                "embed_batch received full frame, not a bbox crop"
            )

            await pipeline.stop()
