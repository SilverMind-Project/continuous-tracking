"""Acceptance tests for skeleton-mode hard-fail (Phase 12 / B-7 / B-12).

``PipelineConfig.allow_skeleton=False`` (the default) must cause
``initialize()`` to raise ``RuntimeError`` when no detector is provided.
``allow_skeleton=True`` must allow ``initialize()`` to succeed without one.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
)


@contextmanager
def _mock_redis_deps() -> Generator[None, None, None]:
    """Suppress Redis connections so tests run without infrastructure."""
    with (
        patch("app.pipeline.frame_pipeline.RedisStreamsTransport") as mock_transport_cls,
        patch("app.pipeline.frame_pipeline.RevisionPublisher") as mock_rev_cls,
        patch("app.pipeline.frame_pipeline.SceneSamplesPublisher") as mock_scene_cls,
    ):
        mock_transport_cls.return_value = AsyncMock()
        mock_rev_cls.return_value = AsyncMock()
        mock_scene_cls.return_value = AsyncMock()
        yield


def _config(**kwargs: object) -> PipelineConfig:
    return PipelineConfig(
        transport=type(
            "TransportConfig",
            (),
            {
                "redis_url": "redis://localhost:6379/0",
                "consumer_group": "cts-orchestrator",
                "consumer_name": "test-1",
                "frames_stream": "frames.ready",
                "events_stream": "tracking.events",
                "revisions_stream": "tracking.revisions",
                "signals_stream": "tracking.signals",
                "scene_samples_stream": "scene.samples",
                "presence_stream": "tracking.presence",
                "dwell_stream": "tracking.dwell",
                "maxlen": 1000,
            },
        )(),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_initialize_without_detector_and_allow_skeleton_false_raises() -> None:
    """Default config (allow_skeleton=False) must raise when detector is None."""
    pipeline = FrameProcessingPipeline(_config())
    with _mock_redis_deps(), pytest.raises(RuntimeError, match="allow_skeleton"):
        await pipeline.initialize(PipelineDependencies(detector=None))


@pytest.mark.asyncio
async def test_initialize_without_detector_and_allow_skeleton_true_succeeds() -> None:
    """allow_skeleton=True must allow initialization without a detector."""
    pipeline = FrameProcessingPipeline(_config(allow_skeleton=True))
    with _mock_redis_deps():
        await pipeline.initialize(PipelineDependencies(detector=None))
    await pipeline.stop()
