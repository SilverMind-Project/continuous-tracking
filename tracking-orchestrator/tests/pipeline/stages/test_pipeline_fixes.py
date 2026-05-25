"""Tests for pipeline-level fixes: shared in-memory repos, bbox_repo reuse."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
    SignalConfig,
)
from app.storage.base import (
    InMemoryBboxAnnotationRepository,
    InMemoryTrajectoryRepository,
)


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


class TestInMemoryTrajectoryRepoSharing:
    @pytest.mark.asyncio
    async def test_pipeline_uses_shared_inmemory_trajectory_repo_for_signal_worker(self) -> None:
        """When no trajectory_repo is injected, the signal worker must use the
        same InMemoryTrajectoryRepository instance as the trajectory writer."""
        pipeline = FrameProcessingPipeline(PipelineConfig(allow_skeleton=True))

        with _mock_redis_deps():
            await pipeline.initialize()

            writer_repo = pipeline._trajectory_writer._repo  # type: ignore[union-attr]
            signal_repo = pipeline._signal_worker._trajectory_repo  # type: ignore[union-attr]

            assert writer_repo is signal_repo, (
                "TrajectoryWriter and DementiaSignalWorker must share "
                "the same InMemoryTrajectoryRepository instance"
            )

            assert isinstance(writer_repo, InMemoryTrajectoryRepository), (
                "Fallback should be InMemoryTrajectoryRepository"
            )

            await pipeline.stop()


class TestKeyframeBboxRepoReuse:
    @pytest.mark.asyncio
    async def test_keyframe_sampler_reuses_injected_bbox_repo(self) -> None:
        """When a bbox_repo is injected into initialize(), the keyframe sampler
        must use that same instance rather than creating a new fallback."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )

        injected_bbox_repo = InMemoryBboxAnnotationRepository()

        with _mock_redis_deps():
            await pipeline.initialize(PipelineDependencies(bbox_repo=injected_bbox_repo))

            sampler_repo = pipeline._keyframe_sampler._bbox_repo  # type: ignore[union-attr]

            assert sampler_repo is injected_bbox_repo, (
                "KeyframeSampler must use the injected bbox_repo, not create a separate fallback"
            )

            # Also verify the pipeline's own bbox_repo is the same instance.
            assert pipeline._bbox_repo is injected_bbox_repo  # type: ignore[union-attr]

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_keyframe_sampler_reuses_pipeline_bbox_repo_default(self) -> None:
        """When no bbox_repo is injected, the keyframe sampler must still use
        the same instance as the pipeline's internally created fallback."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )

        with _mock_redis_deps():
            await pipeline.initialize()

            sampler_repo = pipeline._keyframe_sampler._bbox_repo  # type: ignore[union-attr]
            pipeline_repo = pipeline._bbox_repo

            assert sampler_repo is pipeline_repo, (
                "KeyframeSampler must share the pipeline's bbox_repo fallback"
            )

            assert isinstance(sampler_repo, InMemoryBboxAnnotationRepository)

            await pipeline.stop()
