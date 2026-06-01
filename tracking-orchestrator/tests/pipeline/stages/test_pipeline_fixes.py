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
from app.transport.redis_streams import FrameReady


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
    async def test_default_keyframe_repo_uses_injected_bbox_repo(self) -> None:
        """Default in-memory keyframe storage writes bboxes to the injected bbox repo."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )

        injected_bbox_repo = InMemoryBboxAnnotationRepository()

        with _mock_redis_deps():
            await pipeline.initialize(PipelineDependencies(bbox_repo=injected_bbox_repo))

            keyframe_repo = pipeline._keyframe_sampler._repo  # type: ignore[union-attr]

            assert keyframe_repo._bbox_repo is injected_bbox_repo, (  # type: ignore[attr-defined]
                "InMemoryKeyframeRepository must write bboxes to the injected bbox repo"
            )

            # Also verify the pipeline's own bbox_repo is the same instance.
            assert pipeline._bbox_repo is injected_bbox_repo  # type: ignore[union-attr]

            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_default_keyframe_repo_uses_pipeline_bbox_repo_default(self) -> None:
        """Default in-memory keyframe storage writes bboxes to the pipeline fallback."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=SignalConfig(enabled=False))
        )

        with _mock_redis_deps():
            await pipeline.initialize()

            keyframe_repo = pipeline._keyframe_sampler._repo  # type: ignore[union-attr]
            pipeline_repo = pipeline._bbox_repo

            assert keyframe_repo._bbox_repo is pipeline_repo, (  # type: ignore[attr-defined]
                "InMemoryKeyframeRepository must write bboxes to the pipeline fallback"
            )

            assert isinstance(pipeline_repo, InMemoryBboxAnnotationRepository)

            await pipeline.stop()


class TestCrossCameraPostDetectBatch:
    @pytest.mark.asyncio
    async def test_world_tracking_receives_one_round_from_each_camera(self) -> None:
        """Detector batching must not split overlapping cameras before world tracking."""
        pipeline = FrameProcessingPipeline(PipelineConfig(signals=SignalConfig(enabled=False)))
        pipeline._transport = AsyncMock()

        class FakeRunner:
            def __init__(self) -> None:
                self.seen: list[str] = []

            async def run(self, ctx) -> None:
                self.seen.append(f"{ctx.frame.camera_id}:{ctx.frame.frame_index}")

        class FakeWorldStage:
            def __init__(self) -> None:
                self.rounds: list[list[str]] = []

            async def run_many(self, contexts) -> None:
                self.rounds.append(
                    [f"{ctx.frame.camera_id}:{ctx.frame.frame_index}" for ctx in contexts]
                )

        pre_runner = FakeRunner()
        post_runner = FakeRunner()
        world_stage = FakeWorldStage()
        pipeline._pre_world_runner = pre_runner  # type: ignore[assignment]
        pipeline._post_world_runner = post_runner  # type: ignore[assignment]
        pipeline._world_tracking_stage = world_stage  # type: ignore[assignment]

        frames = [
            FrameReady(camera_id="cam-a", frame_index=2, minio_key="a2.jpg"),
            FrameReady(camera_id="cam-b", frame_index=1, minio_key="b1.jpg"),
            FrameReady(camera_id="cam-a", frame_index=1, minio_key="a1.jpg"),
        ]
        contexts = [pipeline._init_context(frame) for frame in frames]

        await pipeline._process_cross_camera_post_detect_batch(contexts)

        assert world_stage.rounds == [["cam-a:1", "cam-b:1"], ["cam-a:2"]]
        assert pre_runner.seen == ["cam-a:1", "cam-b:1", "cam-a:2"]
        assert sorted(post_runner.seen) == ["cam-a:1", "cam-a:2", "cam-b:1"]
        assert pipeline._transport.ack_frame.await_count == 3  # type: ignore[union-attr]
