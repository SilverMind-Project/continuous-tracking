"""Tests for pipeline-level fixes: shared in-memory repos, bbox_repo reuse."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
)
from app.pipeline.gallery_cache import GalleryCache
from app.storage.base import (
    InMemoryBboxAnnotationRepository,
    InMemoryTrajectoryRepository,
)
from app.storage.gallery import InMemoryGalleryRepository
from app.trajectory.dementia_signals import SignalConfig
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

    @pytest.mark.asyncio
    async def test_pipeline_passes_signal_config_directly_to_worker(self) -> None:
        signal_config = SignalConfig(window_hours=12, max_concurrent_identities=2)
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals=signal_config)
        )

        with _mock_redis_deps():
            await pipeline.initialize()

            assert pipeline._signal_worker is not None
            assert pipeline._signal_worker._cfg is signal_config

            await pipeline.stop()


class TestKeyframeBboxRepoReuse:
    @pytest.mark.asyncio
    async def test_default_keyframe_repo_uses_injected_bbox_repo(self) -> None:
        """Default in-memory keyframe storage writes bboxes to the injected bbox repo."""
        pipeline = FrameProcessingPipeline(
            PipelineConfig(allow_skeleton=True, signals_enabled=False)
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
            PipelineConfig(allow_skeleton=True, signals_enabled=False)
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
        pipeline = FrameProcessingPipeline(PipelineConfig(signals_enabled=False))
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


class _CountingGalleryRepo(InMemoryGalleryRepository):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> list:
        self.call_count += 1
        return await super().list_gallery_entries_for_tracklets(tracklet_ids, limit, allowed_states, model_versions)


class TestGalleryCacheLifecycle:
    @pytest.mark.asyncio
    async def test_cross_camera_path_invalidates_cache_each_round(self) -> None:
        """The gallery cache must be cleared once per world-tracking round on the
        cross-camera batched path so round-2 queries cannot hit round-1 cached data."""
        pipeline = FrameProcessingPipeline(PipelineConfig(signals_enabled=False))
        pipeline._transport = AsyncMock()

        repo = _CountingGalleryRepo()
        # large max_age_s so the staleness backstop never fires during the test
        cache = GalleryCache(repo, max_age_s=999.0)
        cache.invalidate()
        pipeline._gallery_cache = cache  # type: ignore[assignment]

        class FakeReadingRunner:
            def __init__(self) -> None:
                self.seen: list[str] = []

            async def run(self, ctx: object) -> None:
                import typing

                ctx = typing.cast(object, ctx)
                # Simulate a stage that reads from the gallery cache.
                await pipeline._gallery_cache.list_gallery_entries_for_tracklets(  # type: ignore[union-attr]
                    {"tracklet-shared"}
                )
                self.seen.append(f"{ctx.frame.camera_id}:{ctx.frame.frame_index}")  # type: ignore[attr-defined]

        class FakeWorldStage:
            async def run_many(self, contexts: object) -> None:
                pass

        pre_runner = FakeReadingRunner()
        post_runner = FakeReadingRunner()
        world_stage = FakeWorldStage()
        pipeline._pre_world_runner = pre_runner  # type: ignore[assignment]
        pipeline._post_world_runner = post_runner  # type: ignore[assignment]
        pipeline._world_tracking_stage = world_stage  # type: ignore[assignment]

        # Round 1: cam-a:1 + cam-b:1; Round 2: cam-a:2
        frames = [
            FrameReady(camera_id="cam-a", frame_index=1, minio_key="a1.jpg"),
            FrameReady(camera_id="cam-b", frame_index=1, minio_key="b1.jpg"),
            FrameReady(camera_id="cam-a", frame_index=2, minio_key="a2.jpg"),
        ]
        contexts = [pipeline._init_context(frame) for frame in frames]

        await pipeline._process_cross_camera_post_detect_batch(contexts)

        # Round 1: cam-a:1 misses cache -> repo call 1; cam-b:1 hits cache (same key)
        # Round 2: cache cleared by _begin_tracker_round -> cam-a:2 misses cache -> repo call 2
        assert repo.call_count == 2

    @pytest.mark.asyncio
    async def test_non_batched_path_invalidates_cache_each_frame(self) -> None:
        """The gallery cache must be cleared on each _process_frame call so
        gallery reads in frame N cannot return frame N-1 cached data."""
        pipeline = FrameProcessingPipeline(PipelineConfig(signals_enabled=False))
        pipeline._transport = AsyncMock()
        pipeline._detector = object()  # non-None so _process_frame runs _stage_runner

        repo = _CountingGalleryRepo()
        cache = GalleryCache(repo, max_age_s=999.0)
        cache.invalidate()
        pipeline._gallery_cache = cache  # type: ignore[assignment]

        class FakeStageRunner:
            async def run(self, ctx: object) -> None:
                await pipeline._gallery_cache.list_gallery_entries_for_tracklets(  # type: ignore[union-attr]
                    {"tracklet-shared"}
                )

        pipeline._stage_runner = FakeStageRunner()  # type: ignore[assignment]

        frame = FrameReady(camera_id="cam-a", frame_index=1, minio_key="a1.jpg")
        await pipeline._process_frame(frame)
        await pipeline._process_frame(frame)

        # Frame 1: _begin_tracker_round clears cache, stage misses -> repo call 1
        # Frame 2: _begin_tracker_round clears cache, stage misses -> repo call 2
        assert repo.call_count == 2
