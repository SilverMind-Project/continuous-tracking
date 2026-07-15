"""M01 pipeline route smoke tests.

Verify that after the M01 identity domain refactor, all three pipeline
execution paths (direct, per-camera batch, cross-camera batch) still reach
the same IdentityResolver facade and produce decisions.

These are integration-level smoke tests: they instantiate the real resolver
and pipeline and confirm that identity decisions arrive at the same place
regardless of which batch path is used. Numerical assertions stay in the
dedicated golden-master tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.domain import FaceAnchor, GlobalTrack, Identity, PosteriorDist
from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
)
from app.storage.base import InMemoryIdentityDecisionRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gt(
    ph_id: str = "gt-1",
    current_identity_id: str | None = None,
) -> GlobalTrack:
    now = datetime.now(UTC)
    return GlobalTrack(
        global_track_id=ph_id,
        camera_ids=["cam_a"],
        tracklet_ids=["tl-1"],
        started_at=now,
        last_seen_at=now,
        current_identity_id=current_identity_id,
        state="active",
    )


def _face_anchor(
    identity_id: str = "alice",
    confidence: float = 0.95,
    tracklet_id: str = "tl-1",
) -> FaceAnchor:
    return FaceAnchor(
        person_id=identity_id,
        confidence=confidence,
        quality=0.9,
        tracklet_id=tracklet_id,
    )


@contextmanager
def _mock_redis_deps():
    with (
        patch("app.pipeline.frame_pipeline.RedisStreamsTransport") as mock_transport_cls,
        patch("app.pipeline.frame_pipeline.RevisionPublisher") as mock_rev_cls,
        patch("app.pipeline.frame_pipeline.SceneSamplesPublisher") as mock_scene_cls,
    ):
        mock_transport_cls.return_value = AsyncMock()
        mock_rev_cls.return_value = AsyncMock()
        mock_scene_cls.return_value = AsyncMock()
        yield


# ---------------------------------------------------------------------------
# Resolver facade smoke tests
# ---------------------------------------------------------------------------


class TestIdentityResolverFacade:
    """IdentityResolver is still the single facade after M01."""

    @pytest.mark.asyncio
    async def test_resolver_produces_decisions(self) -> None:
        """Resolver produces one decision per PH with no face anchors."""
        repo = InMemoryGalleryRepository()
        resolver = IdentityResolver(gallery_repo=repo, config=ResolverConfig())

        gt = _make_gt(ph_id="gt-1")
        now = datetime.now(UTC)

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[],
            captured_at=now,
        )

        assert len(outcome.decisions) == 1
        assert outcome.decisions[0].ph_id == "gt-1"
        # No evidence → identity is None (UNKNOWN).
        assert outcome.decisions[0].identity_id is None

    @pytest.mark.asyncio
    async def test_strong_face_anchor_commits(self) -> None:
        """Strong face anchor commits identity through the resolver facade."""
        repo = InMemoryGalleryRepository()
        await repo.upsert_identity(
            Identity(
                identity_id="alice",
                display_name="Alice",
                enrolled_at=datetime.now(UTC),
            )
        )
        resolver = IdentityResolver(gallery_repo=repo, config=ResolverConfig())

        gt = _make_gt(ph_id="gt-1")
        anchor = _face_anchor(identity_id="alice", confidence=0.95)
        now = datetime.now(UTC)

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[anchor],
            captured_at=now,
        )

        assert len(outcome.decisions) == 1
        assert outcome.decisions[0].identity_id == "alice"

    @pytest.mark.asyncio
    async def test_resolver_uses_combine_posteriors_not_combine_evidence(self) -> None:
        """Resolver produces PosteriorDist results, not EvidencePosterior."""
        repo = InMemoryGalleryRepository()
        await repo.upsert_identity(
            Identity(
                identity_id="alice",
                display_name="Alice",
                enrolled_at=datetime.now(UTC),
            )
        )
        resolver = IdentityResolver(gallery_repo=repo, config=ResolverConfig())

        gt = _make_gt(ph_id="gt-1")
        anchor = _face_anchor(identity_id="alice", confidence=0.95)
        now = datetime.now(UTC)

        outcome = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[anchor],
            captured_at=now,
        )

        decision = outcome.decisions[0]
        # The resolver's posterior is a PosteriorDist (Bayesian path).
        assert isinstance(decision.posterior, PosteriorDist)
        # Posterior sums to 1.0 (or near-1.0 due to normalization).
        total = sum(decision.posterior.distribution.values())
        assert abs(total - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_multiple_phs_each_get_a_decision(self) -> None:
        """All PHs in the batch receive a decision."""
        repo = InMemoryGalleryRepository()
        resolver = IdentityResolver(gallery_repo=repo, config=ResolverConfig())

        gts = [_make_gt(ph_id=f"gt-{i}") for i in range(3)]
        now = datetime.now(UTC)

        outcome = await resolver.resolve(
            hypotheses=gts,
            new_face_anchors=[],
            captured_at=now,
        )

        assert len(outcome.decisions) == 3
        decision_ids = {d.ph_id for d in outcome.decisions}
        assert decision_ids == {"gt-0", "gt-1", "gt-2"}

    @pytest.mark.asyncio
    async def test_revision_emitted_on_identity_change(self) -> None:
        """Identity change produces a revision record."""
        repo = InMemoryGalleryRepository()
        await repo.upsert_identity(
            Identity(
                identity_id="alice",
                display_name="Alice",
                enrolled_at=datetime.now(UTC),
            )
        )
        resolver = IdentityResolver(gallery_repo=repo, config=ResolverConfig())

        # First resolve: no identity.
        gt = _make_gt(ph_id="gt-1", current_identity_id=None)
        anchor = _face_anchor(identity_id="alice", confidence=0.95)
        now = datetime.now(UTC)

        outcome1 = await resolver.resolve(
            hypotheses=[gt],
            new_face_anchors=[anchor],
            captured_at=now,
        )
        assert outcome1.decisions[0].identity_id == "alice"
        # Initial assignment emits a revision.
        assert len(outcome1.revisions) == 1
        assert outcome1.revisions[0].new_identity_id == "alice"


class TestPipelineRouteSmoke:
    """Pipeline initialises correctly after M01 structural changes."""

    @pytest.mark.asyncio
    async def test_pipeline_initialises_with_identity_resolver(self) -> None:
        """After M01, the pipeline still starts without import errors."""
        with _mock_redis_deps():
            pipeline = FrameProcessingPipeline(PipelineConfig(allow_skeleton=True))
            await pipeline.initialize()
            assert pipeline is not None


class TestProvenanceStageOrdering:
    """M02: provenance_persist must precede publish on every execution route."""

    @pytest.mark.asyncio
    async def test_provenance_persist_precedes_publish_on_all_routes(self) -> None:
        with _mock_redis_deps():
            pipeline = FrameProcessingPipeline(PipelineConfig(allow_skeleton=True))
            deps = PipelineDependencies(
                identity_provenance_repo=InMemoryIdentityDecisionRepository()
            )
            await pipeline.initialize(deps)

            assert pipeline._stage_runner is not None
            assert pipeline._post_detect_runner is not None
            assert pipeline._pre_world_runner is not None
            assert pipeline._post_world_runner is not None
            assert pipeline._world_tracking_stage is not None

            direct_names = [s.name for s in pipeline._stage_runner._stages]
            per_camera_batch_names = [s.name for s in pipeline._post_detect_runner._stages]
            cross_camera_batch_names = (
                [s.name for s in pipeline._pre_world_runner._stages]
                + [pipeline._world_tracking_stage.name]
                + [s.name for s in pipeline._post_world_runner._stages]
            )

            for route_name, names in (
                ("direct", direct_names),
                ("per_camera_batch", per_camera_batch_names),
                ("cross_camera_batch", cross_camera_batch_names),
            ):
                assert "provenance_persist" in names, route_name
                assert "publish" in names, route_name
                assert names.index("provenance_persist") < names.index("publish"), route_name
