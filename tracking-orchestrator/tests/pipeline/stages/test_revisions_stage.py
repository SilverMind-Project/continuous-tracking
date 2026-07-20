"""Tests for RevisionsStage, including the M04 backfill-service wiring.

Covers: (1) RevisionsStage.run invokes the backfill service with the expected
frame fields when outcome_decisions are present, and does nothing when the
service is absent; (2) the same RevisionsStage instance (and therefore the
same wired backfill_service) is reachable from every execution-path runner
the pipeline builds, per the cts-pipeline skill's execution-path rule.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.domain import IdentityDecision, PosteriorDist
from app.pipeline.frame_context import FrameContext
from app.pipeline.frame_pipeline import (
    FrameProcessingPipeline,
    PipelineConfig,
    PipelineDependencies,
)
from app.pipeline.stages.revisions import RevisionsStage
from app.services.identity_correction_service import IdentityCorrectionService
from app.services.unknown_backfill import UnknownBackfillService
from app.storage.base import InMemoryIdentityDecisionRepository, InMemoryPHRepository
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from app.transport.redis_streams import FrameReady

T0 = datetime(2026, 7, 20, 8, 0, 0, tzinfo=UTC)


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


def _frame_ready() -> FrameReady:
    return FrameReady(
        camera_id="cam-1",
        frame_index=0,
        minio_key="k",
        capture_time_unix_ns=0,
    )


def _ctx(outcome_decisions: list[IdentityDecision]) -> FrameContext:
    ctx = FrameContext(frame=_frame_ready(), capture_time=T0, event_time=T0)
    ctx.outcome_decisions = outcome_decisions
    ctx.ph_born_at_by_id = {"ph-1": T0}
    return ctx


# ---------------------------------------------------------------------------
# Unit-level: RevisionsStage.run invokes the backfill service
# ---------------------------------------------------------------------------


class _SpyBackfillService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def process(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_revisions_stage_invokes_backfill_service_when_present() -> None:
    spy = _SpyBackfillService()
    stage = RevisionsStage(backfill_service=spy)  # type: ignore[arg-type]
    decision = IdentityDecision(
        ph_id="ph-1",
        identity_id="alice",
        posterior=PosteriorDist(distribution={"alice": 1.0}),
        revises_previous=True,
        previous_identity_id=None,
        authority="direct_face",
    )

    await stage.run(_ctx([decision]))

    assert len(spy.calls) == 1
    assert spy.calls[0]["outcome_decisions"] == [decision]
    assert spy.calls[0]["event_time"] == T0


@pytest.mark.asyncio
async def test_revisions_stage_skips_backfill_when_no_decisions() -> None:
    spy = _SpyBackfillService()
    stage = RevisionsStage(backfill_service=spy)  # type: ignore[arg-type]

    await stage.run(_ctx([]))

    assert spy.calls == []


@pytest.mark.asyncio
async def test_revisions_stage_tolerates_absent_backfill_service() -> None:
    stage = RevisionsStage(backfill_service=None)
    decision = IdentityDecision(
        ph_id="ph-1",
        identity_id="alice",
        posterior=PosteriorDist(distribution={"alice": 1.0}),
        revises_previous=True,
        previous_identity_id=None,
        authority="direct_face",
    )

    await stage.run(_ctx([decision]))  # must not raise


# ---------------------------------------------------------------------------
# Wiring: the same RevisionsStage instance (with backfill_service set) is
# reachable from every execution-path runner the pipeline builds.
# ---------------------------------------------------------------------------


def _find_revisions_stage(stages: list[object]) -> RevisionsStage | None:
    for s in stages:
        if isinstance(s, RevisionsStage):
            return s
    return None


@pytest.mark.asyncio
async def test_backfill_service_wired_into_revisions_stage_on_every_route() -> None:
    ph_repo = InMemoryPHRepository()
    correction_repo = InMemoryIdentityCorrectionRepository()
    correction_service = IdentityCorrectionService(
        ph_repo=ph_repo,
        correction_repo=correction_repo,
    )
    identity_decision_repo = InMemoryIdentityDecisionRepository()

    pipeline = FrameProcessingPipeline(PipelineConfig(allow_skeleton=True))
    deps = PipelineDependencies(
        ph_repo=ph_repo,
        identity_provenance_repo=identity_decision_repo,
        identity_correction_service=correction_service,
    )

    with _mock_redis_deps():
        await pipeline.initialize(deps)

    assert pipeline._backfill_service is not None
    assert isinstance(pipeline._backfill_service, UnknownBackfillService)

    stage_runner_stage = _find_revisions_stage(pipeline._stage_runner._stages)  # type: ignore[union-attr]
    post_world_stage = _find_revisions_stage(pipeline._post_world_runner._stages)  # type: ignore[union-attr]

    assert stage_runner_stage is not None
    assert post_world_stage is not None
    # Same object: proves the direct-path runner and the cross-camera-batch
    # post-world runner share one RevisionsStage, so wiring the backfill
    # service once (constructor injection) reaches every execution route.
    assert stage_runner_stage is post_world_stage
    assert stage_runner_stage._backfill_service is pipeline._backfill_service


@pytest.mark.asyncio
async def test_backfill_service_absent_when_dependencies_missing() -> None:
    """No identity_correction_service/identity_provenance_repo => no service.

    Matches dev/test wiring (in-memory-only, no Postgres pool) where these
    are optional; RevisionsStage must tolerate backfill_service=None.
    """
    pipeline = FrameProcessingPipeline(PipelineConfig(allow_skeleton=True))

    with _mock_redis_deps():
        await pipeline.initialize()

    assert pipeline._backfill_service is None
    stage = _find_revisions_stage(pipeline._stage_runner._stages)  # type: ignore[union-attr]
    assert stage is not None
    assert stage._backfill_service is None
