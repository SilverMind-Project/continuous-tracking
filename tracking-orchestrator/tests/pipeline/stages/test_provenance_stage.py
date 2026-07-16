"""ProvenancePersistStage: durable identity-decision writes decoupled from the
publish throttle.

Regression coverage for M02 (F2): a `revises_previous` decision produced on a
throttled frame must still be persisted. Folds the former
`test_publish_provenance_persistence.py` cases (M04 write path) since
persistence now lives in this stage, not `PublishStage`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from app.domain import IdentityDecision, IdentityProvenanceDecision, PosteriorDist
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.provenance import ProvenancePersistStage
from app.pipeline.stages.publish import PublishStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.storage.base import InMemoryIdentityDecisionRepository
from app.transport.redis_streams import FrameReady

_T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


def _ctx(decisions: list[IdentityDecision], *, camera_id: str = "cam1") -> FrameContext:
    frame = MagicMock()
    frame.camera_id = camera_id
    frame.minio_key = "minio/test.jpg"
    frame.frame_index = 1
    frame.capture_time_unix_ns = 0
    frame.captured_at = _T0
    ctx = FrameContext(frame=frame, event_time=_T0, capture_time=_T0)  # type: ignore[arg-type]
    ctx.world_snapshots = []
    ctx.outcome_decisions = decisions
    ctx.active_ph_ids = set()
    ctx.domain_detections = []
    ctx.effective_width = 1920
    ctx.effective_height = 1080
    ctx.det_pose_result = {}
    ctx.trail_by_tracklet_snapshot = {}
    ctx.det_posture = {}
    ctx.raw_detections = []
    return ctx


def _decision(
    *, ph_id: str = "00000000-0000-0000-0000-000000000111", revises: bool
) -> IdentityDecision:
    return IdentityDecision(
        ph_id=ph_id,
        identity_id="amma",
        posterior=PosteriorDist({"amma": 0.8, "bob": 0.15, "UNKNOWN": 0.05}),
        revises_previous=revises,
        reason="arcface_authority: amma",
        evidence_backed=True,
        decision_id=f"{ph_id}-decision",
        inferred_identity_id="amma",
        effective_identity_id="amma",
        authority="direct_face",
        decision_source="face",
    )


def _live_config() -> LiveConfigHolder:
    return LiveConfigHolder(CameraRoomMap(), RoomPolygonMap())


async def _run(repo: InMemoryIdentityDecisionRepository, ctx: FrameContext) -> None:
    stage = ProvenancePersistStage(identity_provenance_repo=repo)
    await stage.run(ctx)
    # The decision is saved via asyncio.create_task; let it run.
    await asyncio.sleep(0)


async def test_revising_decision_is_persisted() -> None:
    repo = InMemoryIdentityDecisionRepository()
    await _run(repo, _ctx([_decision(revises=True)]))

    saved, total = await repo.get_by_ph_id("00000000-0000-0000-0000-000000000111")
    assert total == 1
    prov = saved[0]
    assert prov.inferred_identity_id == "amma"
    assert prov.authority == "direct_face"
    assert prov.decision_source == "face"
    assert prov.captured_at == _T0


async def test_held_decision_is_not_persisted() -> None:
    repo = InMemoryIdentityDecisionRepository()
    await _run(repo, _ctx([_decision(revises=False)]))

    _saved, total = await repo.get_by_ph_id("00000000-0000-0000-0000-000000000111")
    assert total == 0


async def test_empty_outcome_decisions_is_noop() -> None:
    """No decisions in the frame → no repo call, no error."""

    class _ExplodingRepo:
        async def save(self, decision: IdentityProvenanceDecision) -> None:
            raise AssertionError("save() must not be called when outcome_decisions is empty")

    stage = ProvenancePersistStage(identity_provenance_repo=_ExplodingRepo())  # type: ignore[arg-type]
    await stage.run(_ctx([]))
    await asyncio.sleep(0)


async def test_aclose_awaits_pending_saves() -> None:
    """aclose() blocks until in-flight saves complete, not just until scheduled."""
    gate = asyncio.Event()
    saved: list[IdentityProvenanceDecision] = []

    class _GatedRepo:
        async def save(self, decision: IdentityProvenanceDecision) -> None:
            await gate.wait()
            saved.append(decision)

    stage = ProvenancePersistStage(identity_provenance_repo=_GatedRepo())  # type: ignore[arg-type]
    await stage.run(_ctx([_decision(revises=True)]))

    aclose_task = asyncio.create_task(stage.aclose())
    await asyncio.sleep(0)
    assert not aclose_task.done()
    assert saved == []

    gate.set()
    await aclose_task
    assert len(saved) == 1


async def test_throttled_frame_still_persists_provenance() -> None:
    """Headline regression: a revises_previous decision on a throttled frame
    is still written to the provenance repo, even though the UI-facing
    publish is dropped by the per-camera throttle.
    """
    repo = InMemoryIdentityDecisionRepository()
    provenance_stage = ProvenancePersistStage(identity_provenance_repo=repo)
    transport = MagicMock()
    transport.publish_event = AsyncMock()
    publish_stage = PublishStage(
        transport=transport, live_config=_live_config(), live_publish_max_hz=3.0
    )

    def _frame_ctx(frame_index: int, ph_id: str) -> FrameContext:
        frame = FrameReady(
            camera_id="cam-1",
            minio_key="test/key",
            width=640,
            height=480,
            frame_index=frame_index,
            capture_time_unix_ns=int(datetime.now(UTC).timestamp() * 1e9),
        )
        frame_ctx = FrameContext(
            frame=frame, event_time=datetime.now(UTC), capture_time=datetime.now(UTC)
        )
        frame_ctx.world_snapshots = []
        frame_ctx.active_ph_ids = set()
        frame_ctx.raw_detections = []
        frame_ctx.domain_detections = []
        frame_ctx.outcome_decisions = [_decision(ph_id=ph_id, revises=True)]
        return frame_ctx

    # Frame 1: published (first frame for this camera), decision persisted.
    ctx1 = _frame_ctx(1, "ph-1")
    await provenance_stage.run(ctx1)
    await publish_stage.run(ctx1)

    # Frame 2: within the throttle interval — publish is dropped, but the
    # revising decision must still reach the provenance repo.
    ctx2 = _frame_ctx(2, "ph-2")
    await provenance_stage.run(ctx2)
    await publish_stage.run(ctx2)

    await asyncio.sleep(0)

    assert transport.publish_event.call_count == 1

    _saved1, total1 = await repo.get_by_ph_id("ph-1")
    _saved2, total2 = await repo.get_by_ph_id("ph-2")
    assert total1 == 1
    assert total2 == 1
