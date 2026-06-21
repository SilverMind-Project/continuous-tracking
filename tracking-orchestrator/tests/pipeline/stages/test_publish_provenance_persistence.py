"""M04 write path: PublishStage persists identity provenance decisions.

Regression coverage for the gap where ``PublishStage`` called a non-existent
``save_decision`` method and mutated a frozen dataclass, so no decision was
ever persisted. Decisions are persisted at identity change points
(``revises_previous``); held rounds are skipped.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock

from app.domain import IdentityDecision, PosteriorDist
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.publish import PublishStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.storage.base import InMemoryIdentityDecisionRepository

_T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


def _ctx(decisions: list[IdentityDecision]) -> FrameContext:
    frame = MagicMock()
    frame.camera_id = "cam1"
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


def _decision(*, revises: bool) -> IdentityDecision:
    return IdentityDecision(
        ph_id="00000000-0000-0000-0000-000000000111",
        identity_id="amma",
        posterior=PosteriorDist({"amma": 0.8, "bob": 0.15, "UNKNOWN": 0.05}),
        revises_previous=revises,
        reason="arcface_authority: amma",
        evidence_backed=True,
        decision_id="00000000-0000-0000-0000-0000000005d1",
        inferred_identity_id="amma",
        effective_identity_id="amma",
        authority="arcface_authority",
        decision_source="face",
    )


async def _run(repo: InMemoryIdentityDecisionRepository, ctx: FrameContext) -> None:
    transport = MagicMock()

    async def _noop(*args, **kwargs):
        return None

    transport.publish_event = _noop
    stage = PublishStage(
        transport=transport,
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
        identity_provenance_repo=repo,
    )
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
    assert prov.authority == "arcface_authority"
    assert prov.decision_source == "face"
    assert prov.captured_at == _T0


async def test_held_decision_is_not_persisted() -> None:
    repo = InMemoryIdentityDecisionRepository()
    await _run(repo, _ctx([_decision(revises=False)]))

    _saved, total = await repo.get_by_ph_id("00000000-0000-0000-0000-000000000111")
    assert total == 0
