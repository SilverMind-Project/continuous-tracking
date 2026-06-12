"""tests for PublishStage building identity map from world_snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from app.domain import (
    IdentityDecision,
    PosteriorDist,
    WorldFrameSnapshot,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.publish import PublishStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap


def _live_config() -> LiveConfigHolder:
    return LiveConfigHolder(CameraRoomMap(), RoomPolygonMap())


def _make_snapshot(
    *,
    ph_id: str = "ph-1",
    identity_id: str | None = None,
    identity_confidence: float = 0.0,
    direct_face_evidence: bool = False,
    captured_at: datetime | None = None,
) -> WorldFrameSnapshot:
    if captured_at is None:
        captured_at = datetime.now(UTC)
    return WorldFrameSnapshot(
        ph_id=ph_id,
        camera_id="cam1",
        frame_index=1,
        captured_at=captured_at,
        floor_x_m=0.0,
        floor_y_m=0.0,
        floor_vx_m_s=0.0,
        floor_vy_m_s=0.0,
        position_sigma_m=0.1,
        identity_id=identity_id,
        identity_confidence=identity_confidence,
        direct_face_evidence=direct_face_evidence,
        detection_confidence=0.9,
    )


def _make_ctx(
    *,
    snapshots: list[WorldFrameSnapshot] | None = None,
    decisions: list[IdentityDecision] | None = None,
    camera_id: str = "cam1",
) -> FrameContext:
    frame = MagicMock()
    frame.camera_id = camera_id
    frame.minio_key = "minio/test.jpg"
    frame.frame_index = 1
    frame.capture_time_unix_ns = 0
    ctx = FrameContext(
        frame=frame,  # type: ignore[arg-type]
        event_time=datetime.now(UTC),
        capture_time=datetime.now(UTC),
    )
    ctx.world_snapshots = snapshots or []
    ctx.outcome_decisions = decisions or []
    ctx.active_ph_ids = set()
    ctx.domain_detections = []
    ctx.effective_width = 1920
    ctx.effective_height = 1080
    ctx.det_pose_result = {}
    ctx.trail_by_tracklet_snapshot = {}
    ctx.det_posture = {}
    ctx.raw_detections = []
    return ctx


class TestPublishIdentityFromSnapshots:
    async def test_publish_identity_map_built_from_snapshots(self) -> None:
        """Snapshot with identity_id → ctx.identities populated."""
        snap = _make_snapshot(
            ph_id="ph-1",
            identity_id="alice",
            identity_confidence=0.91,
            direct_face_evidence=True,
        )
        ctx = _make_ctx(snapshots=[snap])

        transport = AsyncMock()
        stage = PublishStage(transport=transport, live_config=_live_config())
        await stage.run(ctx)

        assert ctx.identities == {"ph-1": ("alice", 0.91)}
        assert ctx.evidence_by_ph["ph-1"] == (0.91, 0.0, True)

    async def test_publish_skips_unknown_identity(self) -> None:
        """Snapshot with identity_id='UNKNOWN' → not included."""
        snap_unknown = _make_snapshot(ph_id="ph-1", identity_id="UNKNOWN")
        snap_none = _make_snapshot(ph_id="ph-2", identity_id=None)
        ctx = _make_ctx(snapshots=[snap_unknown, snap_none])

        transport = AsyncMock()
        stage = PublishStage(transport=transport, live_config=_live_config())
        await stage.run(ctx)

        assert ctx.identities == {}

    async def test_publish_decision_top2_augments_snapshot_evidence(self) -> None:
        """Outcome decision with second probability augments evidence_by_ph."""
        snap = _make_snapshot(ph_id="ph-1", identity_id="alice", identity_confidence=0.91)
        decision = IdentityDecision(
            ph_id="ph-1",
            identity_id="alice",
            posterior=PosteriorDist({"alice": 0.8, "bob": 0.15, "UNKNOWN": 0.05}),
            revises_previous=False,
            reason="test",
            evidence_backed=True,
        )
        ctx = _make_ctx(snapshots=[snap], decisions=[decision])

        transport = AsyncMock()
        stage = PublishStage(transport=transport, live_config=_live_config())
        await stage.run(ctx)

        assert ctx.identities["ph-1"] == ("alice", 0.8)
        # second_probability from decision: 0.15
        assert ctx.evidence_by_ph["ph-1"][1] == 0.15
