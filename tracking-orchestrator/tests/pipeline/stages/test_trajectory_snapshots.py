"""tests for snapshot-driven TrajectoryStage.

Verifies that TrajectoryStage consumes ctx.world_snapshots and writes one
trajectory point per current-camera snapshot per frame.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from app.domain import (
    BoundingBox,
    Detection,
    FloorPoint,
    ObservationGeometry,
    OrientationBin,
    WorldFrameSnapshot,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.trajectory import TrajectoryStage
from app.storage.trajectory import InMemoryTrajectoryRepository
from app.trajectory.motion_energy import MotionEnergyTracker
from app.trajectory.posture import GlobalPostureTracker, PostureScores
from app.trajectory.trajectory_writer import TrajectoryWriter


def _fake_frame_ready(*, camera_id: str = "cam1", frame_index: int = 1) -> MagicMock:
    return MagicMock(camera_id=camera_id, frame_index=frame_index)


def _make_snapshot(
    *,
    ph_id: str = "ph-1",
    camera_id: str = "cam1",
    frame_index: int = 1,
    floor_x_m: float = 1.5,
    floor_y_m: float = 2.3,
    identity_id: str | None = None,
    identity_confidence: float = 0.0,
    room_name: str = "living_room",
    bbox: BoundingBox | None = None,
    captured_at: datetime | None = None,
) -> WorldFrameSnapshot:
    if captured_at is None:
        captured_at = datetime.now(UTC)
    return WorldFrameSnapshot(
        ph_id=ph_id,
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=captured_at,
        floor_x_m=floor_x_m,
        floor_y_m=floor_y_m,
        floor_vx_m_s=0.0,
        floor_vy_m_s=0.0,
        position_sigma_m=0.1,
        identity_id=identity_id,
        identity_confidence=identity_confidence,
        room_name=room_name,
        bbox=bbox,
        detection_confidence=0.9,
    )


def _make_ctx(
    *,
    camera_id: str = "cam1",
    frame_index: int = 1,
    now: datetime | None = None,
    snapshots: list[WorldFrameSnapshot] | None = None,
) -> FrameContext:
    if now is None:
        now = datetime.now(UTC)
    frame = _fake_frame_ready(camera_id=camera_id, frame_index=frame_index)
    ctx = FrameContext(
        frame=frame,  # type: ignore[arg-type]
        event_time=now,
        capture_time=now,
    )
    ctx.world_snapshots = snapshots or []
    ctx.outcome_decisions = []
    ctx.domain_detections = []
    return ctx


class TestTrajectoryWritesFromSnapshots:
    async def test_writes_one_row_per_snapshot_for_current_camera(self) -> None:
        """2 PHs on cam-A, 1 PH on cam-B. Frame is from cam-A → 2 rows."""
        now = datetime.now(UTC)
        snap_a1 = _make_snapshot(ph_id="ph-a1", camera_id="cam-a", captured_at=now)
        snap_a2 = _make_snapshot(ph_id="ph-a2", camera_id="cam-a", captured_at=now)
        snap_b1 = _make_snapshot(ph_id="ph-b1", camera_id="cam-b", captured_at=now)

        repo = InMemoryTrajectoryRepository()
        writer = TrajectoryWriter(repo=repo)
        stage = TrajectoryStage(trajectory_writer=writer)

        ctx = _make_ctx(
            camera_id="cam-a",
            now=now,
            snapshots=[snap_a1, snap_a2, snap_b1],
        )
        await stage.run(ctx)

        assert len(repo._points) == 2
        written_ph_ids = {p.ph_id for p in repo._points}
        assert written_ph_ids == {"ph-a1", "ph-a2"}

    async def test_floor_point_matches_snapshot(self) -> None:
        """Snapshot floor_x_m=1.5, floor_y_m=2.3 → written in metres."""
        now = datetime.now(UTC)
        snap = _make_snapshot(ph_id="ph-1", floor_x_m=1.5, floor_y_m=2.3, captured_at=now)

        repo = InMemoryTrajectoryRepository()
        writer = TrajectoryWriter(repo=repo)
        stage = TrajectoryStage(trajectory_writer=writer)

        ctx = _make_ctx(now=now, snapshots=[snap])
        await stage.run(ctx)

        assert len(repo._points) == 1
        pt = repo._points[0]
        # TrajectoryWriter converts mm→m, so 1500mm → 1.5m
        assert pt.ground_x == 1.5
        assert pt.ground_y == 2.3

    async def test_identity_from_snapshot(self) -> None:
        """Snapshot carries identity_id and identity_confidence."""
        now = datetime.now(UTC)
        snap = _make_snapshot(
            ph_id="ph-1",
            identity_id="alice",
            identity_confidence=0.92,
            captured_at=now,
        )

        repo = InMemoryTrajectoryRepository()
        writer = TrajectoryWriter(repo=repo)
        stage = TrajectoryStage(trajectory_writer=writer)

        ctx = _make_ctx(now=now, snapshots=[snap])
        await stage.run(ctx)

        assert len(repo._points) == 1
        pt = repo._points[0]
        assert pt.identity_id == "alice"
        assert pt.identity_confidence == 0.92

    async def test_skips_other_camera_snapshots(self) -> None:
        """Only current-frame camera's PHs write trajectory."""
        now = datetime.now(UTC)
        snap_a = _make_snapshot(ph_id="ph-a", camera_id="cam-a", captured_at=now)
        snap_b = _make_snapshot(ph_id="ph-b", camera_id="cam-b", captured_at=now)

        repo = InMemoryTrajectoryRepository()
        writer = TrajectoryWriter(repo=repo)
        stage = TrajectoryStage(trajectory_writer=writer)

        ctx = _make_ctx(camera_id="cam-a", now=now, snapshots=[snap_a, snap_b])
        await stage.run(ctx)

        assert len(repo._points) == 1
        assert repo._points[0].ph_id == "ph-a"

    async def test_no_snapshots_no_write(self) -> None:
        """Empty world_snapshots → no trajectory points written."""
        repo = InMemoryTrajectoryRepository()
        writer = TrajectoryWriter(repo=repo)
        stage = TrajectoryStage(trajectory_writer=writer)

        ctx = _make_ctx(snapshots=[])
        await stage.run(ctx)

        assert len(repo._points) == 0

    async def test_non_representative_camera_ingests_posture(self) -> None:
        """A camera that does not represent a PH still feeds its posture to fusion.

        PH ``ph-x`` is represented by ``cam-a`` (snapshot camera). When the frame
        from ``cam-b`` is processed, its own detection of ``ph-x`` must land in the
        GlobalPostureTracker store under ``cam-b`` so cross-camera fusion can use
        it, even though ``cam-b`` writes no trajectory row for ``ph-x``.
        """
        now = datetime.now(UTC)
        snap = _make_snapshot(ph_id="ph-x", camera_id="cam-a", captured_at=now)

        repo = InMemoryTrajectoryRepository()
        writer = TrajectoryWriter(repo=repo)
        posture_tracker = GlobalPostureTracker(required_consecutive=1)
        stage = TrajectoryStage(
            trajectory_writer=writer,
            motion_energy_tracker=MotionEnergyTracker(),
            posture_tracker=posture_tracker,
        )

        ctx = _make_ctx(camera_id="cam-b", now=now, snapshots=[snap])
        ctx.domain_detections = [
            Detection(
                detection_id="det-b",
                camera_id="cam-b",
                bbox=BoundingBox(0.0, 0.0, 10.0, 20.0),
                embedding=[],
                capture_time=now,
                event_time=now,
                ph_id="ph-x",
                floor_point=FloorPoint(0, 0),
            )
        ]
        ctx.det_posture_scores["det-b"] = PostureScores(
            lying=0.9, sitting=0.0, standing_walking=0.0, keypoint_confidence=0.85
        )

        await stage.run(ctx)

        # cam-b wrote no trajectory row (it does not represent ph-x).
        assert len(repo._points) == 0
        # But cam-b's posture evidence is now in the fusion store for ph-x.
        assert "cam-b" in posture_tracker._snapshots["ph-x"]
        assert posture_tracker._snapshots["ph-x"]["cam-b"].lying == 0.9
        assert posture_tracker._snapshots["ph-x"]["cam-b"].view_weight == 1.0

    async def test_non_representative_posture_uses_geometry_view_weight(self) -> None:
        """Posture ingest uses the shared geometry descriptor when it is present."""
        now = datetime.now(UTC)
        snap = _make_snapshot(ph_id="ph-x", camera_id="cam-a", captured_at=now)

        repo = InMemoryTrajectoryRepository()
        writer = TrajectoryWriter(repo=repo)
        posture_tracker = GlobalPostureTracker(required_consecutive=1)
        stage = TrajectoryStage(
            trajectory_writer=writer,
            motion_energy_tracker=MotionEnergyTracker(),
            posture_tracker=posture_tracker,
        )

        ctx = _make_ctx(camera_id="cam-b", now=now, snapshots=[snap])
        ctx.domain_detections = [
            Detection(
                detection_id="det-b",
                camera_id="cam-b",
                bbox=BoundingBox(0.0, 0.0, 10.0, 20.0),
                embedding=[],
                capture_time=now,
                event_time=now,
                ph_id="ph-x",
                floor_point=FloorPoint(0, 0),
            )
        ]
        ctx.det_posture_scores["det-b"] = PostureScores(
            lying=0.9, sitting=0.0, standing_walking=0.0, keypoint_confidence=0.85
        )
        ctx.geometry_by_detection["det-b"] = ObservationGeometry(
            footpoint_px=(5.0, 20.0),
            floor_residual_m=0.02,
            footpoint_reliable=False,
            detection_confidence=0.9,
            crop_quality=0.8,
            orientation=OrientationBin.FRONT,
            orientation_confidence=1.0,
        )

        await stage.run(ctx)

        assert posture_tracker._snapshots["ph-x"]["cam-b"].view_weight == pytest.approx(0.18)

    async def test_room_name_from_snapshot(self) -> None:
        """Trajectory point room_name comes from the snapshot, not camera_room_map."""
        now = datetime.now(UTC)
        snap = _make_snapshot(ph_id="ph-1", room_name="kitchen", captured_at=now)

        repo = InMemoryTrajectoryRepository()
        writer = TrajectoryWriter(repo=repo)
        stage = TrajectoryStage(trajectory_writer=writer)

        ctx = _make_ctx(now=now, snapshots=[snap])
        await stage.run(ctx)

        assert len(repo._points) == 1
        assert repo._points[0].room_name == "kitchen"
