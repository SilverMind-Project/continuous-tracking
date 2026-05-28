"""WT2: tests for the PH-to-GlobalTrack bridge in WorldTrackingStage.

Verifies that:
1. ctx.active_global_tracks is built from open PHs after world tracking.
2. Closed PHs are excluded from the active list.
3. CloseTerminatedStage fires close_track / evict_track only on real PH termination.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from app.domain import (
    BoundingBox,
    Detection,
    FloorPoint,
    GlobalTrack,
    PersonHypothesis,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.trajectory import CloseTerminatedStage
from app.pipeline.stages.world_tracking import (
    WorldTrackingStage,
    _phs_to_global_tracks,
)
from app.storage.base import (
    InMemoryGlobalTrackRepository,
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_frame_ready(*, camera_id: str = "cam1", frame_index: int = 1) -> MagicMock:
    return MagicMock(camera_id=camera_id, frame_index=frame_index)


def _make_detection(
    camera_id: str = "cam1",
    frame_index: int = 1,
    *,
    now: datetime | None = None,
) -> Detection:
    if now is None:
        now = datetime.now(UTC)
    return Detection(
        detection_id=f"det-{camera_id}-{frame_index}",
        camera_id=camera_id,
        bbox=BoundingBox(0, 0, 100, 200),
        embedding=[1.0, 0.0, 0.0],
        capture_time=now,
        event_time=now,
        confidence=0.9,
        floor_point=FloorPoint(0, 0, calibrated=True),
    )


def _make_frame_context(
    *,
    camera_id: str = "cam1",
    frame_index: int = 1,
    now: datetime | None = None,
    detections: list[Detection] | None = None,
) -> FrameContext:
    if now is None:
        now = datetime.now(UTC)
    frame = _fake_frame_ready(camera_id=camera_id, frame_index=frame_index)
    ctx = FrameContext(
        frame=frame,  # type: ignore[arg-type]
        event_time=now,
        capture_time=now,
    )
    ctx.domain_detections = detections or []
    ctx.face_anchors = []
    ctx.new_revisions = []
    ctx.committed_ids = {}
    return ctx


def _make_room_polygons() -> dict[str, list[tuple[float, float]]]:
    return {"room1": [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]}


# ---------------------------------------------------------------------------
# Tests: _phs_to_global_tracks
# ---------------------------------------------------------------------------


class TestPhsToGlobalTracks:
    def test_open_phs_become_active_global_tracks(self) -> None:
        now = datetime.now(UTC)
        ph = PersonHypothesis(
            ph_id="ph-1",
            state_mean=(0.0, 0.0, 0.0, 0.0),
            state_cov=tuple([0.1] * 16),
            born_at=now,
            last_seen_at=now,
            last_seen_camera="cam1",
            observation_count=5,
            active_cameras=frozenset(["cam1", "cam2"]),
            current_identity_id="alice",
            current_identity_committed_at=now,
        )

        result = _phs_to_global_tracks([ph])

        assert len(result) == 1
        gt = result[0]
        assert isinstance(gt, GlobalTrack)
        assert gt.global_track_id == "ph-1"
        assert gt.tracklet_ids == []
        assert set(gt.camera_ids) == {"cam1", "cam2"}
        assert gt.started_at == now
        assert gt.last_seen_at == now
        assert gt.current_identity_id == "alice"
        assert gt.current_identity_committed_at == now
        assert gt.state == "active"

    def test_closed_phs_excluded(self) -> None:
        now = datetime.now(UTC)
        open_ph = PersonHypothesis(
            ph_id="ph-1",
            state_mean=(0.0, 0.0, 0.0, 0.0),
            state_cov=tuple([0.1] * 16),
            born_at=now,
            last_seen_at=now,
            last_seen_camera="cam1",
            observation_count=5,
        )
        closed_ph = PersonHypothesis(
            ph_id="ph-2",
            state_mean=(1.0, 1.0, 0.0, 0.0),
            state_cov=tuple([0.1] * 16),
            born_at=now,
            last_seen_at=now,
            last_seen_camera="cam1",
            observation_count=3,
            closed_at=now,
        )

        result = _phs_to_global_tracks([open_ph, closed_ph])

        assert len(result) == 1
        assert result[0].global_track_id == "ph-1"

    def test_empty_input(self) -> None:
        assert _phs_to_global_tracks([]) == []

    def test_all_closed_input(self) -> None:
        now = datetime.now(UTC)
        closed = PersonHypothesis(
            ph_id="ph-1",
            state_mean=(0.0, 0.0, 0.0, 0.0),
            state_cov=tuple([0.1] * 16),
            born_at=now,
            last_seen_at=now,
            last_seen_camera="cam1",
            observation_count=1,
            closed_at=now,
        )
        assert _phs_to_global_tracks([closed]) == []


# ---------------------------------------------------------------------------
# Tests: WorldTrackingStage populates active_global_tracks
# ---------------------------------------------------------------------------


class TestWorldTrackingStageActiveGlobalTracks:
    async def test_active_global_tracks_built_from_open_phs(self) -> None:
        """WorldTrackingStage populates ctx.active_global_tracks from open PHs."""
        now = datetime.now(UTC)
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()
        cfg = WorldTrackerConfig(min_observations_to_publish=1)
        tracker = WorldTracker(
            ph_repo=ph_repo,
            obs_repo=obs_repo,
            config=cfg,
        )
        stage = WorldTrackingStage(
            tracker=tracker,
            config=cfg,
            room_polygons=_make_room_polygons(),
        )

        # Frame 1: one detection on cam1 — spawns one PH.
        ctx = _make_frame_context(
            camera_id="cam1",
            frame_index=1,
            now=now,
            detections=[_make_detection("cam1", 1, now=now)],
        )
        await stage.run(ctx)

        assert len(ctx.active_global_tracks) == 1, (
            f"expected 1 active GT, got {len(ctx.active_global_tracks)}"
        )
        gt = ctx.active_global_tracks[0]
        assert isinstance(gt, GlobalTrack)
        assert set(gt.camera_ids) == {"cam1"}
        assert gt.global_track_id != ""

    async def test_camera_ids_accumulate_across_cameras(self) -> None:
        """After observations from two cameras, active_cameras reflects both."""
        now = datetime.now(UTC)
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()
        cfg = WorldTrackerConfig(min_observations_to_publish=1)
        tracker = WorldTracker(
            ph_repo=ph_repo,
            obs_repo=obs_repo,
            config=cfg,
        )
        stage = WorldTrackingStage(
            tracker=tracker,
            config=cfg,
            room_polygons=_make_room_polygons(),
        )

        # Frame 1: cam1
        ctx1 = _make_frame_context(
            camera_id="cam1",
            frame_index=1,
            now=now,
            detections=[_make_detection("cam1", 1, now=now)],
        )
        await stage.run(ctx1)

        # Frame 2: cam2, same position — should match existing PH.
        t2 = now + timedelta(seconds=1)
        ctx2 = _make_frame_context(
            camera_id="cam2",
            frame_index=2,
            now=t2,
            detections=[_make_detection("cam2", 2, now=t2)],
        )
        await stage.run(ctx2)

        assert len(ctx2.active_global_tracks) == 1
        gt = ctx2.active_global_tracks[0]
        assert set(gt.camera_ids) == {"cam1", "cam2"}


# ---------------------------------------------------------------------------
# Tests: CloseTerminatedStage regression
# ---------------------------------------------------------------------------


class _FakeTrajectoryWriter:
    def __init__(self) -> None:
        self.closed_tracks: list[tuple[str, datetime]] = []

    async def close_track(self, gt_id: str, *, closed_at: datetime) -> None:
        self.closed_tracks.append((gt_id, closed_at))


class _FakeMotionEnergyTracker:
    def __init__(self) -> None:
        self.evicted: list[str] = []

    def evict_track(self, gt_id: str) -> None:
        self.evicted.append(gt_id)


class _FakePostureTracker:
    def __init__(self) -> None:
        self.evicted: list[str] = []

    def evict_track(self, gt_id: str) -> None:
        self.evicted.append(gt_id)


class TestCloseTerminatedRegression:
    async def test_close_terminated_fires_once_per_ph(self) -> None:
        """close_track / evict_track fire only on real PH closure.

        Frames 1-5: PH A continuously visible at (0,0).
        Frame 6: no observations, time advanced past ph_close_grace_s.
        """
        now = datetime.now(UTC)
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()
        global_track_repo = InMemoryGlobalTrackRepository()

        close_grace = 0.5  # fast for tests
        cfg = WorldTrackerConfig(
            min_observations_to_publish=1,
            ph_close_grace_s=close_grace,
        )
        tracker = WorldTracker(
            ph_repo=ph_repo,
            obs_repo=obs_repo,
            config=cfg,
        )
        world_stage = WorldTrackingStage(
            tracker=tracker,
            config=cfg,
            room_polygons=_make_room_polygons(),
        )

        fake_writer = _FakeTrajectoryWriter()
        fake_motion = _FakeMotionEnergyTracker()
        fake_posture = _FakePostureTracker()
        close_stage = CloseTerminatedStage(
            global_track_repo=global_track_repo,
            trajectory_writer=fake_writer,  # type: ignore[arg-type]
            motion_energy_tracker=fake_motion,  # type: ignore[arg-type]
            posture_tracker=fake_posture,  # type: ignore[arg-type]
        )

        # Frames 1-5: PH visible.
        for f_idx in range(1, 6):
            t = now + timedelta(seconds=f_idx - 1)
            ctx = _make_frame_context(
                camera_id="cam1",
                frame_index=f_idx,
                now=t,
                detections=[_make_detection("cam1", f_idx, now=t)],
            )
            await world_stage.run(ctx)
            await close_stage.run(ctx)

            # No tracks should be terminated while PH is active.
            assert len(fake_writer.closed_tracks) == 0, f"frame {f_idx}: unexpected close_track"
            assert len(fake_motion.evicted) == 0
            assert len(fake_posture.evicted) == 0

        # Frame 6: no detections, time advanced past close-grace.
        t6 = now + timedelta(seconds=5 + close_grace + 0.5)
        ctx6 = _make_frame_context(
            camera_id="cam1",
            frame_index=6,
            now=t6,
            detections=[],  # empty → no observations → PH will close
        )
        await world_stage.run(ctx6)
        await close_stage.run(ctx6)

        assert len(fake_writer.closed_tracks) == 1, (
            f"expected 1 close_track call, got {len(fake_writer.closed_tracks)}"
        )
        assert len(fake_motion.evicted) == 1
        assert len(fake_posture.evicted) == 1

        # Frame 7: still no detections — no new PH → no new close.
        t7 = t6 + timedelta(seconds=1)
        ctx7 = _make_frame_context(
            camera_id="cam1",
            frame_index=7,
            now=t7,
            detections=[],
        )
        await world_stage.run(ctx7)
        await close_stage.run(ctx7)

        # Should not fire again (PH was already closed on frame 6).
        assert len(fake_writer.closed_tracks) == 1
