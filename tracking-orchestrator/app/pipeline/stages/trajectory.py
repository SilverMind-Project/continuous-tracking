"""Trajectory stages: close terminated global tracks and write trajectory points."""

from __future__ import annotations

from structlog import get_logger

from ...domain import FloorPoint, PostureType
from ...inference.schemas import PoseResult
from ...storage.base import GlobalTrackRepository
from ...tracking.floor_projector import FloorProjector
from ...trajectory.motion_energy import MotionEnergyTracker
from ...trajectory.posture import GlobalPostureTracker
from ...trajectory.trajectory_writer import TrajectoryWriter
from ..frame_context import FrameContext
from .base import FrameStage

logger = get_logger(__name__)


class CloseTerminatedStage(FrameStage):
    """Close global tracks that have disappeared from the active set.

    ``ctx.active_global_tracks`` is sourced from open PH state by
    ``WorldTrackingStage`` (WT2 bridge).  A GT is considered terminated
    when it was active on the previous frame but is absent this frame.
    """

    name = "close_terminated"

    def __init__(
        self,
        global_track_repo: GlobalTrackRepository | None = None,
        trajectory_writer: TrajectoryWriter | None = None,
        motion_energy_tracker: MotionEnergyTracker | None = None,
        posture_tracker: GlobalPostureTracker | None = None,
        prev_active_gt_ids: set[str] | None = None,
    ) -> None:
        self._global_track_repo = global_track_repo
        self._trajectory_writer = trajectory_writer
        self._motion_energy_tracker = motion_energy_tracker
        self._posture_tracker = posture_tracker
        # Shared mutable state (owned by pipeline).
        self._prev_active_gt_ids: set[str] = (
            prev_active_gt_ids if prev_active_gt_ids is not None else set()
        )

    async def run(self, ctx: FrameContext) -> None:
        current_gt_ids = {gt.global_track_id for gt in ctx.active_global_tracks}
        terminated_gt_ids = self._prev_active_gt_ids - current_gt_ids
        if not terminated_gt_ids:
            self._prev_active_gt_ids = current_gt_ids
            return

        traj_close_time = ctx.event_time
        for gt_id in terminated_gt_ids:
            logger.debug("Closing terminated global track", global_track_id=gt_id)
            if self._global_track_repo is not None:
                await self._global_track_repo.close_global_track(gt_id)
            if self._trajectory_writer:
                await self._trajectory_writer.close_track(gt_id, closed_at=traj_close_time)
            if self._motion_energy_tracker is not None:
                self._motion_energy_tracker.evict_track(gt_id)
            if self._posture_tracker is not None:
                self._posture_tracker.evict_track(gt_id)
        self._prev_active_gt_ids = current_gt_ids


class TrajectoryStage(FrameStage):
    name = "trajectory"

    def __init__(
        self,
        trajectory_writer: TrajectoryWriter | None = None,
        floor_projector: FloorProjector | None = None,
        motion_energy_tracker: MotionEnergyTracker | None = None,
        posture_tracker: GlobalPostureTracker | None = None,
        tracklet_manager: object | None = None,  # N0: was TrackletManager, deleted
        camera_room_map: dict[str, str] | None = None,
    ) -> None:
        self._trajectory_writer = trajectory_writer
        self._floor_projector = floor_projector
        self._motion_energy_tracker = motion_energy_tracker
        self._posture_tracker = posture_tracker
        self._tracklet_manager = tracklet_manager
        self._camera_room_map = camera_room_map or {}

    async def run(self, ctx: FrameContext) -> None:
        if not ctx.world_snapshots or not self._trajectory_writer:
            return

        traj_time = ctx.event_time
        decision_by_ph = {d.global_track_id: d for d in ctx.outcome_decisions}

        for snap in ctx.world_snapshots:
            if snap.camera_id != ctx.frame.camera_id:
                continue

            # Snap floor_x/y are in metres; FloorPoint uses mm.
            floor_point = FloorPoint(int(snap.floor_x_m * 1000.0), int(snap.floor_y_m * 1000.0))

            # Pose / motion-energy attribution: WT3 transitional — returns
            # (None, None) until WT4 stamps global_track_id on detections.
            pose, det_id = self._find_pose_for_ph(ctx, snap.ph_id)

            gt_posture: PostureType = "unknown"
            gt_motion_energy: float | None = None
            if pose is not None and det_id is not None and self._motion_energy_tracker is not None:
                bbox = snap.bbox
                if bbox is not None:
                    bbox_diag = (bbox.width**2 + bbox.height**2) ** 0.5
                    me = self._motion_energy_tracker.update(
                        snap.ph_id, pose, traj_time, bbox_diag_px=bbox_diag
                    )
                    gt_motion_energy = me.mean_keypoint_velocity_px_s
                posture_scores = ctx.det_posture_scores.get(det_id)
                if posture_scores is not None and self._posture_tracker is not None:
                    gt_posture = self._posture_tracker.update(
                        global_track_id=snap.ph_id,
                        camera_id=snap.camera_id,
                        scores=posture_scores,
                        active_camera_ids=[snap.camera_id],
                        motion_energy=gt_motion_energy,
                    )
                    ctx.det_posture[det_id] = gt_posture

            decision = decision_by_ph.get(snap.ph_id)
            identity_confidence = (
                decision.posterior.top_identity()[1]
                if decision is not None
                else snap.identity_confidence
            )

            await self._trajectory_writer.write(
                identity_id=snap.identity_id,
                global_track_id=snap.ph_id,
                room_name=snap.room_name,
                floor_point=floor_point,
                captured_at=traj_time,
                identity_confidence=identity_confidence,
                posture="unknown",
                motion_energy=None,
            )

    def _find_pose_for_ph(
        self, ctx: FrameContext, ph_id: str
    ) -> tuple[PoseResult | None, str | None]:
        """WT3 transitional: match by global_track_id on detection.

        Returns (None, None) until WT4 stamps det.global_track_id = ph_id.
        """
        for det in ctx.domain_detections:
            if det.global_track_id == ph_id:
                return ctx.det_pose_result.get(det.detection_id), det.detection_id
        return None, None
