"""Trajectory stages: close terminated global tracks and write trajectory points."""

from __future__ import annotations

from structlog import get_logger

from ...domain import BoundingBox, FloorPoint, PostureType
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
        if not ctx.outcome_decisions or not self._trajectory_writer:
            return

        traj_time = ctx.event_time
        room_name = self._camera_room_map.get(ctx.frame.camera_id, "")

        gt_bbox: dict[str, BoundingBox] = {}
        if ctx.active_tracklets and self._floor_projector:
            for tracklet in ctx.active_tracklets:
                if tracklet.camera_id != ctx.frame.camera_id:
                    continue
                last_bbox: BoundingBox | None = getattr(tracklet, "last_bbox", None)
                if last_bbox is None:
                    continue
                for gt in ctx.active_global_tracks:
                    if tracklet.tracklet_id in gt.tracklet_ids:
                        gt_bbox[gt.global_track_id] = last_bbox
                        break

        for decision in ctx.outcome_decisions:
            gt_bbox_entry = gt_bbox.get(decision.global_track_id)
            if gt_bbox_entry is None:
                continue

            floor_point = (
                self._floor_projector.project(ctx.frame.camera_id, gt_bbox_entry)
                if self._floor_projector is not None
                else FloorPoint(0, 0)
            )
            _top_id, top_prob = decision.posterior.top_identity()

            gt_posture: PostureType = "unknown"
            gt_motion_energy: float | None = None
            pose, matching_detection_id = self._find_pose_for_gt(ctx, decision.global_track_id)

            if pose is not None and matching_detection_id is not None:
                if self._motion_energy_tracker is not None:
                    bbox_diag = (gt_bbox_entry.width**2 + gt_bbox_entry.height**2) ** 0.5
                    me = self._motion_energy_tracker.update(
                        decision.global_track_id, pose, traj_time, bbox_diag_px=bbox_diag
                    )
                    gt_motion_energy = me.mean_keypoint_velocity_px_s

                posture_scores = ctx.det_posture_scores.get(matching_detection_id)
                if posture_scores is not None and self._posture_tracker is not None:
                    gt_obj = next(
                        (
                            gt
                            for gt in ctx.active_global_tracks
                            if gt.global_track_id == decision.global_track_id
                        ),
                        None,
                    )
                    active_camera_ids = gt_obj.camera_ids if gt_obj else [ctx.frame.camera_id]
                    prev_posture = self._posture_tracker.committed_posture(decision.global_track_id)
                    gt_posture = self._posture_tracker.update(
                        global_track_id=decision.global_track_id,
                        camera_id=ctx.frame.camera_id,
                        scores=posture_scores,
                        active_camera_ids=active_camera_ids,
                        motion_energy=gt_motion_energy,
                    )
                    if prev_posture is not None and prev_posture != gt_posture:
                        logger.info(
                            "Posture changed",
                            global_track_id=decision.global_track_id,
                            camera_id=ctx.frame.camera_id,
                            previous=prev_posture,
                            current=gt_posture,
                        )
                    ctx.det_posture[matching_detection_id] = gt_posture
            elif matching_detection_id is not None:
                gt_posture = ctx.det_posture.get(matching_detection_id, "unknown")

            gt_identity = ctx.committed_ids.get(decision.global_track_id)
            await self._trajectory_writer.write(
                identity_id=gt_identity,
                global_track_id=decision.global_track_id,
                room_name=room_name,
                floor_point=floor_point,
                captured_at=traj_time,
                identity_confidence=top_prob,
                posture=gt_posture,
                motion_energy=gt_motion_energy,
            )

    def _find_pose_for_gt(
        self, ctx: FrameContext, global_track_id: str
    ) -> tuple[PoseResult | None, str | None]:
        for domain_det in ctx.domain_detections:
            tid = (
                self._tracklet_manager.get_tracklet_id_for_detection(domain_det.detection_id)  # type: ignore[attr-defined]
                if self._tracklet_manager
                else ""
            )
            if not tid:
                continue
            gt_for_det = next(
                (gt.global_track_id for gt in ctx.active_global_tracks if tid in gt.tracklet_ids),
                "",
            )
            if gt_for_det != global_track_id:
                continue
            pose = ctx.det_pose_result.get(domain_det.detection_id)
            return pose, domain_det.detection_id
        return None, None
