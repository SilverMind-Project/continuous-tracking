"""Trajectory stages: close terminated PHs and write trajectory points."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from structlog import get_logger

from ...domain import FloorPoint, PostureType
from ...inference.schemas import PoseResult
from ...tracking.floor_projector import FloorProjector
from ...trajectory.motion_energy import MotionEnergyTracker
from ...trajectory.posture import GlobalPostureTracker
from ...trajectory.trajectory_writer import TrajectoryWriter
from ..frame_context import FrameContext
from .base import FrameStage

if TYPE_CHECKING:
    from ...transport.dwell_publisher import DwellPublisher
    from ...transport.presence_publisher import PresencePublisher

logger = get_logger(__name__)


class ClosePHStage(FrameStage):
    """Close PHs that have disappeared from the active set.

    Uses ``ctx.active_ph_ids`` directly. No GlobalTrackRepository dependency.
    Closes trajectory, motion energy, and posture state by PH id.

    Emits presence-disappeared and dwell-ended events on PH close.
    """

    name = "close_ph"

    def __init__(
        self,
        trajectory_writer: TrajectoryWriter | None = None,
        motion_energy_tracker: MotionEnergyTracker | None = None,
        posture_tracker: GlobalPostureTracker | None = None,
        prev_active_ph_ids: set[str] | None = None,
        presence_publisher: PresencePublisher | None = None,
        dwell_publisher: DwellPublisher | None = None,
    ) -> None:
        self._trajectory_writer = trajectory_writer
        self._motion_energy_tracker = motion_energy_tracker
        self._posture_tracker = posture_tracker
        self._prev_active_ph_ids: set[str] = (
            prev_active_ph_ids if prev_active_ph_ids is not None else set()
        )
        self._presence_publisher = presence_publisher
        self._dwell_publisher = dwell_publisher
        # Per-PH state for presence and dwell event emission.
        self._seen_ph_ids: set[str] = set()
        self._last_identity_by_ph: dict[str, str | None] = {}
        self._last_room_by_ph: dict[str, str] = {}
        self._room_entered_at: dict[str, datetime] = {}

    async def run(self, ctx: FrameContext) -> None:
        current_ph_ids = ctx.active_ph_ids
        event_time_ns = int(ctx.event_time.timestamp() * 1e9)

        # Detect new PHs and emit presence-appeared + dwell-started.
        # Only emit when the PH has a WorldFrameSnapshot, which proves it met
        # min_observations_to_publish (gated in WorldTracker.step).  PHs that
        # enter active_ph_ids but lack a snapshot are not yet confirmed.
        new_ph_ids = current_ph_ids - self._seen_ph_ids
        if new_ph_ids:
            snap_by_ph = {s.ph_id: s for s in ctx.world_snapshots}
            for ph_id in new_ph_ids:
                snap = snap_by_ph.get(ph_id)
                if snap is None:
                    # PH not yet confirmed — skip until min_observations_to_publish met.
                    continue
                identity_id = snap.identity_id if snap else None
                room_name = snap.room_name if snap else ""
                self._last_identity_by_ph[ph_id] = identity_id
                self._last_room_by_ph[ph_id] = room_name
                self._room_entered_at[ph_id] = ctx.event_time
                # Emit presence-appeared.
                if self._presence_publisher is not None:
                    await self._presence_publisher.publish_appeared(
                        ph_id=ph_id,
                        identity_id=identity_id,
                        room_name=room_name,
                        event_time_unix_ns=event_time_ns,
                    )
                # Emit dwell-started on first room assignment.
                if self._dwell_publisher is not None and room_name:
                    await self._dwell_publisher.publish_started(
                        ph_id=ph_id,
                        identity_id=identity_id,
                        room_name=room_name,
                        event_time_unix_ns=event_time_ns,
                    )
                # Only mark as seen once presence event is emitted.
                self._seen_ph_ids.add(ph_id)

        # Update identity and room tracking for active PHs from snapshots.
        for snap in ctx.world_snapshots:
            ph_id = snap.ph_id
            if ph_id not in current_ph_ids:
                continue
            prev_room = self._last_room_by_ph.get(ph_id)
            # Update identity.
            if snap.identity_id:
                self._last_identity_by_ph[ph_id] = snap.identity_id
            # Detect room change → dwell-ended for old room, dwell-started for new.
            if (
                prev_room
                and snap.room_name
                and snap.room_name != prev_room
                and self._dwell_publisher is not None
            ):
                # Compute dwell duration in the old room.
                entered_at = self._room_entered_at.get(ph_id)
                old_duration_s = (
                    int((ctx.event_time - entered_at).total_seconds())
                    if entered_at is not None
                    else 0
                )
                await self._dwell_publisher.publish_ended(
                    ph_id=ph_id,
                    identity_id=self._last_identity_by_ph.get(ph_id),
                    room_name=prev_room,
                    event_time_unix_ns=event_time_ns,
                    duration_s=old_duration_s,
                )
                await self._dwell_publisher.publish_started(
                    ph_id=ph_id,
                    identity_id=self._last_identity_by_ph.get(ph_id),
                    room_name=snap.room_name,
                    event_time_unix_ns=event_time_ns,
                )
                self._room_entered_at[ph_id] = ctx.event_time
            self._last_room_by_ph[ph_id] = snap.room_name

        terminated_ph_ids = self._prev_active_ph_ids - current_ph_ids
        close_time = ctx.event_time

        # Close tracks first to capture dwell duration, then emit tier-2 events.
        for ph_id in terminated_ph_ids:
            identity_id = self._last_identity_by_ph.get(ph_id)
            room_name = self._last_room_by_ph.get(ph_id, "")
            duration_s = 0

            logger.debug("Closing terminated PH", ph_id=ph_id)
            if self._trajectory_writer:
                duration_s = await self._trajectory_writer.close_track(ph_id, closed_at=close_time)
            if self._motion_energy_tracker is not None:
                self._motion_energy_tracker.evict_track(ph_id)
            if self._posture_tracker is not None:
                self._posture_tracker.evict_track(ph_id)

            if self._presence_publisher is not None:
                await self._presence_publisher.publish_disappeared(
                    ph_id=ph_id,
                    identity_id=identity_id,
                    room_name=room_name,
                    event_time_unix_ns=event_time_ns,
                )
            if self._dwell_publisher is not None:
                await self._dwell_publisher.publish_ended(
                    ph_id=ph_id,
                    identity_id=identity_id,
                    room_name=room_name,
                    event_time_unix_ns=event_time_ns,
                    duration_s=duration_s,
                )

        self._prev_active_ph_ids = current_ph_ids
        # Clean up state for terminated PHs.
        for ph_id in terminated_ph_ids:
            self._seen_ph_ids.discard(ph_id)
            self._last_identity_by_ph.pop(ph_id, None)
            self._last_room_by_ph.pop(ph_id, None)
            self._room_entered_at.pop(ph_id, None)


class TrajectoryStage(FrameStage):
    """Writes trajectory points from WorldFrameSnapshots."""

    name = "trajectory"

    def __init__(
        self,
        trajectory_writer: TrajectoryWriter | None = None,
        floor_projector: FloorProjector | None = None,
        motion_energy_tracker: MotionEnergyTracker | None = None,
        posture_tracker: GlobalPostureTracker | None = None,
    ) -> None:
        self._trajectory_writer = trajectory_writer
        self._floor_projector = floor_projector
        self._motion_energy_tracker = motion_energy_tracker
        self._posture_tracker = posture_tracker

    async def run(self, ctx: FrameContext) -> None:
        if not ctx.world_snapshots or not self._trajectory_writer:
            return

        traj_time = ctx.event_time
        decision_by_ph = {d.ph_id: d for d in ctx.outcome_decisions}

        # Start fresh dwell segments for revived PHs before writing trajectory
        # points, so dwell rows are clean and auditable (not resurrecting old dwells).
        for snap in ctx.world_snapshots:
            if snap.ph_id in ctx.revived_ph_ids:
                await self._trajectory_writer.start_segment(
                    ph_id=snap.ph_id,
                    identity_id=snap.identity_id,
                    room_name=snap.room_name,
                    entered_at=traj_time,
                )

        for snap in ctx.world_snapshots:
            if snap.camera_id != ctx.frame.camera_id:
                continue

            floor_point = FloorPoint(int(snap.floor_x_m * 1000.0), int(snap.floor_y_m * 1000.0))

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
                ph_id=snap.ph_id,
                room_name=snap.room_name,
                floor_point=floor_point,
                captured_at=traj_time,
                identity_confidence=identity_confidence,
                posture=gt_posture,
                motion_energy=gt_motion_energy,
            )

    def _find_pose_for_ph(
        self, ctx: FrameContext, ph_id: str
    ) -> tuple[PoseResult | None, str | None]:
        """Match pose to PH via backfilled ph_id."""
        for det in ctx.domain_detections:
            if det.ph_id == ph_id:
                return ctx.det_pose_result.get(det.detection_id), det.detection_id
        return None, None
