"""TrajectoryWriter: writes person_trajectories rows and manages room dwells.

One trajectory point is written per committed identity decision. When the
room changes (detected by comparing the current room stored in memory against
the incoming room_name), the open dwell is closed and a new one is opened.

This module does NOT import from transport or pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import cast

from ..domain import FloorPoint, PersonTrajectoryPoint, PostureType, RoomDwell
from ..storage.base import TrajectoryRepository

# When motion_energy is below this threshold the frame is considered "still".
_STILL_ENERGY_FLOOR = 0.005
# Maximum seconds to accumulate per observation gap when still.  Prevents
# adding hours of stillness after a long gap between frames.
_MAX_STILL_ACCUMULATION_PER_GAP_S: float = 60.0


class TrajectoryWriter:
    """Writes trajectory points and manages room dwell intervals.

    Usage::

        writer = TrajectoryWriter(repo=trajectory_repo)

        # Called once per committed identity decision per frame.
        await writer.write(
            identity_id="alice",
            ph_id="gt-001",
            room_name="kitchen",
            floor_point=FloorPoint(3500, 2100),
            captured_at=datetime.now(UTC),
            identity_confidence=0.92,
            posture="standing",
            motion_energy=0.012,
        )

        # Called when a PH is closed.
        await writer.close_track("gt-001", closed_at=datetime.now(UTC))
    """

    def __init__(self, repo: TrajectoryRepository) -> None:
        self._repo = repo
        # In-memory room state per ph_id.
        self._current_room: dict[str, str] = {}
        # Open dwell per ph_id (not yet closed/exited).
        self._open_dwell: dict[str, RoomDwell] = {}
        # Per-dwell posture counts for modal calculation.
        self._dwell_posture_counts: dict[str, dict[str, int]] = {}
        # Contiguous still-second accumulator per dwell.
        self._dwell_still_acc: dict[str, float] = {}
        # Last observation time per ph_id (for elapsed-time stillness).
        self._last_obs_time: dict[str, datetime] = {}

    async def write(
        self,
        identity_id: str | None,
        ph_id: str,
        room_name: str,
        floor_point: FloorPoint,
        captured_at: datetime,
        identity_confidence: float = 0.0,
        posture: PostureType = "unknown",
        position_sigma_m: float = 0.0,
        primary_camera_id: str = "",
        contributing_camera_count: int = 1,
        footpoint_reliable: bool = True,
        motion_energy: float | None = None,
        floor_speed_m_s: float | None = None,
    ) -> PersonTrajectoryPoint:
        """Write one trajectory point and update the room dwell state.

        Args:
            identity_id: committed identity for this track, or None when
                the Bayesian resolver has not yet committed.
            ph_id: the PH this person belongs to.
            room_name: current room (resolved from camera to stream assignment).
            floor_point: ground-plane position in millimeters.
            captured_at: wall-clock time of the observation.
            identity_confidence: posterior probability of the top identity.
            posture: classified posture for this frame.
            position_sigma_m: PH floor-position uncertainty in metres.
            primary_camera_id: stabilized best-view camera for this point.
            contributing_camera_count: camera observations fused into this point.
            footpoint_reliable: representative footpoint reliability for this point.
            motion_energy: mean keypoint velocity (None when pose unavailable).

        Returns:
            The persisted PersonTrajectoryPoint.
        """
        point = PersonTrajectoryPoint(
            identity_id=identity_id,
            ph_id=ph_id,
            observed_at=captured_at,
            room_name=room_name,
            ground_x=floor_point.x_mm / 1000.0,
            ground_y=floor_point.y_mm / 1000.0,
            posture=posture,
            identity_confidence=identity_confidence,
            position_sigma_m=position_sigma_m,
            primary_camera_id=primary_camera_id,
            contributing_camera_count=contributing_camera_count,
            footpoint_reliable=footpoint_reliable,
            motion_energy=motion_energy,
            floor_speed_m_s=floor_speed_m_s,
        )
        await self._repo.save_trajectory_point(point)

        await self._handle_dwell(
            identity_id,
            ph_id,
            room_name,
            captured_at,
            identity_confidence,
            posture,
            motion_energy,
        )

        return point

    async def start_segment(
        self,
        ph_id: str,
        identity_id: str | None,
        room_name: str,
        entered_at: datetime,
    ) -> None:
        """Start a clean dwell segment for a revived PH.

        When a PH is revived after being closed, any dwell still open from its
        previous lifecycle must be finalized first — otherwise creating a fresh
        dwell here orphans the old row (its ``exited_at`` stays NULL and it
        overwrites the repository's open-dwell id cache).  This method finalizes
        then creates a new dwell segment so trajectory writing can resume.

        Must be called before the first ``write()`` for the revived PH in the
        same frame, so ``_handle_dwell`` can detect a room change or merge
        posture counts correctly.
        """
        # Close any lingering open dwell at its last observed time (best effort)
        # before starting the new segment, so no row is left with exited_at NULL.
        # Skipping this orphaned the old dwell row (the phantom-stillness leak).
        prior_exit = self._last_obs_time.get(ph_id) or entered_at
        await self._finalize_open_dwell(ph_id, prior_exit)
        self._current_room.pop(ph_id, None)

        # Create a fresh dwell entry for the new segment.
        new_dwell = RoomDwell(
            dwell_id=str(uuid.uuid4()),
            identity_id=identity_id,
            ph_id=ph_id,
            room_name=room_name,
            entered_at=entered_at,
            entry_confidence=0.0,
            min_motion_energy=None,
            still_seconds=0,
        )
        await self._repo.save_room_dwell(new_dwell)
        self._open_dwell[ph_id] = new_dwell
        self._current_room[ph_id] = room_name
        self._dwell_posture_counts[ph_id] = {}
        self._last_obs_time[ph_id] = entered_at

    async def _finalize_open_dwell(self, ph_id: str, closed_at: datetime) -> int:
        """Persist ``exited_at`` for the open dwell of ``ph_id`` and clear its state.

        This is the single place that closes a dwell row.  It MUST be called
        before discarding a PH's open dwell, otherwise the row is left with
        ``exited_at IS NULL`` forever (the leak that produced phantom stillness
        signals — an aged open dwell reads as 60-minute immobility).

        Returns the dwell duration in seconds, or 0 if no open dwell existed.
        ``_current_room`` is intentionally left untouched; callers decide
        whether the PH keeps a current room (room change / revival) or not
        (termination).
        """
        dwell = self._open_dwell.pop(ph_id, None)
        posture_counts = self._dwell_posture_counts.pop(ph_id, {})
        still_seconds = self._dwell_still_acc.pop(ph_id, 0.0)
        self._last_obs_time.pop(ph_id, None)
        if dwell is None:
            return 0
        duration = int((closed_at - dwell.entered_at).total_seconds())
        closed = RoomDwell(
            dwell_id=dwell.dwell_id,
            identity_id=dwell.identity_id,
            ph_id=dwell.ph_id,
            room_name=dwell.room_name,
            entered_at=dwell.entered_at,
            exited_at=closed_at,
            duration_seconds=duration,
            entry_confidence=dwell.entry_confidence,
            primary_posture=_modal_posture(posture_counts),
            min_motion_energy=dwell.min_motion_energy,
            still_seconds=dwell.still_seconds + int(still_seconds),
            activity_summary={},
        )
        await self._repo.update_room_dwell(closed)
        return duration

    async def close_track(self, ph_id: str, closed_at: datetime) -> int:
        """Close the open dwell for a terminated PH.

        Returns the dwell duration in seconds, or 0 if no open dwell existed.
        """
        duration = await self._finalize_open_dwell(ph_id, closed_at)
        self._current_room.pop(ph_id, None)
        return duration

    async def reconcile_open_dwells(self, closed_at: datetime) -> int:
        """Close dwell rows left open by a previous process lifecycle.

        Writer dwell state is in-memory and process-local, so any dwell open
        when the orchestrator stopped can never be closed by ``close_track``
        after a restart (the in-memory handle is gone).  Called once at startup,
        this delegates to the repository to stamp ``exited_at`` on every dangling
        open dwell, using each dwell's last observed trajectory point as the exit
        time (falling back to ``entered_at``).  Returns the rows closed.
        """
        return await self._repo.close_dangling_open_dwells(closed_at)

    async def close_all(self, closed_at: datetime) -> None:
        """Close all open dwells and clear per-track state.

        Called on pipeline shutdown to prevent unbounded memory growth from
        per-track state that is never cleared otherwise.
        """
        for ph_id in list(self._open_dwell):
            await self.close_track(ph_id, closed_at)
        self._current_room.clear()
        self._dwell_posture_counts.clear()
        self._dwell_still_acc.clear()

    async def _handle_dwell(
        self,
        identity_id: str | None,
        ph_id: str,
        room_name: str,
        captured_at: datetime,
        identity_confidence: float,
        posture: PostureType = "unknown",
        motion_energy: float | None = None,
    ) -> None:
        prev_room = self._current_room.get(ph_id)
        if prev_room == room_name:
            # Same room: update posture counts and motion energy for open dwell.
            open_dwell = self._open_dwell.get(ph_id)
            if open_dwell is not None:
                # Accumulate posture count for modal calculation.
                pc = self._dwell_posture_counts.setdefault(ph_id, {})
                pc[posture] = pc.get(posture, 0) + 1
                # Track minimum motion energy.
                if motion_energy is not None:
                    new_min = (
                        min(open_dwell.min_motion_energy, motion_energy)
                        if open_dwell.min_motion_energy is not None
                        else motion_energy
                    )
                    object.__setattr__(open_dwell, "min_motion_energy", new_min)
                # Accumulate still seconds using elapsed time since last observation.
                if motion_energy is not None and motion_energy < _STILL_ENERGY_FLOOR:
                    last_time = self._last_obs_time.get(ph_id)
                    if last_time is not None:
                        elapsed = (captured_at - last_time).total_seconds()
                        elapsed = min(elapsed, _MAX_STILL_ACCUMULATION_PER_GAP_S)
                    else:
                        elapsed = 0.0
                    self._dwell_still_acc[ph_id] = self._dwell_still_acc.get(ph_id, 0.0) + elapsed
                self._last_obs_time[ph_id] = captured_at
            return

        # Room changed (or first observation for this track): close the old dwell.
        await self._finalize_open_dwell(ph_id, captured_at)

        new_dwell = RoomDwell(
            dwell_id=str(uuid.uuid4()),
            identity_id=identity_id,
            ph_id=ph_id,
            room_name=room_name,
            entered_at=captured_at,
            entry_confidence=identity_confidence,
            min_motion_energy=motion_energy,
            still_seconds=0,  # accumulated per observation in _handle_dwell
        )
        # Track first observation time for elapsed-time stillness.
        self._last_obs_time[ph_id] = captured_at
        await self._repo.save_room_dwell(new_dwell)
        self._open_dwell[ph_id] = new_dwell
        self._current_room[ph_id] = room_name
        # Initialize posture count for the new dwell.
        self._dwell_posture_counts[ph_id] = {posture: 1}


def _modal_posture(counts: dict[str, int]) -> PostureType:
    """Return the most frequent posture, defaulting to 'unknown'."""
    if not counts:
        return "unknown"
    return cast(PostureType, max(counts.items(), key=lambda item: item[1])[0])
