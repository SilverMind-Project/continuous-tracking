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

        When a PH is revived after being closed, its previous dwell was already
        finalized and removed from ``_open_dwell``.  This method creates a new
        dwell segment so trajectory writing can resume without relying on an
        implicit side effect from ``_handle_dwell``.

        Must be called before the first ``write()`` for the revived PH in the
        same frame, so ``_handle_dwell`` can detect a room change or merge
        posture counts correctly.
        """
        # Clear any stale state that might linger from the previous lifecycle.
        self._current_room.pop(ph_id, None)
        self._open_dwell.pop(ph_id, None)
        self._dwell_posture_counts.pop(ph_id, None)
        self._dwell_still_acc.pop(ph_id, None)
        self._last_obs_time.pop(ph_id, None)

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

    async def close_track(self, ph_id: str, closed_at: datetime) -> int:
        """Close the open dwell for a terminated PH.

        Returns the dwell duration in seconds, or 0 if no open dwell existed.
        """
        dwell = self._open_dwell.pop(ph_id, None)
        if dwell is not None:
            duration = int((closed_at - dwell.entered_at).total_seconds())
            posture_counts = self._dwell_posture_counts.pop(ph_id, {})
            modal_posture = _modal_posture(posture_counts)
            still_seconds = self._dwell_still_acc.pop(ph_id, 0.0)
            self._last_obs_time.pop(ph_id, None)
            closed = RoomDwell(
                dwell_id=dwell.dwell_id,
                identity_id=dwell.identity_id,
                ph_id=dwell.ph_id,
                room_name=dwell.room_name,
                entered_at=dwell.entered_at,
                exited_at=closed_at,
                duration_seconds=duration,
                entry_confidence=dwell.entry_confidence,
                primary_posture=modal_posture,
                min_motion_energy=dwell.min_motion_energy,
                still_seconds=dwell.still_seconds + int(still_seconds),
                activity_summary={},
            )
            await self._repo.update_room_dwell(closed)
            self._current_room.pop(ph_id, None)
            return duration
        return 0

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

        # Room changed (or first observation for this track).
        existing = self._open_dwell.get(ph_id)
        if existing is not None:
            duration = int((captured_at - existing.entered_at).total_seconds())
            posture_counts = self._dwell_posture_counts.pop(ph_id, {})
            modal_posture = _modal_posture(posture_counts)
            still_seconds = self._dwell_still_acc.pop(ph_id, 0.0)
            self._last_obs_time.pop(ph_id, None)
            closed = RoomDwell(
                dwell_id=existing.dwell_id,
                identity_id=existing.identity_id,
                ph_id=existing.ph_id,
                room_name=existing.room_name,
                entered_at=existing.entered_at,
                exited_at=captured_at,
                duration_seconds=duration,
                entry_confidence=existing.entry_confidence,
                primary_posture=modal_posture,
                min_motion_energy=existing.min_motion_energy,
                still_seconds=existing.still_seconds + int(still_seconds),
                activity_summary={},
            )
            await self._repo.update_room_dwell(closed)

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
