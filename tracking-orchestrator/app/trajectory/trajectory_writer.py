"""TrajectoryWriter: writes person_trajectories rows and manages room dwells.

One trajectory point is written per committed identity decision. When the
room changes (detected by comparing the current room stored in memory against
the incoming room_name), the open dwell is closed and a new one is opened.

This module does NOT import from transport or pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from ..domain import FloorPoint, PersonTrajectoryPoint, RoomDwell
from ..storage.base import TrajectoryRepository


class TrajectoryWriter:
    """Writes trajectory points and manages room dwell intervals.

    Usage::

        writer = TrajectoryWriter(repo=trajectory_repo)

        # Called once per committed identity decision per frame.
        await writer.write(
            identity_id="alice",
            global_track_id="gt-001",
            room_name="kitchen",
            floor_point=FloorPoint(3500, 2100),
            captured_at=datetime.now(UTC),
            identity_confidence=0.92,
        )

        # Called when a global track is closed.
        await writer.close_track("gt-001", closed_at=datetime.now(UTC))
    """

    def __init__(self, repo: TrajectoryRepository) -> None:
        self._repo = repo
        # In-memory room state per global_track_id.
        self._current_room: dict[str, str] = {}
        # Open dwell per global_track_id (not yet closed/exited).
        self._open_dwell: dict[str, RoomDwell] = {}

    async def write(
        self,
        identity_id: str,
        global_track_id: str,
        room_name: str,
        floor_point: FloorPoint,
        captured_at: datetime,
        identity_confidence: float = 0.0,
    ) -> PersonTrajectoryPoint:
        """Write one trajectory point and update the room dwell state.

        Args:
            identity_id: committed identity for this track.
            global_track_id: the GlobalTrack this person belongs to.
            room_name: current room (resolved from camera → stream assignment).
            floor_point: ground-plane position in millimeters.
            captured_at: wall-clock time of the observation.
            identity_confidence: posterior probability of the top identity.

        Returns:
            The persisted PersonTrajectoryPoint.
        """
        point = PersonTrajectoryPoint(
            identity_id=identity_id,
            global_track_id=global_track_id,
            observed_at=captured_at,
            room_name=room_name,
            ground_x=floor_point.x_mm / 1000.0,
            ground_y=floor_point.y_mm / 1000.0,
            posture="unknown",
            identity_confidence=identity_confidence,
        )
        await self._repo.save_trajectory_point(point)

        await self._handle_dwell(
            identity_id, global_track_id, room_name, captured_at, identity_confidence
        )

        return point

    async def close_track(self, global_track_id: str, closed_at: datetime) -> None:
        """Close the open dwell for a terminated global track."""
        dwell = self._open_dwell.pop(global_track_id, None)
        if dwell is not None:
            duration = int((closed_at - dwell.entered_at).total_seconds())
            closed = RoomDwell(
                dwell_id=dwell.dwell_id,
                identity_id=dwell.identity_id,
                global_track_id=dwell.global_track_id,
                room_name=dwell.room_name,
                entered_at=dwell.entered_at,
                exited_at=closed_at,
                duration_seconds=duration,
                entry_confidence=dwell.entry_confidence,
                primary_posture="unknown",
                activity_summary={},
            )
            await self._repo.update_room_dwell(closed)
            self._current_room.pop(global_track_id, None)

    async def _handle_dwell(
        self,
        identity_id: str,
        global_track_id: str,
        room_name: str,
        captured_at: datetime,
        identity_confidence: float,
    ) -> None:
        prev_room = self._current_room.get(global_track_id)
        if prev_room == room_name:
            return  # still in the same room, dwell continues

        # Room changed (or first observation for this track).
        existing = self._open_dwell.get(global_track_id)
        if existing is not None:
            duration = int((captured_at - existing.entered_at).total_seconds())
            closed = RoomDwell(
                dwell_id=existing.dwell_id,
                identity_id=existing.identity_id,
                global_track_id=existing.global_track_id,
                room_name=existing.room_name,
                entered_at=existing.entered_at,
                exited_at=captured_at,
                duration_seconds=duration,
                entry_confidence=existing.entry_confidence,
                primary_posture="unknown",
                activity_summary={},
            )
            await self._repo.update_room_dwell(closed)

        new_dwell = RoomDwell(
            dwell_id=str(uuid.uuid4()),
            identity_id=identity_id,
            global_track_id=global_track_id,
            room_name=room_name,
            entered_at=captured_at,
            entry_confidence=identity_confidence,
        )
        await self._repo.save_room_dwell(new_dwell)
        self._open_dwell[global_track_id] = new_dwell
        self._current_room[global_track_id] = room_name
