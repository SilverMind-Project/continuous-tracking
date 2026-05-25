"""Trajectory point, room dwell, and keyframe storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..domain import PersonTrajectoryPoint, RoomDwell, TaggedKeyframe


class TrajectoryRepository(ABC):
    """Persist person trajectory points and room dwell intervals."""

    @abstractmethod
    async def save_trajectory_point(self, point: PersonTrajectoryPoint) -> None:
        """Append a confirmed trajectory point."""

    @abstractmethod
    async def save_room_dwell(self, dwell: RoomDwell) -> None:
        """Open a new room dwell interval."""

    @abstractmethod
    async def update_room_dwell(self, dwell: RoomDwell) -> None:
        """Close (or update) an existing room dwell interval."""

    @abstractmethod
    async def get_open_dwell(self, identity_id: str, global_track_id: str) -> RoomDwell | None:
        """Return the open (not-yet-exited) dwell for a track, if any."""

    @abstractmethod
    async def list_trajectory_points(
        self,
        identity_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[PersonTrajectoryPoint]:
        """List trajectory points with optional filters."""

    @abstractmethod
    async def list_room_dwells(
        self,
        identity_id: str | None = None,
        room_name: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[RoomDwell]:
        """List room dwell intervals with optional filters."""


class KeyframeRepository(ABC):
    """Persist tagged keyframes."""

    @abstractmethod
    async def save_keyframe(self, keyframe: TaggedKeyframe) -> None:
        """Store a tagged keyframe."""

    @abstractmethod
    async def get_keyframe(self, keyframe_id: str) -> TaggedKeyframe | None:
        """Retrieve a tagged keyframe by ID."""

    @abstractmethod
    async def update_retention(self, keyframe_id: str, expires_at: datetime) -> bool:
        """Update keyframe retention expiry. Returns True if the row existed."""

    @abstractmethod
    async def list_keyframes(
        self,
        tracklet_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[TaggedKeyframe]:
        """List keyframes with optional filters."""


class InMemoryTrajectoryRepository(TrajectoryRepository):
    """In-memory store for trajectory points and room dwells."""

    def __init__(self) -> None:
        self._points: list[PersonTrajectoryPoint] = []
        self._open_dwells: dict[tuple[str | None, str], RoomDwell] = {}
        self._closed_dwells: list[RoomDwell] = []

    async def save_trajectory_point(self, point: PersonTrajectoryPoint) -> None:
        self._points.append(point)

    async def save_room_dwell(self, dwell: RoomDwell) -> None:
        key = (dwell.identity_id, dwell.global_track_id)
        self._open_dwells[key] = dwell

    async def update_room_dwell(self, dwell: RoomDwell) -> None:
        key = (dwell.identity_id, dwell.global_track_id)
        self._open_dwells.pop(key, None)
        self._closed_dwells.append(dwell)

    async def get_open_dwell(self, identity_id: str, global_track_id: str) -> RoomDwell | None:
        return self._open_dwells.get((identity_id, global_track_id))

    async def list_trajectory_points(
        self,
        identity_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[PersonTrajectoryPoint]:
        results = list(self._points)
        if identity_id is not None:
            results = [p for p in results if p.identity_id == identity_id]
        if global_track_id is not None:
            results = [p for p in results if p.global_track_id == global_track_id]
        if after is not None:
            results = [p for p in results if p.observed_at >= after]
        results.sort(key=lambda p: p.observed_at, reverse=True)
        return results[:limit]

    async def list_room_dwells(
        self,
        identity_id: str | None = None,
        room_name: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[RoomDwell]:
        all_dwells = list(self._open_dwells.values()) + self._closed_dwells
        if identity_id is not None:
            all_dwells = [d for d in all_dwells if d.identity_id == identity_id]
        if room_name is not None:
            all_dwells = [d for d in all_dwells if d.room_name == room_name]
        if after is not None:
            all_dwells = [d for d in all_dwells if d.entered_at >= after]
        all_dwells.sort(key=lambda d: d.entered_at, reverse=True)
        return all_dwells[:limit]


class InMemoryKeyframeRepository(KeyframeRepository):
    """In-memory store for tagged keyframes."""

    def __init__(self) -> None:
        self._keyframes: dict[str, TaggedKeyframe] = {}

    async def save_keyframe(self, keyframe: TaggedKeyframe) -> None:
        self._keyframes[keyframe.keyframe_id] = keyframe

    async def get_keyframe(self, keyframe_id: str) -> TaggedKeyframe | None:
        return self._keyframes.get(keyframe_id)

    async def update_retention(self, keyframe_id: str, expires_at: datetime) -> bool:
        keyframe = self._keyframes.get(keyframe_id)
        if keyframe is None:
            return False
        self._keyframes[keyframe_id] = TaggedKeyframe(
            keyframe_id=keyframe.keyframe_id,
            tracklet_id=keyframe.tracklet_id,
            global_track_id=keyframe.global_track_id,
            camera_id=keyframe.camera_id,
            minio_key=keyframe.minio_key,
            captured_at=keyframe.captured_at,
            annotations=keyframe.annotations,
            tag_reason=keyframe.tag_reason,
            expires_at=expires_at,
        )
        return True

    async def list_keyframes(
        self,
        tracklet_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[TaggedKeyframe]:
        results = list(self._keyframes.values())
        if tracklet_id is not None:
            results = [k for k in results if k.tracklet_id == tracklet_id]
        if global_track_id is not None:
            results = [k for k in results if k.global_track_id == global_track_id]
        if after is not None:
            results = [k for k in results if k.captured_at >= after]
        results.sort(key=lambda k: k.captured_at, reverse=True)
        return results[:limit]
