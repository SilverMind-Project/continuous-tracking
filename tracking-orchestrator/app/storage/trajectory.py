"""Trajectory point, room dwell, and keyframe storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..domain import BboxAnnotation, PersonTrajectoryPoint, RoomDwell, TaggedKeyframe
from .annotations import BboxAnnotationRepository


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
    async def get_open_dwell(self, identity_id: str, ph_id: str) -> RoomDwell | None:
        """Return the open (not-yet-exited) dwell for a track, if any."""

    @abstractmethod
    async def close_dangling_open_dwells(self, closed_at: datetime) -> int:
        """Close every dwell with ``exited_at IS NULL`` (restart reconciliation).

        Each row's ``exited_at`` is set to its last observed trajectory point
        (falling back to ``entered_at``), bounded by ``closed_at``.  Returns the
        number of rows closed.
        """

    @abstractmethod
    async def list_trajectory_points(
        self,
        identity_id: str | None = None,
        ph_id: str | None = None,
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
        ph_id: str | None = None,
        before: datetime | None = None,
    ) -> list[RoomDwell]:
        """List room dwell intervals with optional filters.

        ``ph_id`` and ``before`` (identity-continuity M04) let a caller select
        one PH's dwells within an explicit ``[after, before]`` window, as the
        internal dwell-range endpoint needs; ``after``/``before`` filter on
        ``entered_at``.
        """


class KeyframeRepository(ABC):
    """Persist tagged keyframes."""

    @abstractmethod
    async def save_keyframe(self, keyframe: TaggedKeyframe) -> None:
        """Store a tagged keyframe."""

    @abstractmethod
    async def save_keyframe_with_bbox_annotations(
        self,
        keyframe: TaggedKeyframe,
        bbox_annotations: list[BboxAnnotation],
    ) -> None:
        """Store a tagged keyframe and its bbox evidence atomically when possible."""

    @abstractmethod
    async def get_keyframe(self, keyframe_id: str) -> TaggedKeyframe | None:
        """Retrieve a tagged keyframe by ID."""

    @abstractmethod
    async def update_retention(self, keyframe_id: str, expires_at: datetime) -> bool:
        """Update keyframe retention expiry. Returns True if the row existed."""

    @abstractmethod
    async def list_keyframes(
        self,
        ph_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[TaggedKeyframe]:
        """List keyframes with optional filters."""

    @abstractmethod
    async def list_for_read_model(
        self,
        *,
        camera_id: str | None = None,
        tag_reason: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 5000,
    ) -> list[TaggedKeyframe]:
        """List keyframes for the M07 physical-frame read model.

        Returns trigger rows matching the frame-level filters (camera, tag
        reason, capture window) ordered by capture time descending. The read
        model groups these into physical-frame cards, so all rows for a given
        source frame must be returned together; identity-level filters are
        applied after grouping.
        """


class InMemoryTrajectoryRepository(TrajectoryRepository):
    """In-memory store for trajectory points and room dwells."""

    def __init__(self) -> None:
        self._points: list[PersonTrajectoryPoint] = []
        self._open_dwells: dict[tuple[str | None, str], RoomDwell] = {}
        self._closed_dwells: list[RoomDwell] = []

    async def save_trajectory_point(self, point: PersonTrajectoryPoint) -> None:
        self._points.append(point)

    async def save_room_dwell(self, dwell: RoomDwell) -> None:
        key = (dwell.identity_id, dwell.ph_id)
        self._open_dwells[key] = dwell

    async def update_room_dwell(self, dwell: RoomDwell) -> None:
        key = (dwell.identity_id, dwell.ph_id)
        self._open_dwells.pop(key, None)
        self._closed_dwells.append(dwell)

    async def get_open_dwell(self, identity_id: str, ph_id: str) -> RoomDwell | None:
        return self._open_dwells.get((identity_id, ph_id))

    async def close_dangling_open_dwells(self, closed_at: datetime) -> int:
        from dataclasses import replace

        last_obs_by_ph: dict[str, datetime] = {}
        for p in self._points:
            prev = last_obs_by_ph.get(p.ph_id)
            if prev is None or p.observed_at > prev:
                last_obs_by_ph[p.ph_id] = p.observed_at

        count = 0
        for key, dwell in list(self._open_dwells.items()):
            exit_at = min(last_obs_by_ph.get(dwell.ph_id, dwell.entered_at), closed_at)
            if exit_at < dwell.entered_at:
                exit_at = dwell.entered_at
            self._open_dwells.pop(key, None)
            self._closed_dwells.append(
                replace(
                    dwell,
                    exited_at=exit_at,
                    duration_seconds=int((exit_at - dwell.entered_at).total_seconds()),
                )
            )
            count += 1
        return count

    async def list_trajectory_points(
        self,
        identity_id: str | None = None,
        ph_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[PersonTrajectoryPoint]:
        results = list(self._points)
        if identity_id is not None:
            results = [p for p in results if p.identity_id == identity_id]
        if ph_id is not None:
            results = [p for p in results if p.ph_id == ph_id]
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
        ph_id: str | None = None,
        before: datetime | None = None,
    ) -> list[RoomDwell]:
        all_dwells = list(self._open_dwells.values()) + self._closed_dwells
        if identity_id is not None:
            all_dwells = [d for d in all_dwells if d.identity_id == identity_id]
        if room_name is not None:
            all_dwells = [d for d in all_dwells if d.room_name == room_name]
        if after is not None:
            all_dwells = [d for d in all_dwells if d.entered_at >= after]
        if ph_id is not None:
            all_dwells = [d for d in all_dwells if d.ph_id == ph_id]
        if before is not None:
            all_dwells = [d for d in all_dwells if d.entered_at <= before]
        all_dwells.sort(key=lambda d: d.entered_at, reverse=True)
        return all_dwells[:limit]


class InMemoryKeyframeRepository(KeyframeRepository):
    """In-memory store for tagged keyframes."""

    def __init__(self, bbox_repo: BboxAnnotationRepository | None = None) -> None:
        self._keyframes: dict[str, TaggedKeyframe] = {}
        self._bbox_repo = bbox_repo

    async def save_keyframe(self, keyframe: TaggedKeyframe) -> None:
        self._keyframes[keyframe.keyframe_id] = keyframe

    async def save_keyframe_with_bbox_annotations(
        self,
        keyframe: TaggedKeyframe,
        bbox_annotations: list[BboxAnnotation],
    ) -> None:
        await self.save_keyframe(keyframe)
        if self._bbox_repo is not None and bbox_annotations:
            await self._bbox_repo.save_bbox_annotations(bbox_annotations)

    async def get_keyframe(self, keyframe_id: str) -> TaggedKeyframe | None:
        return self._keyframes.get(keyframe_id)

    async def update_retention(self, keyframe_id: str, expires_at: datetime) -> bool:
        keyframe = self._keyframes.get(keyframe_id)
        if keyframe is None:
            return False
        self._keyframes[keyframe_id] = TaggedKeyframe(
            keyframe_id=keyframe.keyframe_id,
            ph_id=keyframe.ph_id,
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
        ph_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[TaggedKeyframe]:
        results = list(self._keyframes.values())
        if ph_id is not None:
            results = [k for k in results if k.ph_id == ph_id]
        if after is not None:
            results = [k for k in results if k.captured_at >= after]
        results.sort(key=lambda k: k.captured_at, reverse=True)
        return results[:limit]

    async def list_for_read_model(
        self,
        *,
        camera_id: str | None = None,
        tag_reason: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 5000,
    ) -> list[TaggedKeyframe]:
        results = list(self._keyframes.values())
        if camera_id is not None:
            results = [k for k in results if k.camera_id == camera_id]
        if tag_reason is not None:
            results = [k for k in results if k.tag_reason == tag_reason]
        if after is not None:
            results = [k for k in results if k.captured_at >= after]
        if before is not None:
            results = [k for k in results if k.captured_at <= before]
        results.sort(key=lambda k: k.captured_at, reverse=True)
        return results[:limit]
