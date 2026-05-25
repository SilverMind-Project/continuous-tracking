"""Do-not-fuse hint storage."""

from __future__ import annotations

from typing import Protocol


class DoNotFuseRepository(Protocol):
    """Persist pairs of (tracklet_id, global_track_id) that must never be fused."""

    async def add_hint(
        self, tracklet_id: str, global_track_id: str, created_by: str = "system"
    ) -> None: ...

    async def is_blocked(self, tracklet_id: str, global_track_id: str) -> bool: ...

    async def get_hints_for_tracklet(self, tracklet_id: str) -> list[str]:
        """Returns list of global_track_ids blocked for this tracklet."""
        ...

    async def remove_hint(self, tracklet_id: str, global_track_id: str) -> None:
        """Allow re-fusion (used if caregiver reverses the correction)."""
        ...


class InMemoryDoNotFuseRepository:
    """In-memory store for do-not-fuse hints."""

    def __init__(self) -> None:
        self._hints: set[tuple[str, str]] = set()

    async def add_hint(
        self, tracklet_id: str, global_track_id: str, created_by: str = "system"
    ) -> None:
        self._hints.add((tracklet_id, global_track_id))

    async def is_blocked(self, tracklet_id: str, global_track_id: str) -> bool:
        return (tracklet_id, global_track_id) in self._hints

    async def get_hints_for_tracklet(self, tracklet_id: str) -> list[str]:
        return [gt for (tr, gt) in self._hints if tr == tracklet_id]

    async def remove_hint(self, tracklet_id: str, global_track_id: str) -> None:
        self._hints.discard((tracklet_id, global_track_id))
