"""Thread-safe live map of hardware-neutral camera profiles.

Populated at startup and refreshed on every ``CCConfigSyncService`` poll from
Cognitive Companion's camera registry. Stages and services read frozen
``CameraProfile`` values from this map rather than raw CC dicts.
"""

from __future__ import annotations

import asyncio

from app.domain import CameraProfile


class CameraProfileMap:
    """Thread-safe live map of ``camera_id`` to frozen :class:`CameraProfile`.

    Supports both async reads and lock-free synchronous reads (``get_sync``) because
    ``set_all`` atomically swaps the backing dictionary reference.
    """

    def __init__(self, initial: list[CameraProfile] | None = None) -> None:
        self._lock = asyncio.Lock()
        self._profiles: dict[str, CameraProfile] = (
            {p.camera_id: p for p in initial} if initial else {}
        )
        self._version: int = 1 if initial else 0

    async def get(self, camera_id: str) -> CameraProfile | None:
        async with self._lock:
            return self._profiles.get(camera_id)

    def get_sync(self, camera_id: str) -> CameraProfile | None:
        """Return the latest profile for *camera_id* without awaiting the lock."""
        return self._profiles.get(camera_id)

    async def set_all(self, profiles: list[CameraProfile]) -> None:
        async with self._lock:
            self._profiles = {p.camera_id: p for p in profiles}
            self._version += 1

    async def snapshot(self) -> list[CameraProfile]:
        """Return every synced camera profile, ordered by camera id."""
        async with self._lock:
            return sorted(self._profiles.values(), key=lambda p: p.camera_id)

    async def version(self) -> int:
        async with self._lock:
            return self._version

    def version_sync(self) -> int:
        """Return the current profile map version without awaiting the lock."""
        return self._version
