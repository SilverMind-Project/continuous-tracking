"""Thread-safe camera-to-room mapping.

Replaces the static ``settings.yaml.cameras`` dict that was never populated.
Populated by ``CCConfigSyncService`` on every poll cycle from CC's room
registry, which is the single source of truth.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CameraRoomBinding:
    camera_id: str
    room_id: str
    room_name: str
    bound_at: datetime


class CameraRoomMap:
    """Thread-safe live map of camera → room binding.

    Read by pipeline stages (PostureStage, TrajectoryStage, PublishStage)
    to attribute detections to rooms. Written by CCConfigSyncService.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._bindings: dict[str, CameraRoomBinding] = {}
        self._version: int = 0

    async def get(self, camera_id: str) -> CameraRoomBinding | None:
        async with self._lock:
            return self._bindings.get(camera_id)

    async def set_all(self, bindings: list[CameraRoomBinding]) -> None:
        async with self._lock:
            self._bindings = {b.camera_id: b for b in bindings}
            self._version += 1

    async def version(self) -> int:
        async with self._lock:
            return self._version
