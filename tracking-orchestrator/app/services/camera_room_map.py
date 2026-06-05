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


@dataclass(frozen=True)
class RoomPolygonBinding:
    room_id: str
    room_name: str
    polygon_m: list[tuple[float, float]]
    bound_at: datetime


class RoomPolygonMap:
    """Thread-safe live map of room polygons in floor-plan metre coordinates."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._bindings: dict[str, RoomPolygonBinding] = {}
        self._version: int = 0

    async def set_all(self, bindings: list[RoomPolygonBinding]) -> None:
        async with self._lock:
            self._bindings = {b.room_id: b for b in bindings}
            self._version += 1

    async def snapshot(self) -> tuple[dict[str, list[tuple[float, float]]], dict[str, str]]:
        async with self._lock:
            polygons = {
                room_id: list(binding.polygon_m) for room_id, binding in self._bindings.items()
            }
            names = {room_id: binding.room_name for room_id, binding in self._bindings.items()}
            return polygons, names

    async def version(self) -> int:
        async with self._lock:
            return self._version
