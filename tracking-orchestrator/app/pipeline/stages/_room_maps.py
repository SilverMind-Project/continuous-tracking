"""Helpers for stages that consume CC-synced room maps."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...services.camera_room_map import CameraRoomMap, RoomPolygonMap


async def camera_room_name(camera_room_map: CameraRoomMap, camera_id: str) -> str | None:
    """Return the CC-synced room name for a camera, or None when unbound."""
    binding = await camera_room_map.get(camera_id)
    return binding.room_name if binding is not None else None


async def camera_room_names(
    camera_room_map: CameraRoomMap, camera_ids: Iterable[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for camera_id in camera_ids:
        room_name = await camera_room_name(camera_room_map, camera_id)
        if room_name:
            result[camera_id] = room_name
    return result


async def room_polygon_snapshot(
    room_polygon_map: RoomPolygonMap,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, str]]:
    """Return room polygons and display names from the CC-synced polygon map."""
    polygons_raw, names_raw = await room_polygon_map.snapshot()
    polygons = {str(k): list(v) for k, v in polygons_raw.items()}
    names = {str(k): str(v) for k, v in names_raw.items()}
    return polygons, names
