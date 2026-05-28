"""WTR5: Room resolution from floor point and polygons."""

from __future__ import annotations

from app.tracking.world.helpers import is_in_any_room_polygon, resolve_room


def test_point_inside_room_gets_room_id():
    """A point inside a known polygon gets the room id."""
    polygons = {
        "living_room": [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)],
        "kitchen": [(5.0, 0.0), (8.0, 0.0), (8.0, 3.0), (5.0, 3.0)],
    }
    camera_map = {"cam-1": "living_room", "cam-2": "kitchen"}

    room_id, room_name = resolve_room(2.5, 2.0, "cam-1", polygons, camera_map)
    assert room_id == "living_room"
    assert room_name == "living_room"


def test_point_outside_all_rooms_falls_back_to_camera_map():
    """A point outside all polygons falls back to camera→room map for display."""
    polygons = {
        "living_room": [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)],
    }
    camera_map = {"cam-1": "living_room"}

    # Point is far outside the room.
    room_id, room_name = resolve_room(100.0, 100.0, "cam-1", polygons, camera_map)
    # Falls back to camera map.
    assert room_id == "living_room"
    assert room_name == "living_room"


def test_is_in_any_room_polygon():
    """Points inside/outside room polygons are correctly classified."""
    polygons = {
        "r1": [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
    }

    assert is_in_any_room_polygon(1.5, 1.5, polygons) is True
    assert is_in_any_room_polygon(10.0, 10.0, polygons) is False
    assert is_in_any_room_polygon(1.5, 1.5, {}) is False
