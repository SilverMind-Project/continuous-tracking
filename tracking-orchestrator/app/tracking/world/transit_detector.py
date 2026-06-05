"""Detects PH directional crossings through transit zones.

Pure function: no I/O, no DB. Called by the world tracking stage when a
PH floor point updates. Transit zones supplied to this detector must already
be in floor-plan metre coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from shapely.geometry import LineString, Polygon

from ...domain import RoomTransitionEvent, TransitZone


@dataclass
class _PHCrossingState:
    """Mutable per-PH crossing state (not persisted, in-memory only)."""

    ph_id: str
    last_floor: tuple[float, float] | None = None


class TransitDetector:
    """Detects one enter/exit event per directional transit-zone crossing.

    direction_vec points from the inside room toward the outside room.
    Moving opposite that vector means outside-to-inside and emits enter;
    moving with the vector means inside-to-outside and emits exit.
    """

    def __init__(self, min_displacement_m: float = 0.2) -> None:
        self._min_displacement_m = min_displacement_m
        self._states: dict[str, _PHCrossingState] = {}

    def check(
        self,
        ph_id: str,
        floor_x_m: float,
        floor_y_m: float,
        zones: list[TransitZone],
        now: datetime,
    ) -> list[RoomTransitionEvent]:
        """Check *ph_id* against all *zones*. Returns new events (may be empty)."""
        state = self._states.get(ph_id)
        if state is None:
            state = _PHCrossingState(ph_id=ph_id)
            self._states[ph_id] = state

        current = (floor_x_m, floor_y_m)
        prev = state.last_floor
        state.last_floor = current
        if prev is None:
            return []

        movement = (current[0] - prev[0], current[1] - prev[1])
        events: list[RoomTransitionEvent] = []
        segment = LineString([prev, current])
        for zone in zones:
            direction_norm = self._direction_norm(zone.direction_vec)
            if direction_norm == 0.0:
                continue

            directional_displacement = self._projected_displacement_m(
                movement, zone.direction_vec, direction_norm
            )
            if abs(directional_displacement) < self._min_displacement_m:
                continue

            poly = Polygon(zone.polygon)
            if not segment.intersects(poly):
                continue

            prev_side = self._side(prev, zone, poly)
            current_side = self._side(current, zone, poly)
            if prev_side == 0 or current_side == 0 or prev_side == current_side:
                continue

            direction = "enter" if directional_displacement < 0 else "exit"
            events.append(
                RoomTransitionEvent(
                    ph_id=ph_id,
                    transit_zone_id=zone.zone_id,
                    direction=direction,
                    inside_room_id=zone.inside_room_id,
                    outside_room_id=zone.outside_room_id,
                    floor_x_m=floor_x_m,
                    floor_y_m=floor_y_m,
                    event_time=now,
                )
            )

        return events

    @staticmethod
    def _direction_norm(direction_vec: tuple[float, float]) -> float:
        return math.hypot(direction_vec[0], direction_vec[1])

    @staticmethod
    def _projected_displacement_m(
        movement: tuple[float, float],
        direction_vec: tuple[float, float],
        direction_norm: float,
    ) -> float:
        dot = movement[0] * direction_vec[0] + movement[1] * direction_vec[1]
        return dot / direction_norm

    @staticmethod
    def _side(
        point: tuple[float, float], zone: TransitZone, poly: Polygon, epsilon: float = 1e-9
    ) -> int:
        centroid = poly.centroid
        value = (point[0] - centroid.x) * zone.direction_vec[0] + (
            point[1] - centroid.y
        ) * zone.direction_vec[1]
        if abs(value) <= epsilon:
            return 0
        return 1 if value > 0 else -1

    def remove_ph(self, ph_id: str) -> None:
        """Remove per-PH state when a PH is closed."""
        self._states.pop(ph_id, None)
