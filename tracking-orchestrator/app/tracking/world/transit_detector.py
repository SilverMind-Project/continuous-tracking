"""Detects PH crossing transit zones (M2).

Pure function: no I/O, no DB. Called by the world tracker or a pipeline
stage when a PH's floor point updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class TransitZone:
    """A door/threshold zone (mirrors domain type for this module)."""

    zone_id: str
    name: str
    kind: str
    polygon: list[tuple[float, float]]
    inside_room_id: str
    outside_room_id: str
    direction_vec: tuple[float, float]


@dataclass(frozen=True)
class RoomTransitionEvent:
    ph_id: str
    transit_zone_id: str
    direction: str  # "enter" | "exit"
    inside_room_id: str
    outside_room_id: str
    floor_x_m: float
    floor_y_m: float
    event_time: datetime


@dataclass
class _PHCrossingState:
    """Mutable per-PH crossing state (not persisted, in-memory only)."""

    ph_id: str
    inside_zone_ids: set[str] = field(default_factory=set)
    last_floor: tuple[float, float] | None = None


class TransitDetector:
    """Detects PH crossing transit zones via polygon containment + direction.

    Maintains per-PH in-memory state to debounce repeated crossings and
    enforce a minimum directional displacement (>= 0.2 m) before firing.
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

        prev = state.last_floor
        state.last_floor = (floor_x_m, floor_y_m)

        events: list[RoomTransitionEvent] = []
        for zone in zones:
            inside_now = self._point_in_polygon(floor_x_m, floor_y_m, zone.polygon)
            was_inside = zone.zone_id in state.inside_zone_ids

            if inside_now and not was_inside:
                # Entered the zone.
                direction = self._resolve_direction(floor_x_m, floor_y_m, prev, zone.direction_vec)
                state.inside_zone_ids.add(zone.zone_id)
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
            elif not inside_now and was_inside:
                # Exited the zone.
                state.inside_zone_ids.discard(zone.zone_id)
                events.append(
                    RoomTransitionEvent(
                        ph_id=ph_id,
                        transit_zone_id=zone.zone_id,
                        direction="exit",
                        inside_room_id=zone.inside_room_id,
                        outside_room_id=zone.outside_room_id,
                        floor_x_m=floor_x_m,
                        floor_y_m=floor_y_m,
                        event_time=now,
                    )
                )

        return events

    def _resolve_direction(
        self,
        fx: float,
        fy: float,
        prev: tuple[float, float] | None,
        direction_vec: tuple[float, float],
    ) -> str:
        """Determine crossing direction: 'enter' or 'exit'.

        Projects the displacement vector onto the zone's direction vector.
        Positive dot product → moving in the 'enter' (inside) direction.
        """
        if prev is None:
            return "enter"
        dx = fx - prev[0]
        dy = fy - prev[1]
        disp = (dx * dx + dy * dy) ** 0.5
        if disp < self._min_displacement_m:
            return "enter"  # not enough displacement; default
        dot = dx * direction_vec[0] + dy * direction_vec[1]
        return "enter" if dot >= 0 else "exit"

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def remove_ph(self, ph_id: str) -> None:
        """Remove per-PH state when a PH is closed."""
        self._states.pop(ph_id, None)
