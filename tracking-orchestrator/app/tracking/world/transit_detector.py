"""Detects PH crossing transit zones (M2, updated WTR5).

Pure function: no I/O, no DB. Called by the world tracker or a pipeline
stage when a PH's floor point updates.

WTR5: Uses shapely.geometry.Polygon for point-in-polygon containment
instead of hand-rolled ray-casting. TransitZone and RoomTransitionEvent
are imported from the domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from shapely.geometry import Point, Polygon

from ...domain import RoomTransitionEvent, TransitZone


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
    Uses shapely.geometry.Polygon for containment tests.
    """

    def __init__(self, min_displacement_m: float = 0.2) -> None:
        self._min_displacement_m = min_displacement_m
        self._states: dict[str, _PHCrossingState] = {}
        # Cache shapely prepared geometries per zone for performance.
        self._zone_polygons: dict[str, Polygon] = {}

    def _get_zone_polygon(self, zone: TransitZone) -> Polygon:
        """Return a cached shapely Polygon for *zone*."""
        if zone.zone_id not in self._zone_polygons:
            self._zone_polygons[zone.zone_id] = Polygon(zone.polygon)
        return self._zone_polygons[zone.zone_id]

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

        point = Point(floor_x_m, floor_y_m)
        events: list[RoomTransitionEvent] = []
        for zone in zones:
            poly = self._get_zone_polygon(zone)
            inside_now = poly.contains(point)
            was_inside = zone.zone_id in state.inside_zone_ids

            if inside_now and not was_inside:
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
            return "enter"
        dot = dx * direction_vec[0] + dy * direction_vec[1]
        return "enter" if dot >= 0 else "exit"

    def remove_ph(self, ph_id: str) -> None:
        """Remove per-PH state when a PH is closed."""
        self._states.pop(ph_id, None)
