"""WTR5: TransitDetector tests with shapely-backed polygons."""
from __future__ import annotations

from datetime import UTC, datetime

from app.domain import TransitZone
from app.tracking.world.transit_detector import TransitDetector


def _make_zone(
    zone_id: str = "tz-1",
    polygon: list | None = None,
    direction_vec: tuple[float, float] = (1.0, 0.0),
    inside_room_id: str = "1",
    outside_room_id: str = "2",
) -> TransitZone:
    if polygon is None:
        polygon = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]  # 2x1 rectangle
    return TransitZone(
        zone_id=zone_id,
        name="Test Zone",
        kind="door",
        polygon=[(float(p[0]), float(p[1])) for p in polygon],
        inside_room_id=inside_room_id,
        outside_room_id=outside_room_id,
        direction_vec=direction_vec,
    )


def test_enter_event_fires_once():
    """Entering a zone from outside produces exactly one enter event."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone()

    # First check: outside the zone.
    events = detector.check("ph-1", -1.0, -1.0, [zone], now)
    assert len(events) == 0

    # Second check: now inside the zone.
    events = detector.check("ph-1", 1.0, 0.5, [zone], now)
    assert len(events) == 1
    assert events[0].direction == "enter"
    assert events[0].ph_id == "ph-1"


def test_exit_event_fires_once():
    """Exiting a zone produces exactly one exit event."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone()

    # Enter first.
    detector.check("ph-1", 1.0, 0.5, [zone], now)
    # Exit.
    events = detector.check("ph-1", -1.0, -1.0, [zone], now)
    assert len(events) == 1
    assert events[0].direction == "exit"


def test_jitter_inside_zone_does_not_duplicate():
    """Staying inside a zone across multiple checks does not produce more events."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone()

    # Enter.
    events = detector.check("ph-1", 1.0, 0.5, [zone], now)
    assert len(events) == 1

    # Stay inside (jitter).
    events = detector.check("ph-1", 1.1, 0.4, [zone], now)
    assert len(events) == 0
    events = detector.check("ph-1", 0.9, 0.6, [zone], now)
    assert len(events) == 0


def test_direction_resolves_enter_vs_exit():
    """Moving from outside→inside with positive dot product produces 'enter'."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone(direction_vec=(1.0, 0.0))  # inside direction is +x

    # Move from left to right (positive x displacement → enter).
    detector.check("ph-1", -1.0, 0.5, [zone], now)  # outside
    events = detector.check("ph-1", 1.0, 0.5, [zone], now)  # enter zone
    assert len(events) == 1
    assert events[0].direction == "enter"


def test_remove_ph_clears_state():
    """After remove_ph, the same PH ID starts fresh."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone()

    detector.check("ph-1", 1.0, 0.5, [zone], now)  # enter
    detector.remove_ph("ph-1")

    # After removal, the PH is outside again. Entering should fire again.
    events = detector.check("ph-1", 1.0, 0.5, [zone], now)
    assert len(events) == 1
    assert events[0].direction == "enter"
