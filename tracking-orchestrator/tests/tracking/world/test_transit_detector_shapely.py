"""TransitDetector tests with shapely-backed metre-space polygons."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import TransitZone
from app.tracking.world.transit_detector import TransitDetector


def _make_zone(
    zone_id: str = "tz-1",
    polygon: list[tuple[float, float]] | None = None,
    direction_vec: tuple[float, float] = (1.0, 0.0),
    inside_room_id: str = "3",
    outside_room_id: str = "5",
) -> TransitZone:
    if polygon is None:
        polygon = [(4.8, 1.0), (5.2, 1.0), (5.2, 3.0), (4.8, 3.0)]
    return TransitZone(
        zone_id=zone_id,
        name="Bathroom Door",
        kind="door",
        polygon=polygon,
        inside_room_id=inside_room_id,
        outside_room_id=outside_room_id,
        direction_vec=direction_vec,
    )


def test_realistic_metre_units_crossing_fires():
    """A multi-metre floor polygon catches the normalized-vs-metres regression."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone(
        polygon=[(7.8, 4.4), (8.2, 4.4), (8.2, 5.2), (7.8, 5.2)],
        direction_vec=(20.0, 0.0),
    )

    assert detector.check("ph-1", 8.8, 4.8, [zone], now) == []
    events = detector.check("ph-1", 7.4, 4.8, [zone], now)

    assert len(events) == 1
    assert events[0].direction == "enter"


def test_outside_to_inside_emits_one_enter_event():
    """Moving opposite direction_vec enters the inside room."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone()

    assert detector.check("ph-1", 6.0, 2.0, [zone], now) == []
    events = detector.check("ph-1", 4.0, 2.0, [zone], now)
    follow_up = detector.check("ph-1", 3.5, 2.0, [zone], now)

    assert len(events) == 1
    assert events[0].direction == "enter"
    assert events[0].inside_room_id == "3"
    assert events[0].outside_room_id == "5"
    assert follow_up == []


def test_inside_to_outside_emits_one_exit_event():
    """Moving with direction_vec exits to the outside room."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone()

    assert detector.check("ph-1", 4.0, 2.0, [zone], now) == []
    events = detector.check("ph-1", 6.0, 2.0, [zone], now)
    follow_up = detector.check("ph-1", 6.5, 2.0, [zone], now)

    assert len(events) == 1
    assert events[0].direction == "exit"
    assert follow_up == []


def test_lingering_on_threshold_emits_no_event():
    """Sub-threshold movement across the door line is debounced."""
    detector = TransitDetector(min_displacement_m=0.2)
    now = datetime.now(UTC)
    zone = _make_zone()

    assert detector.check("ph-1", 5.05, 2.0, [zone], now) == []
    assert detector.check("ph-1", 4.95, 2.0, [zone], now) == []
    assert detector.check("ph-1", 5.04, 2.0, [zone], now) == []


def test_two_phs_crossing_opposite_directions_emit_correct_events():
    """Per-PH state keeps simultaneous opposite crossings independent."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone()

    assert detector.check("ph-enter", 6.0, 2.0, [zone], now) == []
    assert detector.check("ph-exit", 4.0, 2.5, [zone], now) == []

    enter_events = detector.check("ph-enter", 4.0, 2.0, [zone], now)
    exit_events = detector.check("ph-exit", 6.0, 2.5, [zone], now)

    assert [event.direction for event in enter_events] == ["enter"]
    assert [event.direction for event in exit_events] == ["exit"]


def test_remove_ph_clears_state():
    """After remove_ph, the same PH ID starts fresh."""
    detector = TransitDetector()
    now = datetime.now(UTC)
    zone = _make_zone()

    detector.check("ph-1", 6.0, 2.0, [zone], now)
    assert detector.check("ph-1", 4.0, 2.0, [zone], now)
    detector.remove_ph("ph-1")

    assert detector.check("ph-1", 6.0, 2.0, [zone], now) == []
    events = detector.check("ph-1", 4.0, 2.0, [zone], now)
    assert [event.direction for event in events] == ["enter"]
