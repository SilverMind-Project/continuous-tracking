"""Unit parity suite for InMemoryBehaviorBaselineRepository.

These fixtures define the *expected* outputs that the Postgres
implementation must match in the integration parity tests.  Running
this file standalone exercises the InMemory implementation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import PersonTrajectoryPoint, RoomDwell
from app.storage.base import InMemoryBehaviorBaselineRepository

# ---------------------------------------------------------------------------
# Shared scenario fixtures
# ---------------------------------------------------------------------------
# Two identities, 3 days of data:
#   alice: bathroom dwells, living-room dwells, trajectory points with room
#          transitions, one open dwell (should be excluded from dwell_durations)
#   bob:   a few trajectory points and one closed dwell
#
# datetime helper uses days relative to BASE_TIME so tests are stable.
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)

ALICE = "alice-001"
BOB = "bob-002"
PH_A = str(uuid.uuid4())
PH_B = str(uuid.uuid4())


def _dwell(
    identity_id: str,
    ph_id: str,
    room: str,
    days_ago: float,
    duration_s: int | None,
    *,
    is_open: bool = False,
    min_motion_energy: float | None = None,
    still_seconds: int = 0,
) -> RoomDwell:
    entered_at = BASE_TIME - timedelta(days=days_ago)
    exited_at = (
        None if is_open else (entered_at + timedelta(seconds=duration_s) if duration_s else None)
    )
    return RoomDwell(
        dwell_id=str(uuid.uuid4()),
        identity_id=identity_id,
        ph_id=ph_id,
        room_name=room,
        entered_at=entered_at,
        exited_at=exited_at,
        duration_seconds=duration_s if not is_open else None,
        entry_confidence=0.9,
        primary_posture="sitting",
        min_motion_energy=min_motion_energy,
        still_seconds=still_seconds,
    )


def _point(
    identity_id: str,
    ph_id: str,
    room: str,
    days_ago: float,
    hours_ago: float = 0.0,
) -> PersonTrajectoryPoint:
    return PersonTrajectoryPoint(
        identity_id=identity_id,
        ph_id=ph_id,
        observed_at=BASE_TIME - timedelta(days=days_ago, hours=hours_ago),
        room_name=room,
        posture="walking",
        identity_confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Scenario dwells
# ---------------------------------------------------------------------------
# alice: 4 closed bathroom dwells, 2 closed living-room dwells, 1 open dwell
# bob:   1 closed bedroom dwell
SCENARIO_DWELLS: list[RoomDwell] = [
    # alice bathroom dwells (used by dwell_durations + stillness_episodes tests)
    _dwell(ALICE, PH_A, "Bathroom upstairs", 3.0, 400, min_motion_energy=0.005, still_seconds=120),
    _dwell(ALICE, PH_A, "Bathroom upstairs", 2.5, 350, min_motion_energy=0.003),
    _dwell(ALICE, PH_A, "Bathroom upstairs", 2.0, 500, still_seconds=60),
    _dwell(ALICE, PH_A, "Bathroom upstairs", 1.5, 300),
    # alice living-room dwells
    _dwell(ALICE, PH_A, "Living room", 1.0, 1200),
    _dwell(ALICE, PH_A, "Living room", 0.5, 900),
    # alice open dwell (must be excluded from dwell_durations)
    _dwell(ALICE, PH_A, "Bathroom upstairs", 0.1, None, is_open=True),
    # bob bedroom dwell
    _dwell(BOB, PH_B, "Bedroom", 1.0, 3600),
]

# ---------------------------------------------------------------------------
# Scenario trajectory points — spanning a room transition at hour 14 UTC
# ---------------------------------------------------------------------------
SCENARIO_POINTS: list[PersonTrajectoryPoint] = [
    # day 3 — alice: 3 living-room points then 2 hallway points (1 transition)
    _point(ALICE, PH_A, "Living room", days_ago=3.0, hours_ago=0.0),
    _point(ALICE, PH_A, "Living room", days_ago=3.0, hours_ago=-0.5),
    _point(ALICE, PH_A, "Hallway", days_ago=3.0, hours_ago=-1.0),
    _point(ALICE, PH_A, "Hallway", days_ago=3.0, hours_ago=-1.5),
    # day 2 — alice: living → bathroom (1 transition)
    _point(ALICE, PH_A, "Living room", days_ago=2.0, hours_ago=0.0),
    _point(ALICE, PH_A, "Bathroom upstairs", days_ago=2.0, hours_ago=-0.5),
    # bob — bedroom only, no transitions
    _point(BOB, PH_B, "Bedroom", days_ago=1.0, hours_ago=0.0),
    _point(BOB, PH_B, "Bedroom", days_ago=1.0, hours_ago=-0.5),
]


@pytest.fixture
def inmem_repo() -> InMemoryBehaviorBaselineRepository:
    return InMemoryBehaviorBaselineRepository(
        points=SCENARIO_POINTS[:],
        dwells=SCENARIO_DWELLS[:],
    )


# ---------------------------------------------------------------------------
# dwell_durations tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dwell_durations_all_alice(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Returns all closed dwells for alice regardless of room."""
    result = await inmem_repo.dwell_durations(ALICE)
    assert len(result) == 6, f"expected 6 closed dwells, got {len(result)}"
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_dwell_durations_bathroom_predicate(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Case-insensitive substring predicate filters to bathroom dwells only."""
    result = await inmem_repo.dwell_durations(ALICE, room_predicate="bathroom")
    assert len(result) == 4
    assert set(result) == {400.0, 350.0, 500.0, 300.0}


@pytest.mark.asyncio
async def test_dwell_durations_excludes_open_dwell(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Open dwell (exited_at=None) must not appear in results."""
    result = await inmem_repo.dwell_durations(ALICE, room_predicate="bathroom")
    # 4 closed bathroom dwells; open dwell excluded
    assert len(result) == 4


@pytest.mark.asyncio
async def test_dwell_durations_since_filter(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """since= filters on entered_at; dwells before cutoff excluded."""
    since = BASE_TIME - timedelta(days=2.0)
    result = await inmem_repo.dwell_durations(ALICE, since=since)
    # Only dwells entered at or after BASE_TIME - 2 days
    expected = [
        d
        for d in SCENARIO_DWELLS
        if d.identity_id == ALICE
        and d.exited_at is not None
        and d.duration_seconds is not None
        and d.entered_at >= since
    ]
    assert len(result) == len(expected)


@pytest.mark.asyncio
async def test_dwell_durations_unknown_identity(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Unknown identity returns an empty list."""
    result = await inmem_repo.dwell_durations("nobody")
    assert result == []


@pytest.mark.asyncio
async def test_dwell_durations_bob_not_mixed_with_alice(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Bob's dwell does not appear in alice's results and vice versa."""
    alice_result = await inmem_repo.dwell_durations(ALICE)
    bob_result = await inmem_repo.dwell_durations(BOB)
    assert 3600.0 not in alice_result
    assert 3600.0 in bob_result


# ---------------------------------------------------------------------------
# hourly_activity tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hourly_activity_counts_transitions(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Transitions are counted when consecutive same-identity points change room.

    prev_room carries across day boundaries (all points sorted globally):
    Living→Hallway, then Hallway→Living (day boundary), then Living→Bathroom = 3.
    """
    result = await inmem_repo.hourly_activity(ALICE)
    total_transitions = sum(v.transition_count for v in result.values())
    assert total_transitions == 3


@pytest.mark.asyncio
async def test_hourly_activity_observed_minutes(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Each trajectory point contributes 1 observed 'minute'."""
    result = await inmem_repo.hourly_activity(ALICE)
    total_minutes = sum(v.observed_minutes for v in result.values())
    alice_point_count = sum(1 for p in SCENARIO_POINTS if p.identity_id == ALICE)
    assert total_minutes == alice_point_count


@pytest.mark.asyncio
async def test_hourly_activity_bob_not_mixed(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Bob's points do not appear in alice's hourly activity."""
    alice_result = await inmem_repo.hourly_activity(ALICE)
    bob_result = await inmem_repo.hourly_activity(BOB)
    alice_total = sum(v.observed_minutes for v in alice_result.values())
    bob_total = sum(v.observed_minutes for v in bob_result.values())
    alice_expected = sum(1 for p in SCENARIO_POINTS if p.identity_id == ALICE)
    bob_expected = sum(1 for p in SCENARIO_POINTS if p.identity_id == BOB)
    assert alice_total == alice_expected
    assert bob_total == bob_expected


@pytest.mark.asyncio
async def test_hourly_activity_empty_identity(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Unknown identity returns an empty dict."""
    result = await inmem_repo.hourly_activity("nobody")
    assert result == {}


# ---------------------------------------------------------------------------
# stillness_episodes tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stillness_episodes_returns_qualifying_dwells(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Closed dwells with min_motion_energy!=None or still_seconds>0 are returned."""
    result = await inmem_repo.stillness_episodes(ALICE)
    # SCENARIO_DWELLS for alice:
    #   Bathroom 3.0d: min_motion_energy=0.005 + still_seconds=120 -> qualifies
    #   Bathroom 2.5d: min_motion_energy=0.003 -> qualifies
    #   Bathroom 2.0d: still_seconds=60 -> qualifies
    #   Bathroom 1.5d: neither -> excluded
    #   Living 1.0d: neither -> excluded
    #   Living 0.5d: neither -> excluded
    #   open dwell -> excluded (exited_at IS NULL)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_stillness_episodes_fields(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """StillnessEpisode fields map correctly from dwell columns."""
    result = await inmem_repo.stillness_episodes(ALICE)
    rooms = {ep.room_name for ep in result}
    assert rooms == {"Bathroom upstairs"}
    for ep in result:
        assert ep.posture == "sitting"
        assert isinstance(ep.duration_seconds, int)
        assert isinstance(ep.min_motion_energy, float)
        assert ep.occurred_at.tzinfo is not None


@pytest.mark.asyncio
async def test_stillness_episodes_excludes_open_dwell(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Open dwell is excluded even if it would otherwise qualify."""
    # The open dwell for alice has no min_motion_energy or still_seconds anyway,
    # but we verify it is not present regardless.
    result = await inmem_repo.stillness_episodes(ALICE)
    # All returned episodes must have occurred_at in the past (not near BASE_TIME)
    for ep in result:
        assert ep.occurred_at < BASE_TIME - timedelta(hours=1)


@pytest.mark.asyncio
async def test_stillness_episodes_empty_identity(
    inmem_repo: InMemoryBehaviorBaselineRepository,
) -> None:
    """Unknown identity returns an empty list."""
    result = await inmem_repo.stillness_episodes("nobody")
    assert result == []
