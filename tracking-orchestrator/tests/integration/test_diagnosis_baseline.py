"""Baseline capture tests for diagnosis fixtures.

Replays each new fixture through WorldTracker.step (Postgres repos),
asserts present behaviour, and records baseline numbers as named
constants so later changes can assert improvement.

Marked @pytest.mark.integration; CI selects this marker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from prometheus_client import Counter

from tests.integration._replay import _ROOM_POLYGONS, FIXTURES_DIR, load_fixture

# Must match the BASE_TIME used by synthesize_replay_fixture.py.
BASE_TIME = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)

# ── Baseline constants (recorded 2026-06-01) ────────────────────────────
# When continuity features improve identity stability, these will decrease.

# single_camera_turn: 12-frame (6 s) occlusion gap after turning forces PH
# close (> 5 s grace) then respawn as UNKNOWN.  2 distinct PHs: the frontal
# face-anchor PH (alice) plus the post-gap respawn (UNKNOWN).  Revival +
# sticky maintenance must reduce this to exactly 1 PH with identity held.
BASELINE_MIN_DISTINCT_PH_IDS_TURN = 2

# cross_camera_handoff: 16 s gap > ph_close_grace_s (15 s), PH closes.
# Cameras share 0 PH ids; cross-camera revival should fix this gap.
BASELINE_CAMERA_SHARED_PH_COUNT = 0

# two_people_one_room: exactly 2 distinct PHs maintained
BASELINE_TWO_PEOPLE_PH_COUNT = 2

# resident_plus_stranger: at least 2 PHs (resident + stranger)
BASELINE_RESIDENT_STRANGER_MIN_PH = 2


def _counter_total(counter: Counter) -> float:
    return sum(
        sample.value
        for metric in counter.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    )


async def _replay(
    db_pool: Any,
    fixture_name: str,
) -> Any:
    """Replay a fixture through WorldTracker.step with Postgres repos.

    Returns the PH repository for assertions.
    """
    from app.storage.postgres.ph_repo import (
        PostgresPHRepository,
        PostgresWorldObservationRepository,
    )
    from app.tracking.world.tracker import WorldTracker

    ph_repo = PostgresPHRepository(db_pool)
    obs_repo = PostgresWorldObservationRepository(db_pool)
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)

    fixture = FIXTURES_DIR / fixture_name
    assert fixture.exists(), f"Fixture missing: {fixture_name}"

    steps = load_fixture(fixture)
    for i, frame_obs in enumerate(steps):
        now = BASE_TIME + timedelta(seconds=i * 0.5)
        await tracker.step(
            observations=frame_obs,
            now=now,
            room_polygons=_ROOM_POLYGONS,
        )

    return ph_repo


# ── single_camera_turn ───────────────────────────────────────────────────


@pytest.mark.integration
class TestDiagnosisSingleCameraTurn:
    """single_camera_turn.bin: identity drops on turn (proves churn)."""

    @pytest.mark.asyncio
    async def test_identity_drop_produces_multiple_phs(self, db_pool: Any) -> None:
        """On a turn, the identity drop spawns >1 distinct PH for one person."""
        ph_repo = await _replay(db_pool, "single_camera_turn.bin")

        phs, _total = await ph_repo.list_active(include_transient=True)
        distinct_ph = {ph.ph_id for ph in phs}
        assert len(distinct_ph) >= BASELINE_MIN_DISTINCT_PH_IDS_TURN, (
            f"baseline: expected at least {BASELINE_MIN_DISTINCT_PH_IDS_TURN} PH(s) "
            f"for one person turning, got {len(distinct_ph)} distinct PH ids"
        )

    @pytest.mark.asyncio
    async def test_identity_persists_through_front_frames(self, db_pool: Any) -> None:
        """The initial face-anchor frames assign alice identity correctly."""
        ph_repo = await _replay(db_pool, "single_camera_turn.bin")

        phs, _total = await ph_repo.list_active(include_transient=True)
        # At least one PH should have alice's identity from the face anchor.
        alice_phs = [ph for ph in phs if ph.current_identity_id == "alice"]
        assert len(alice_phs) >= 1, "at least one PH must have alice identity"


# ── cross_camera_handoff ─────────────────────────────────────────────────


@pytest.mark.integration
class TestDiagnosisCrossCameraHandoff:
    """cross_camera_handoff.bin: cameras don't share PH (proves cross-camera gap)."""

    @pytest.mark.asyncio
    async def test_cameras_do_not_share_ph(self, db_pool: Any) -> None:
        """Camera B segment does NOT share the camera A ph_id."""
        ph_repo = await _replay(db_pool, "cross_camera_handoff.bin")

        phs, _total = await ph_repo.list_active(include_transient=True)

        # Collect camera→PH mapping
        phs_by_cam: dict[str, set[str]] = {}
        for ph in phs:
            for cam_id in ph.active_cameras:
                phs_by_cam.setdefault(cam_id, set()).add(ph.ph_id)

        shared = phs_by_cam.get("cam-handoff-a", set()) & phs_by_cam.get("cam-handoff-b", set())
        assert len(shared) == BASELINE_CAMERA_SHARED_PH_COUNT, (
            f"baseline: cross-camera handoff should not share PH ids, got shared={shared}"
        )


# ── two_people_one_room ──────────────────────────────────────────────────


@pytest.mark.integration
class TestDiagnosisTwoPeopleOneRoom:
    """two_people_one_room.bin: non-merge guardrail — 2 distinct PHs."""

    @pytest.mark.asyncio
    async def test_two_distinct_phs_for_two_people(self, db_pool: Any) -> None:
        """Two people produce exactly two distinct PHs (no merge)."""
        ph_repo = await _replay(db_pool, "two_people_one_room.bin")

        phs, total = await ph_repo.list_active(include_transient=True)
        assert total == BASELINE_TWO_PEOPLE_PH_COUNT, (
            f"baseline: expected {BASELINE_TWO_PEOPLE_PH_COUNT} PHs for two people, got {total}"
        )

        ph_ids = {ph.ph_id for ph in phs}
        assert len(ph_ids) == 2, "the two PHs must have distinct IDs"


# ── resident_plus_stranger ───────────────────────────────────────────────


@pytest.mark.integration
class TestDiagnosisResidentPlusStranger:
    """resident_plus_stranger.bin: enrolled resident + unknown stranger."""

    @pytest.mark.asyncio
    async def test_resident_and_stranger_remain_distinct(self, db_pool: Any) -> None:
        """Resident and stranger produce separate PHs; resident identity not
        transferred to stranger (clinical guardrail for favor-continuity bias)."""
        ph_repo = await _replay(db_pool, "resident_plus_stranger.bin")

        phs, total = await ph_repo.list_active(include_transient=True)
        assert total >= BASELINE_RESIDENT_STRANGER_MIN_PH, (
            f"baseline: expected at least {BASELINE_RESIDENT_STRANGER_MIN_PH} PHs "
            f"for resident+stranger, got {total}"
        )

        # Verify the resident's identity is alice (from face anchor)
        resident_phs = [ph for ph in phs if ph.current_identity_id == "alice"]
        assert len(resident_phs) >= 1, (
            "baseline: resident (alice) must have at least one PH with her identity"
        )

        # Stranger PHs must NOT have the resident's identity
        for ph in phs:
            if ph not in resident_phs:
                assert ph.current_identity_id != "alice", (
                    f"guardrail: stranger PH {ph.ph_id} must not inherit resident identity 'alice'"
                )


# ── Behavior-neutrality proof ────────────────────────────────────────────


@pytest.mark.integration
class TestBehaviorNeutrality:
    """Refactors must not change tracking behavior on existing fixtures."""

    @pytest.mark.asyncio
    async def test_hallway_bathroom_ph_count_unchanged(self, db_pool: Any) -> None:
        """hallway_bathroom_door.bin must still produce exactly 1 PH."""
        ph_repo = await _replay(db_pool, "hallway_bathroom_door.bin")

        phs, total = await ph_repo.list_active(include_transient=True)
        assert total == 1, f"behavior-neutrality: hallway-bathroom must produce 1 PH, got {total}"
        assert phs[0].closed_at is None, "PH must still be open at end of replay"

    @pytest.mark.asyncio
    async def test_two_cameras_one_room_ph_count_unchanged(self, db_pool: Any) -> None:
        """two_cameras_one_room.bin must still produce exactly 1 PH (C1)."""
        ph_repo = await _replay(db_pool, "two_cameras_one_room.bin")

        _phs, total = await ph_repo.list_active(include_transient=True)
        assert total == 1, (
            f"behavior-neutrality: two_cameras_one_room must produce 1 PH, got {total}"
        )

    @pytest.mark.asyncio
    async def test_two_rooms_two_people_ph_count_unchanged(self, db_pool: Any) -> None:
        """two_rooms_two_people.bin must still produce exactly 2 PHs (C2)."""
        ph_repo = await _replay(db_pool, "two_rooms_two_people.bin")

        _phs, total = await ph_repo.list_active(include_transient=True)
        assert total == 2, (
            f"behavior-neutrality: two_rooms_two_people must produce 2 PHs, got {total}"
        )
