"""Integration tests for PostgresBehaviorBaselineRepository.

These tests run against a real Postgres testcontainer (migrated schema).
Marked with @pytest.mark.integration so they are skipped during fast
local make check and included in make ci.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from tests.unit.storage.test_baseline_repo_inmemory_parity import (
    ALICE,
    BASE_TIME,
    BOB,
    SCENARIO_DWELLS,
    SCENARIO_POINTS,
    InMemoryBehaviorBaselineRepository,
)

# ---------------------------------------------------------------------------
# Helpers: raw INSERT helpers (use repo methods where they exist)
# ---------------------------------------------------------------------------


async def _insert_trajectory_point(conn: Any, pt: Any) -> None:
    await conn.execute(
        """
        INSERT INTO continuous_tracking.person_trajectories
            (observed_at, identity_id, ph_id, room_name,
             ground_x, ground_y, posture, identity_confidence, motion_energy)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        pt.observed_at,
        pt.identity_id,
        pt.ph_id,
        pt.room_name,
        pt.ground_x,
        pt.ground_y,
        pt.posture,
        pt.identity_confidence,
        pt.motion_energy,
    )


async def _insert_dwell(conn: Any, d: Any) -> None:
    await conn.execute(
        """
        INSERT INTO continuous_tracking.room_dwells
            (identity_id, ph_id, room_name, entered_at, exited_at,
             duration_seconds, entry_confidence, primary_posture,
             activity_summary, min_motion_energy, still_seconds)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
        d.identity_id,
        d.ph_id,
        d.room_name,
        d.entered_at,
        d.exited_at,
        d.duration_seconds,
        d.entry_confidence,
        d.primary_posture,
        json.dumps(d.activity_summary),
        d.min_motion_energy,
        d.still_seconds,
    )


_DUMMY_STATE_MEAN = [0.0, 0.0, 0.0, 0.0]
_DUMMY_STATE_COV = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


async def _ensure_identity(conn: Any, identity_id: str) -> None:
    """Upsert a row into identities so FK constraints are satisfied."""
    await conn.execute(
        """
        INSERT INTO continuous_tracking.identities (identity_id, display_name)
        VALUES ($1, $1)
        ON CONFLICT (identity_id) DO NOTHING
        """,
        identity_id,
    )


async def _ensure_ph(conn: Any, ph_id: str, now: datetime) -> None:
    """Upsert a minimal PersonHypothesis row so FK constraints are satisfied."""
    await conn.execute(
        """
        INSERT INTO continuous_tracking.person_hypotheses
            (ph_id, born_at, last_seen_at, last_seen_camera,
             observation_count, state_mean, state_cov, metadata)
        VALUES ($1, $2, $2, 'test-cam', 1, $3, $4, '{}')
        ON CONFLICT (ph_id) DO NOTHING
        """,
        ph_id,
        now,
        _DUMMY_STATE_MEAN,
        _DUMMY_STATE_COV,
    )


async def _seed_scenario(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        # Ensure identities and PHs exist (FK constraints).
        now = datetime.now(UTC)
        seen_ids: set[str] = set()
        seen_phs: set[str] = set()
        for pt in SCENARIO_POINTS:
            if pt.identity_id and pt.identity_id not in seen_ids:
                await _ensure_identity(conn, pt.identity_id)
                seen_ids.add(pt.identity_id)
            if pt.ph_id not in seen_phs:
                await _ensure_ph(conn, pt.ph_id, now)
                seen_phs.add(pt.ph_id)
        for d in SCENARIO_DWELLS:
            if d.identity_id and d.identity_id not in seen_ids:
                await _ensure_identity(conn, d.identity_id)
                seen_ids.add(d.identity_id)
            if d.ph_id not in seen_phs:
                await _ensure_ph(conn, d.ph_id, now)
                seen_phs.add(d.ph_id)
        for pt in SCENARIO_POINTS:
            await _insert_trajectory_point(conn, pt)
        for d in SCENARIO_DWELLS:
            await _insert_dwell(conn, d)


async def _truncate_tables(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        # Truncate in dependency order; CASCADE handles referencing tables.
        await conn.execute("TRUNCATE continuous_tracking.identities CASCADE")
        await conn.execute("TRUNCATE continuous_tracking.dementia_signals CASCADE")


# ---------------------------------------------------------------------------
# Fixture: seeded pool
# ---------------------------------------------------------------------------


@pytest.fixture
async def seeded_pool(db_pool: asyncpg.Pool) -> asyncpg.Pool:
    """Truncate relevant tables, seed the shared scenario, yield the pool."""
    await _truncate_tables(db_pool)
    await _seed_scenario(db_pool)
    yield db_pool
    await _truncate_tables(db_pool)


# ---------------------------------------------------------------------------
# Test 1: Parity with InMemory
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPostgresBaselineRepoParity:
    """PostgresBehaviorBaselineRepository must match InMemoryBehaviorBaselineRepository
    on all three methods given the same scenario data."""

    @pytest.mark.asyncio
    async def test_dwell_durations_all_alice(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        inmem = InMemoryBehaviorBaselineRepository(
            points=SCENARIO_POINTS[:], dwells=SCENARIO_DWELLS[:]
        )
        pg = PostgresBehaviorBaselineRepository(seeded_pool)

        expected = sorted(await inmem.dwell_durations(ALICE))
        actual = sorted(await pg.dwell_durations(ALICE))
        assert actual == expected

    @pytest.mark.asyncio
    async def test_dwell_durations_bathroom_predicate(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        inmem = InMemoryBehaviorBaselineRepository(
            points=SCENARIO_POINTS[:], dwells=SCENARIO_DWELLS[:]
        )
        pg = PostgresBehaviorBaselineRepository(seeded_pool)

        expected = sorted(await inmem.dwell_durations(ALICE, room_predicate="bathroom"))
        actual = sorted(await pg.dwell_durations(ALICE, room_predicate="bathroom"))
        assert actual == expected

    @pytest.mark.asyncio
    async def test_dwell_durations_since_filter(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        since = BASE_TIME - timedelta(days=2.0)
        inmem = InMemoryBehaviorBaselineRepository(
            points=SCENARIO_POINTS[:], dwells=SCENARIO_DWELLS[:]
        )
        pg = PostgresBehaviorBaselineRepository(seeded_pool)

        expected = sorted(await inmem.dwell_durations(ALICE, since=since))
        actual = sorted(await pg.dwell_durations(ALICE, since=since))
        assert actual == expected

    @pytest.mark.asyncio
    async def test_dwell_durations_unknown_identity(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        pg = PostgresBehaviorBaselineRepository(seeded_pool)
        assert await pg.dwell_durations("nobody") == []

    @pytest.mark.asyncio
    async def test_hourly_activity_transition_counts(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        inmem = InMemoryBehaviorBaselineRepository(
            points=SCENARIO_POINTS[:], dwells=SCENARIO_DWELLS[:]
        )
        pg = PostgresBehaviorBaselineRepository(seeded_pool)

        expected = await inmem.hourly_activity(ALICE)
        actual = await pg.hourly_activity(ALICE)

        exp_transitions = sum(v.transition_count for v in expected.values())
        act_transitions = sum(v.transition_count for v in actual.values())
        assert act_transitions == exp_transitions

        exp_minutes = sum(v.observed_minutes for v in expected.values())
        act_minutes = sum(v.observed_minutes for v in actual.values())
        assert act_minutes == exp_minutes

    @pytest.mark.asyncio
    async def test_hourly_activity_bob_only(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        inmem = InMemoryBehaviorBaselineRepository(
            points=SCENARIO_POINTS[:], dwells=SCENARIO_DWELLS[:]
        )
        pg = PostgresBehaviorBaselineRepository(seeded_pool)

        expected = await inmem.hourly_activity(BOB)
        actual = await pg.hourly_activity(BOB)

        exp_minutes = sum(v.observed_minutes for v in expected.values())
        act_minutes = sum(v.observed_minutes for v in actual.values())
        assert act_minutes == exp_minutes

    @pytest.mark.asyncio
    async def test_hourly_activity_unknown_identity(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        pg = PostgresBehaviorBaselineRepository(seeded_pool)
        assert await pg.hourly_activity("nobody") == {}

    @pytest.mark.asyncio
    async def test_stillness_episodes_count(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        inmem = InMemoryBehaviorBaselineRepository(
            points=SCENARIO_POINTS[:], dwells=SCENARIO_DWELLS[:]
        )
        pg = PostgresBehaviorBaselineRepository(seeded_pool)

        expected = await inmem.stillness_episodes(ALICE)
        actual = await pg.stillness_episodes(ALICE)
        assert len(actual) == len(expected)

    @pytest.mark.asyncio
    async def test_stillness_episodes_fields(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        pg = PostgresBehaviorBaselineRepository(seeded_pool)
        episodes = await pg.stillness_episodes(ALICE)
        for ep in episodes:
            assert ep.room_name == "Bathroom upstairs"
            assert ep.posture == "sitting"
            assert ep.occurred_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_stillness_episodes_unknown_identity(self, seeded_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        pg = PostgresBehaviorBaselineRepository(seeded_pool)
        assert await pg.stillness_episodes("nobody") == []


# ---------------------------------------------------------------------------
# Test 2: End-to-end signal proof (Finding 1 is fixed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBathroomDwellSignalWithPostgresBaseline:
    """Prove that a Postgres-backed DementiaSignalWorker emits a real z_score.

    Seeding 6 closed baseline dwells + 1 very long open dwell triggers the
    bathroom_dwell_anomaly detector with a non-null z_score and baseline.
    Today this test would fail (InMemory baseline always returns no samples).
    """

    @pytest.mark.asyncio
    async def test_signal_carries_z_score_and_baseline(self, db_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository
        from app.storage.postgres.signal_repo import PostgresDementiaSignalRepository
        from app.storage.postgres.trajectory_repo import PostgresTrajectoryRepository
        from app.trajectory.dementia_signals import DementiaSignalWorker, SignalConfig

        await _truncate_tables(db_pool)

        identity_id = "test-resident-" + str(uuid.uuid4())[:8]
        ph_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        # -- Seed 6 closed bathroom dwells ~300 s each (prior days) --
        async with db_pool.acquire() as conn:
            await _ensure_identity(conn, identity_id)
            await _ensure_ph(conn, ph_id, now)
            for day in range(1, 7):
                entered_at = now - timedelta(days=day, hours=2)
                exited_at = entered_at + timedelta(seconds=300)
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.room_dwells
                        (identity_id, ph_id, room_name, entered_at, exited_at,
                         duration_seconds, entry_confidence, primary_posture,
                         activity_summary, min_motion_energy, still_seconds)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    identity_id,
                    ph_id,
                    "Bathroom",
                    entered_at,
                    exited_at,
                    300,
                    0.9,
                    "sitting",
                    json.dumps({}),
                    None,
                    0,
                )

            # -- Seed 1 open bathroom dwell of 4000 s --
            open_entered = now - timedelta(seconds=4000)
            await conn.execute(
                """
                INSERT INTO continuous_tracking.room_dwells
                    (identity_id, ph_id, room_name, entered_at,
                     entry_confidence, primary_posture, activity_summary,
                     min_motion_energy, still_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                identity_id,
                ph_id,
                "Bathroom",
                open_entered,
                0.9,
                "sitting",
                json.dumps({}),
                None,
                0,
            )

            # -- Seed 1 trajectory point (so worker picks up the identity) --
            await conn.execute(
                """
                INSERT INTO continuous_tracking.person_trajectories
                    (observed_at, identity_id, ph_id, room_name,
                     ground_x, ground_y, posture, identity_confidence, motion_energy)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                now - timedelta(minutes=5),
                identity_id,
                ph_id,
                "Bathroom",
                0.0,
                0.0,
                "sitting",
                0.9,
                None,
            )

        trajectory_repo = PostgresTrajectoryRepository(db_pool)
        signal_repo = PostgresDementiaSignalRepository(db_pool)
        baseline_repo = PostgresBehaviorBaselineRepository(db_pool)

        worker = DementiaSignalWorker(
            trajectory_repo=trajectory_repo,
            signal_repo=signal_repo,
            baseline_repo=baseline_repo,
            cfg=SignalConfig(
                onset_consecutive_windows=1,  # emit on first trigger
                min_baseline_n=5,
                bathroom_z_threshold=2.0,  # lower threshold to ensure emission
            ),
        )

        signals = await worker.run_once(now=now)

        bathroom_signals = [s for s in signals if s.signal_kind == "bathroom_dwell_anomaly"]
        assert len(bathroom_signals) >= 1, (
            "Expected bathroom_dwell_anomaly signal — Finding 1 not fixed: "
            "no signal emitted even with 6 closed baseline dwells"
        )
        signal = bathroom_signals[0]
        assert signal.z_score is not None, (
            "z_score must be non-null when baseline repo has 6+ samples"
        )
        assert signal.baseline is not None, (
            "baseline must be non-null when baseline repo has 6+ samples"
        )


# ---------------------------------------------------------------------------
# Test 3: InMemory/Postgres parity for daily_window_rates and pacing_window_rates
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDailyWindowRatesParity:
    """PostgresBehaviorBaselineRepository.daily_window_rates must match InMemory on same data."""

    @pytest.mark.asyncio
    async def test_evening_window_parity(self, db_pool: asyncpg.Pool) -> None:
        from app.domain import PersonTrajectoryPoint
        from app.storage.base import InMemoryBehaviorBaselineRepository
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        await _truncate_tables(db_pool)
        identity_id = "parity-dwr-" + str(uuid.uuid4())[:8]
        ph_id = str(uuid.uuid4())
        now = datetime(2026, 1, 20, 0, 0, 0, tzinfo=UTC)

        # 5 evenings: each has 3 points (kitchen, kitchen, hallway → 1 transition).
        pts: list[PersonTrajectoryPoint] = []
        for day in range(1, 6):
            base = (now - timedelta(days=day)).replace(hour=18, minute=0, second=0, microsecond=0)
            for offset, room in [(0, "kitchen"), (60, "kitchen"), (120, "hallway")]:
                pts.append(
                    PersonTrajectoryPoint(
                        identity_id=identity_id,
                        ph_id=ph_id,
                        observed_at=base + timedelta(minutes=offset),
                        room_name=room,
                        identity_confidence=0.9,
                    )
                )

        async with db_pool.acquire() as conn:
            await _ensure_identity(conn, identity_id)
            await _ensure_ph(conn, ph_id, now)
            for pt in pts:
                await _insert_trajectory_point(conn, pt)

        inmem = InMemoryBehaviorBaselineRepository(points=pts[:])
        pg = PostgresBehaviorBaselineRepository(db_pool)
        since = now - timedelta(days=7)

        expected = await inmem.daily_window_rates(
            identity_id, 17, 22, "UTC", since=since, until=now
        )
        actual = await pg.daily_window_rates(identity_id, 17, 22, "UTC", since=since, until=now)

        assert len(actual) == len(expected), (
            f"row count mismatch: pg={len(actual)} inmem={len(expected)}"
        )
        for act, exp in zip(actual, expected, strict=True):
            assert act.local_date == exp.local_date
            assert act.transition_count == exp.transition_count
            assert act.observed_points == exp.observed_points

    @pytest.mark.asyncio
    async def test_nighttime_wrapping_window_parity(self, db_pool: asyncpg.Pool) -> None:
        from app.domain import PersonTrajectoryPoint
        from app.storage.base import InMemoryBehaviorBaselineRepository
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        await _truncate_tables(db_pool)
        identity_id = "parity-night-" + str(uuid.uuid4())[:8]
        ph_id = str(uuid.uuid4())
        now = datetime(2026, 2, 10, 6, 0, 0, tzinfo=UTC)

        # 4 nights: each night has a 23:30 and a 01:30 point spanning midnight.
        pts: list[PersonTrajectoryPoint] = []
        for day in range(1, 5):
            night_start = (now - timedelta(days=day)).replace(
                hour=23, minute=30, second=0, microsecond=0
            )
            early_morning = (now - timedelta(days=day - 1)).replace(
                hour=1, minute=30, second=0, microsecond=0
            )
            pts.append(
                PersonTrajectoryPoint(
                    identity_id=identity_id,
                    ph_id=ph_id,
                    observed_at=night_start,
                    room_name="bedroom",
                    identity_confidence=0.9,
                )
            )
            pts.append(
                PersonTrajectoryPoint(
                    identity_id=identity_id,
                    ph_id=ph_id,
                    observed_at=early_morning,
                    room_name="kitchen",
                    identity_confidence=0.9,
                )
            )

        async with db_pool.acquire() as conn:
            await _ensure_identity(conn, identity_id)
            await _ensure_ph(conn, ph_id, now)
            for pt in pts:
                await _insert_trajectory_point(conn, pt)

        inmem = InMemoryBehaviorBaselineRepository(points=pts[:])
        pg = PostgresBehaviorBaselineRepository(db_pool)
        since = now - timedelta(days=7)

        expected = await inmem.daily_window_rates(identity_id, 22, 6, "UTC", since=since, until=now)
        actual = await pg.daily_window_rates(identity_id, 22, 6, "UTC", since=since, until=now)

        assert len(actual) == len(expected), (
            f"row count mismatch: pg={len(actual)} inmem={len(expected)}"
        )
        for act, exp in zip(actual, expected, strict=True):
            assert act.local_date == exp.local_date
            assert act.transition_count == exp.transition_count
            assert act.observed_points == exp.observed_points


@pytest.mark.integration
class TestPacingWindowRatesParity:
    """PostgresBehaviorBaselineRepository.pacing_window_rates must match InMemory on same data."""

    @pytest.mark.asyncio
    async def test_dense_windows_parity(self, db_pool: asyncpg.Pool) -> None:
        from app.domain import PersonTrajectoryPoint
        from app.storage.base import InMemoryBehaviorBaselineRepository
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository

        await _truncate_tables(db_pool)
        identity_id = "parity-pacing-" + str(uuid.uuid4())[:8]
        ph_id = str(uuid.uuid4())
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
        since = now - timedelta(days=7)

        # 5 dense 30-min windows (15 pts each, 2 transitions) aligned to since.
        pts: list[PersonTrajectoryPoint] = []
        for w in range(5):
            window_base = since + timedelta(minutes=w * 30)
            for pt_idx in range(15):
                room = "kitchen" if pt_idx < 13 else "hallway" if pt_idx == 13 else "bedroom"
                pts.append(
                    PersonTrajectoryPoint(
                        identity_id=identity_id,
                        ph_id=ph_id,
                        observed_at=window_base + timedelta(minutes=pt_idx * 2),
                        room_name=room,
                        identity_confidence=0.9,
                    )
                )

        async with db_pool.acquire() as conn:
            await _ensure_identity(conn, identity_id)
            await _ensure_ph(conn, ph_id, now)
            for pt in pts:
                await _insert_trajectory_point(conn, pt)

        inmem = InMemoryBehaviorBaselineRepository(points=pts[:])
        pg = PostgresBehaviorBaselineRepository(db_pool)

        expected = sorted(await inmem.pacing_window_rates(identity_id, 30, since=since, until=now))
        actual = sorted(await pg.pacing_window_rates(identity_id, 30, since=since, until=now))

        assert len(actual) == len(expected), (
            f"window count mismatch: pg={len(actual)} inmem={len(expected)}"
        )
        for act, exp in zip(actual, expected, strict=True):
            assert abs(act - exp) < 1e-6, f"rate mismatch: pg={act} inmem={exp}"


# ---------------------------------------------------------------------------
# Test 3: Cold-start still works (empty history)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBathroomDwellColdStart:
    """With no baseline samples the worker must not crash and must use the
    absolute threshold path (z_score=None, severity capped)."""

    @pytest.mark.asyncio
    async def test_cold_start_no_crash(self, db_pool: asyncpg.Pool) -> None:
        from app.storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository
        from app.storage.postgres.signal_repo import PostgresDementiaSignalRepository
        from app.storage.postgres.trajectory_repo import PostgresTrajectoryRepository
        from app.trajectory.dementia_signals import DementiaSignalWorker, SignalConfig

        await _truncate_tables(db_pool)

        identity_id = "cold-start-" + str(uuid.uuid4())[:8]
        ph_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        # Seed only an open dwell (no closed baseline dwells)
        async with db_pool.acquire() as conn:
            await _ensure_identity(conn, identity_id)
            await _ensure_ph(conn, ph_id, now)
            open_entered = now - timedelta(seconds=4000)
            await conn.execute(
                """
                INSERT INTO continuous_tracking.room_dwells
                    (identity_id, ph_id, room_name, entered_at,
                     entry_confidence, primary_posture, activity_summary,
                     min_motion_energy, still_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                identity_id,
                ph_id,
                "Bathroom",
                open_entered,
                0.9,
                "sitting",
                json.dumps({}),
                None,
                0,
            )
            await conn.execute(
                """
                INSERT INTO continuous_tracking.person_trajectories
                    (observed_at, identity_id, ph_id, room_name,
                     ground_x, ground_y, posture, identity_confidence, motion_energy)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                now - timedelta(minutes=5),
                identity_id,
                ph_id,
                "Bathroom",
                0.0,
                0.0,
                "sitting",
                0.9,
                None,
            )

        trajectory_repo = PostgresTrajectoryRepository(db_pool)
        signal_repo = PostgresDementiaSignalRepository(db_pool)
        baseline_repo = PostgresBehaviorBaselineRepository(db_pool)

        worker = DementiaSignalWorker(
            trajectory_repo=trajectory_repo,
            signal_repo=signal_repo,
            baseline_repo=baseline_repo,
            cfg=SignalConfig(
                onset_consecutive_windows=1,
                min_baseline_n=5,
                # Use default bathroom_absolute_threshold_seconds=2700;
                # open dwell is 4000s so it should fire via cold-start path.
            ),
        )

        signals = await worker.run_once(now=now)

        # Cold-start must not crash; if a signal fires it must have z_score=None.
        # (Severity may be demoted to "info" by the quality gate when the window
        # has sparse trajectory points; that is expected behavior.)
        for s in signals:
            if s.signal_kind == "bathroom_dwell_anomaly":
                assert s.z_score is None, (
                    "Cold start must emit with z_score=None (no baseline samples)"
                )
                assert s.baseline is None
