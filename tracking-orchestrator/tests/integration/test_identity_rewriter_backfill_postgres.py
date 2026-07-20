"""Postgres proof of the NULL-only backfill invariant (identity-continuity M04).

Proves ``PostgresIdentityRewriter.backfill_null_rows`` relabels only
identity-NULL ``person_trajectories``/``room_dwells`` rows in the window and
never touches a row that already carries a (possibly different) identity.
Marked @pytest.mark.integration; CI selects this marker against a
testcontainer (``make ci``).
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from app.services.identity_rewriter import PostgresIdentityRewriter

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import rollback_backfill  # noqa: E402

pytestmark = pytest.mark.integration

T0 = datetime(2026, 7, 20, 8, 0, 0, tzinfo=UTC)

_DUMMY_STATE_MEAN = [0.0, 0.0, 0.0, 0.0]
_DUMMY_STATE_COV = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


async def _ensure_identity(conn: Any, identity_id: str) -> None:
    await conn.execute(
        """
        INSERT INTO continuous_tracking.identities (identity_id, display_name)
        VALUES ($1, $1)
        ON CONFLICT (identity_id) DO NOTHING
        """,
        identity_id,
    )


async def _ensure_ph(conn: Any, ph_id: str, now: datetime) -> None:
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


async def _insert_trajectory_point(
    conn: Any, ph_id: str, identity_id: str | None, observed_at: datetime
) -> None:
    await conn.execute(
        """
        INSERT INTO continuous_tracking.person_trajectories
            (observed_at, identity_id, ph_id, room_name, ground_x, ground_y,
             posture, identity_confidence)
        VALUES ($1, $2, $3, 'living_room', 0.0, 0.0, 'standing', 0.5)
        """,
        observed_at,
        identity_id,
        ph_id,
    )


async def _insert_dwell(
    conn: Any, ph_id: str, identity_id: str | None, entered_at: datetime
) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO continuous_tracking.room_dwells
            (identity_id, ph_id, room_name, entered_at, exited_at, entry_confidence)
        VALUES ($1, $2, 'living_room', $3, $3, 0.5)
        RETURNING id
        """,
        identity_id,
        ph_id,
        entered_at,
    )
    return int(row["id"])


async def _truncate(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE continuous_tracking.identities CASCADE")


@pytest.mark.asyncio
async def test_backfill_null_rows_fills_only_null(db_pool: asyncpg.Pool) -> None:
    await _truncate(db_pool)
    ph_id = str(uuid.uuid4())
    async with db_pool.acquire() as conn:
        await _ensure_identity(conn, "alice")
        await _ensure_identity(conn, "carol")
        await _ensure_ph(conn, ph_id, T0)

        # NULL row inside the window: must be relabelled.
        await _insert_trajectory_point(conn, ph_id, None, T0 + timedelta(minutes=30))
        # Non-NULL row inside the window (already labelled "carol" by a prior,
        # unrelated commit): must be untouched -- this is the invariant test.
        await _insert_trajectory_point(conn, ph_id, "carol", T0 + timedelta(minutes=45))

        await _insert_dwell(conn, ph_id, None, T0 + timedelta(minutes=20))
        await _insert_dwell(conn, ph_id, "carol", T0 + timedelta(minutes=50))

    rewriter = PostgresIdentityRewriter(db_pool)
    await rewriter.backfill_null_rows("rev-1", ph_id, "alice", T0, T0 + timedelta(hours=1))

    async with db_pool.acquire() as conn:
        traj_rows = await conn.fetch(
            "SELECT identity_id FROM continuous_tracking.person_trajectories "
            "WHERE ph_id = $1 ORDER BY observed_at",
            ph_id,
        )
        dwell_rows = await conn.fetch(
            "SELECT identity_id FROM continuous_tracking.room_dwells "
            "WHERE ph_id = $1 ORDER BY entered_at",
            ph_id,
        )

    assert [r["identity_id"] for r in traj_rows] == ["alice", "carol"]
    assert [r["identity_id"] for r in dwell_rows] == ["alice", "carol"]


@pytest.mark.asyncio
async def test_backfill_window_bounds_inclusive_exclusive(db_pool: asyncpg.Pool) -> None:
    await _truncate(db_pool)
    ph_id = str(uuid.uuid4())
    window_start = T0
    window_end = T0 + timedelta(hours=1)
    async with db_pool.acquire() as conn:
        await _ensure_identity(conn, "alice")
        await _ensure_ph(conn, ph_id, T0)

        # Exactly on the boundaries: BETWEEN is inclusive on both ends.
        await _insert_trajectory_point(conn, ph_id, None, window_start)
        await _insert_trajectory_point(conn, ph_id, None, window_end)
        # Outside the window: must not be touched.
        await _insert_trajectory_point(conn, ph_id, None, window_start - timedelta(seconds=1))
        await _insert_trajectory_point(conn, ph_id, None, window_end + timedelta(seconds=1))

    rewriter = PostgresIdentityRewriter(db_pool)
    await rewriter.backfill_null_rows("rev-1", ph_id, "alice", window_start, window_end)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT observed_at, identity_id FROM continuous_tracking.person_trajectories "
            "WHERE ph_id = $1 ORDER BY observed_at",
            ph_id,
        )

    by_time = {r["observed_at"]: r["identity_id"] for r in rows}
    assert by_time[window_start - timedelta(seconds=1)] is None
    assert by_time[window_start] == "alice"
    assert by_time[window_end] == "alice"
    assert by_time[window_end + timedelta(seconds=1)] is None


@pytest.mark.asyncio
async def test_rollback_script_restores_null(
    db_pool: asyncpg.Pool, migrated_postgres_url: str
) -> None:
    """scripts/rollback_backfill.py restores identity_id to NULL (M04)."""
    await _truncate(db_pool)
    ph_id = str(uuid.uuid4())
    revision_id = str(uuid.uuid4())
    window_start = T0
    window_end = T0 + timedelta(hours=3)

    async with db_pool.acquire() as conn:
        await _ensure_identity(conn, "alice")
        await _ensure_identity(conn, "carol")
        await _ensure_ph(conn, ph_id, T0)

        # Simulate what UnknownBackfillService._apply already did: NULL rows
        # relabelled to "alice", plus an unrelated already-labelled row that
        # must survive the rollback untouched.
        await _insert_trajectory_point(conn, ph_id, "alice", window_start + timedelta(minutes=10))
        await _insert_dwell(conn, ph_id, "alice", window_start + timedelta(minutes=5))
        await _insert_trajectory_point(conn, ph_id, "carol", window_end + timedelta(minutes=10))

        await conn.execute(
            """
            INSERT INTO continuous_tracking.identity_revision_ranges
                (range_id, revision_id, ph_id, effective_identity_id, authority,
                 range_start, range_end)
            VALUES ($1::uuid, $2::uuid, $3::uuid, $4, 'inferred', $5, $6)
            """,
            str(uuid.uuid4()),
            revision_id,
            ph_id,
            "alice",
            window_start,
            window_end,
        )

    # Dry run must not write.
    dry_report = await rollback_backfill.run_rollback(
        migrated_postgres_url, revision_id, apply=False
    )
    assert dry_report["person_trajectories_matched"] == 1
    assert dry_report["room_dwells_matched"] == 1
    async with db_pool.acquire() as conn:
        still_alice = await conn.fetchval(
            "SELECT COUNT(*) FROM continuous_tracking.person_trajectories "
            "WHERE ph_id = $1::uuid AND identity_id = 'alice'",
            ph_id,
        )
    assert still_alice == 1

    # Apply: restores the "alice" rows to NULL, leaves "carol" untouched.
    apply_report = await rollback_backfill.run_rollback(
        migrated_postgres_url, revision_id, apply=True
    )
    assert apply_report["person_trajectories_matched"] == 1
    assert apply_report["room_dwells_matched"] == 1

    async with db_pool.acquire() as conn:
        traj_rows = await conn.fetch(
            "SELECT identity_id FROM continuous_tracking.person_trajectories "
            "WHERE ph_id = $1::uuid ORDER BY observed_at",
            ph_id,
        )
        dwell_rows = await conn.fetch(
            "SELECT identity_id FROM continuous_tracking.room_dwells WHERE ph_id = $1::uuid",
            ph_id,
        )

    assert [r["identity_id"] for r in traj_rows] == [None, "carol"]
    assert [r["identity_id"] for r in dwell_rows] == [None]
