"""Shared fixtures for integration tests.

All integration tests require TEST_DATABASE_URL and are skipped by default
in ``make check``.  Opt in via ``pytest -m integration`` when a testcontainer
Postgres is available.
"""

from __future__ import annotations

import os

import asyncpg
import pytest


@pytest.fixture
async def db_pool():
    """Create an asyncpg pool, run migrations, and clean up after.

    Requires TEST_DATABASE_URL pointing to a disposable Postgres instance.
    """
    url = os.environ["TEST_DATABASE_URL"]
    pool = await asyncpg.create_pool(url)

    from app.storage.migrations import MigrationRunner

    runner = MigrationRunner(pool)
    await runner.migrate()

    try:
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("TRUNCATE continuous_tracking.person_hypotheses CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.world_observations CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.ph_revisions CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.ph_merges CASCADE")
        await pool.close()
