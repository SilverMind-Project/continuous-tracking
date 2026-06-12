"""Session-scoped Postgres testcontainer shared by integration and contract tests.

Starts one TimescaleDB container per pytest session. All integration tests
share the same container; table-level truncation between tests ensures
isolation without the overhead of container restarts.
"""

from __future__ import annotations

import atexit
import contextlib
from collections.abc import Generator
from pathlib import Path

import asyncpg
import pytest
from testcontainers.postgres import PostgresContainer

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

_container_ref: PostgresContainer | None = None


def _stop_on_exit() -> None:
    global _container_ref
    if _container_ref is not None:
        with contextlib.suppress(Exception):
            _container_ref.stop()
        _container_ref = None


atexit.register(_stop_on_exit)


@pytest.fixture(scope="session")
def _postgres_container() -> Generator[PostgresContainer, None, None]:
    global _container_ref

    from testcontainers.core.config import testcontainers_config  # type: ignore[import-untyped]

    testcontainers_config.ryuk_disabled = True

    container: PostgresContainer = PostgresContainer("timescale/timescaledb-ha:pg18")
    try:
        container.start()
        _container_ref = container
        yield container
    finally:
        with contextlib.suppress(Exception):
            container.stop()
        _container_ref = None


@pytest.fixture(scope="session")
def postgres_url(_postgres_container: PostgresContainer) -> str:
    """asyncpg-compatible DSN for the session testcontainer."""
    url = _postgres_container.get_connection_url()
    # testcontainers emits postgresql+psycopg2://... ; strip dialect for asyncpg
    return url.replace("postgresql+psycopg2://", "postgresql://", 1)


@pytest.fixture(scope="session")
async def migrated_postgres_url(postgres_url: str) -> str:
    """Run CTS migrations once for the session; return the URL.

    Used by ``db_pool`` (per-test isolation) and by integration subdir
    conftest files that expose it as TEST_DATABASE_URL for Hypothesis tests.
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(postgres_url)
    async with pool.acquire() as conn:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS continuous_tracking")
        # pgvectorscale (vectorscale) provides the diskann access method
        # used in migration 0001_init.
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE")

    from app.storage.migrations import MigrationRunner

    runner = MigrationRunner(pool, _MIGRATIONS_DIR)
    await runner.migrate()
    await pool.close()
    return postgres_url


@pytest.fixture
async def db_pool(migrated_postgres_url: str) -> asyncpg.Pool:
    """Migrated asyncpg pool for one test; truncates CTS tables on teardown.

    Migrations are already applied at session scope by ``migrated_postgres_url``.
    Requesting this fixture implicitly starts the testcontainer.
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(migrated_postgres_url)
    try:
        yield pool
    finally:
        async with pool.acquire() as conn:
            await conn.execute("TRUNCATE continuous_tracking.person_hypotheses CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.world_observations CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.ph_revisions CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.ph_merges CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.gait_bouts CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.gait_daily CASCADE")
            await conn.execute("TRUNCATE continuous_tracking.identities CASCADE")
        await pool.close()
