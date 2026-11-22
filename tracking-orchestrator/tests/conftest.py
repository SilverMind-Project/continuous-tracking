"""Session-scoped Postgres testcontainer shared by integration and contract tests.

Starts one TimescaleDB container per pytest session. All integration tests
share the same container; table-level truncation between tests ensures
isolation without the overhead of container restarts.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import socket
from collections.abc import Generator
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from testcontainers.community.postgres import PostgresContainer


@pytest.fixture(autouse=True)
def _block_network_in_unit_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent non-integration tests from opening IP sockets.

    Local dev machines often run Redis/Postgres/MQTT on localhost, which lets
    unmocked clients pass locally and fail in CI where no service containers run
    during unit tests. Tests that need real sockets must carry
    ``@pytest.mark.integration``.
    """
    if request.node.get_closest_marker("integration") is not None:
        return

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_create_connection = asyncio.BaseEventLoop.create_connection

    def _guarded_connect(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(
                f"Unit test attempted outbound network connection to {address!r}. "
                "Mock the client or mark the test @pytest.mark.integration."
            )
        return orig_connect(self, address)

    def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(
                f"Unit test attempted outbound network connection to {address!r}. "
                "Mock the client or mark the test @pytest.mark.integration."
            )
        return orig_connect_ex(self, address)

    async def _guarded_create_connection(
        self: asyncio.BaseEventLoop, *args: Any, **kwargs: Any
    ) -> Any:
        sock = kwargs.get("sock")
        if sock is None or sock.family in (socket.AF_INET, socket.AF_INET6):
            host = args[1] if len(args) > 1 else kwargs.get("host")
            port = args[2] if len(args) > 2 else kwargs.get("port")
            raise RuntimeError(
                f"Unit test attempted outbound async connection to ({host!r}, {port!r}). "
                "Mock the client or mark the test @pytest.mark.integration."
            )
        return await orig_create_connection(self, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", _guarded_create_connection)


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

    # TimescaleDB's job scheduler is off: a background policy job (the 14-day retention on
    # face_observation_stats drops chunks holding fixture rows dated before the window) races the
    # tests' own writes and DDL, deadlocking a migration rollback or deleting rows mid-test. Jobs
    # still register, so tests can assert a policy exists; none runs on its own.
    container: PostgresContainer = PostgresContainer("timescale/timescaledb-ha:pg18").with_command(
        "postgres -c timescaledb.max_background_workers=0"
    )
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
            await conn.execute("TRUNCATE continuous_tracking.face_observation_stats")
            await conn.execute("TRUNCATE continuous_tracking.ph_height_samples")
        await pool.close()
