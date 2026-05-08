"""Database migration runner with advisory locking for multi-replica safety.

Uses a ``_schema_version`` table to track applied migrations and
``pg_try_advisory_lock`` so only one replica runs migrations at a time.
Supports an ``.up.sql`` / ``.down.sql`` convention for explicit rollback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from structlog import get_logger

logger = get_logger(__name__)

# Stable 64-bit integer for pg_try_advisory_lock.
LOCK_KEY = 0x4354535F4D4947  # "CTS_MIG"


class MigrationRunner:
    """Applies and rolls back numbered SQL migrations via asyncpg.

    Migration files live in *migrations_dir* and follow the convention
    ``NNNN_description.up.sql`` for forward migrations and
    ``NNNN_description.down.sql`` for rollback.  Legacy ``NNNN_description.sql``
    files (without the ``.up`` infix) are also accepted as forward-only
    migrations.
    """

    def __init__(self, pool: Any, migrations_dir: Path) -> None:
        self._pool = pool
        self._dir = migrations_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def migrate(self) -> int:
        """Apply all pending *up* migrations.  Returns the number applied.

        Returns 0 when another process holds the advisory lock — the
        no-op is logged and the caller can proceed safely.
        """
        async with self._pool.acquire() as conn:
            if not await self._try_lock(conn):
                logger.info("Migration lock held by another process; skipping")
                return 0
            try:
                await self._ensure_schema_version(conn)
                applied = 0
                for name in await self._pending(conn):
                    await self._apply_up(conn, name)
                    applied += 1
                if applied:
                    logger.info("Migrations applied", count=applied)
                return applied
            finally:
                await self._unlock(conn)

    async def rollback(self, steps: int = 1) -> int:
        """Roll back the last *steps* applied migrations.  Returns count rolled back."""
        async with self._pool.acquire() as conn:
            if not await self._try_lock(conn):
                logger.info("Migration lock held by another process; skipping")
                return 0
            try:
                await self._ensure_schema_version(conn)
                applied = await self._get_applied(conn)
                to_rollback = list(reversed(applied))[:steps]
                for name in to_rollback:
                    await self._apply_down(conn, name)
                if to_rollback:
                    logger.info("Migrations rolled back", count=len(to_rollback))
                return len(to_rollback)
            finally:
                await self._unlock(conn)

    async def status(self) -> dict[str, list[str]]:
        """Return ``{applied: [...], pending: [...]}`` for inspection."""
        async with self._pool.acquire() as conn:
            await self._ensure_schema_version(conn)
            return {
                "applied": await self._get_applied(conn),
                "pending": await self._pending(conn),
            }

    # ------------------------------------------------------------------
    # Lock helpers
    # ------------------------------------------------------------------

    async def _try_lock(self, conn: Any) -> bool:
        row = await conn.fetchrow("SELECT pg_try_advisory_lock($1)", LOCK_KEY)
        return bool(row[0])

    async def _unlock(self, conn: Any) -> None:
        await conn.execute("SELECT pg_advisory_unlock($1)", LOCK_KEY)

    # ------------------------------------------------------------------
    # Schema version table
    # ------------------------------------------------------------------

    async def _ensure_schema_version(self, conn: Any) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_version (
                filename   TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _up_files(self) -> dict[str, Path]:
        """Return ``{base_name: path}`` for every up-migration found on disk."""
        if not self._dir.is_dir():
            return {}
        result: dict[str, Path] = {}
        for f in sorted(self._dir.iterdir()):
            if not f.name[0].isdigit():
                continue
            if f.name.endswith(".up.sql"):
                base = f.name[: -len(".up.sql")]
            elif f.name.endswith(".down.sql"):
                continue
            elif f.suffix == ".sql":
                base = f.name[: -len(".sql")]
            else:
                continue
            result[base] = f
        return result

    def _down_path(self, base_name: str) -> Path | None:
        """Return the ``.down.sql`` path for *base_name* if it exists."""
        p = self._dir / f"{base_name}.down.sql"
        return p if p.is_file() else None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def _get_applied(self, conn: Any) -> list[str]:
        rows = await conn.fetch("SELECT filename FROM _schema_version ORDER BY filename")
        return [r["filename"] for r in rows]

    async def _pending(self, conn: Any) -> list[str]:
        applied = set(await self._get_applied(conn))
        return [b for b in self._up_files() if b not in applied]

    # ------------------------------------------------------------------
    # Apply / rollback
    # ------------------------------------------------------------------

    @staticmethod
    def _no_transaction(sql: str) -> bool:
        """Return True if the migration declares it must run outside a transaction."""
        for line in sql.splitlines():
            stripped = line.strip()
            if stripped == "-- migrate:no-transaction":
                return True
            if stripped and not stripped.startswith("--"):
                break
        return False

    @staticmethod
    def _split_statements(sql: str) -> list[str]:
        """Split a SQL file into individual statements for non-transactional execution.

        asyncpg sends multi-statement strings via the simple query protocol,
        which PostgreSQL wraps in an implicit transaction — breaking DDL like
        CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous).  Executing
        one statement at a time avoids that implicit transaction.
        """
        stmts = []
        for raw in sql.split(";"):
            stmt = raw.strip()
            # Drop pure-comment blocks that have no executable SQL
            non_comment = "\n".join(
                line for line in stmt.splitlines() if not line.strip().startswith("--")
            ).strip()
            if non_comment:
                stmts.append(stmt)
        return stmts

    async def _apply_up(self, conn: Any, base_name: str) -> None:
        path = self._up_files()[base_name]
        sql = path.read_text()
        logger.info("Applying migration", filename=path.name, base=base_name)
        if self._no_transaction(sql):
            # Execute each statement individually so none runs inside an implicit
            # transaction (required for CREATE MATERIALIZED VIEW WITH timescaledb.continuous).
            for stmt in self._split_statements(sql):
                await conn.execute(stmt)
            await conn.execute("INSERT INTO _schema_version (filename) VALUES ($1)", base_name)
        else:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO _schema_version (filename) VALUES ($1)", base_name
                )
        logger.info("Migration applied", filename=path.name)

    async def _apply_down(self, conn: Any, base_name: str) -> None:
        down = self._down_path(base_name)
        if down is None:
            logger.warning("No down migration found; skipping rollback", base=base_name)
            return
        sql = down.read_text()
        logger.info("Rolling back migration", filename=down.name, base=base_name)
        if self._no_transaction(sql):
            for stmt in self._split_statements(sql):
                await conn.execute(stmt)
            await conn.execute("DELETE FROM _schema_version WHERE filename = $1", base_name)
        else:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "DELETE FROM _schema_version WHERE filename = $1", base_name
                )
        logger.info("Migration rolled back", filename=down.name)
