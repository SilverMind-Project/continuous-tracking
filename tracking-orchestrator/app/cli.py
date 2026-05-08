"""``cts-db`` CLI for running database migrations independently of the server.

Usage::

    cts-db migrate          # apply pending migrations
    cts-db rollback -n 1    # roll back last migration
    cts-db status           # show applied and pending

Reads ``database.url`` from ``config/settings.yaml`` (or the path set via
``ORCHESTRATOR_CONFIG_PATH``).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
from structlog import get_logger

from .config import settings
from .storage.migrations import MigrationRunner

logger = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _normalize_dsn(dsn: str) -> str:
    """Strip the ``+asyncpg`` SQLAlchemy scheme prefix if present.

    asyncpg expects a plain ``postgresql://`` or ``postgres://`` DSN.
    """
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("+asyncpg", "", 1)
    return dsn


async def _run(args: argparse.Namespace) -> int:
    dsn = settings.get("database.url", "")
    if not dsn:
        print("database.url must be configured in config/settings.yaml", file=sys.stderr)
        return 1

    dsn = _normalize_dsn(dsn)
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    try:
        runner = MigrationRunner(pool, MIGRATIONS_DIR)
        if args.command == "migrate":
            n = await runner.migrate()
            print(f"Applied {n} migration(s)" if n else "Nothing to apply")
        elif args.command == "rollback":
            n = await runner.rollback(steps=args.steps)
            print(f"Rolled back {n} migration(s)" if n else "Nothing to roll back")
        elif args.command == "status":
            st = await runner.status()
            print(f"Applied ({len(st['applied'])}):")
            for a in st["applied"]:
                print(f"  {a}")
            print(f"Pending ({len(st['pending'])}):")
            for p in st["pending"]:
                print(f"  {p}")
        return 0
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cts-db",
        description="Continuous Tracking — database migration tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="Apply pending up migrations")
    rollback_p = sub.add_parser("rollback", help="Roll back applied migrations")
    rollback_p.add_argument(
        "-n", "--steps", type=int, default=1, help="Number of migrations to roll back"
    )
    sub.add_parser("status", help="Show applied and pending migrations")

    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))
