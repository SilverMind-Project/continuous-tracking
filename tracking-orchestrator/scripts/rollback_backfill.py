#!/usr/bin/env python3
"""Roll back a single Unknown-segment backfill (identity-continuity M04).

Restores ``identity_id`` to NULL for ``person_trajectories``/``room_dwells``
rows that a specific ``inferred_backfill`` revision relabelled. Scoped
strictly to the physical rows the backfill wrote: it does not retire the
underlying ``identity_revision_ranges`` row (the effective-identity overlay
`GalleryRepository`-style read models join against). If the overlay must also
stop reporting the backfilled label, use the existing operator-correction
``IdentityCorrectionService.compensate()`` flow, which already supersedes
ranges correctly; this script only undoes the M04 row relabel.

Default mode: dry-run (reports what would change, writes nothing). Use
--apply to execute.

Required env var: DATABASE_URL (postgresql+asyncpg://... or postgresql://...)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import asyncpg

log = logging.getLogger("rollback_backfill")


def _dsn_from_env() -> str:
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return raw.replace("postgresql+asyncpg://", "postgresql://")


async def _find_inferred_range(conn: asyncpg.Connection, revision_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT range_id, ph_id, effective_identity_id, range_start, range_end
        FROM continuous_tracking.identity_revision_ranges
        WHERE revision_id = $1 AND authority = 'inferred'
        """,
        revision_id,
    )
    return dict(row) if row is not None else None


async def run_rollback(dsn: str, revision_id: str, apply: bool) -> dict[str, Any]:
    conn = await asyncpg.connect(dsn)
    try:
        range_row = await _find_inferred_range(conn, revision_id)
        if range_row is None:
            raise SystemExit(
                f"No inferred revision range found for revision_id={revision_id!r}. "
                "Only inferred_backfill revisions can be rolled back by this script."
            )

        ph_id = str(range_row["ph_id"])
        identity_id = range_row["effective_identity_id"]
        start = range_row["range_start"]
        end = range_row["range_end"]

        if identity_id is None:
            raise SystemExit(
                f"revision_id={revision_id!r} has no effective_identity_id; nothing to roll back."
            )

        traj_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM continuous_tracking.person_trajectories
            WHERE ph_id = $1::uuid AND identity_id = $2
              AND observed_at BETWEEN $3 AND $4
            """,
            ph_id,
            identity_id,
            start,
            end,
        )
        dwell_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM continuous_tracking.room_dwells
            WHERE ph_id = $1::uuid AND identity_id = $2
              AND entered_at BETWEEN $3 AND $4
            """,
            ph_id,
            identity_id,
            start,
            end,
        )

        report: dict[str, Any] = {
            "revision_id": revision_id,
            "ph_id": ph_id,
            "identity_id": identity_id,
            "range_start": start.isoformat(),
            "range_end": end.isoformat(),
            "dry_run": not apply,
            "person_trajectories_matched": int(traj_count),
            "room_dwells_matched": int(dwell_count),
        }

        if apply:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE continuous_tracking.person_trajectories
                    SET identity_id = NULL
                    WHERE ph_id = $1::uuid AND identity_id = $2
                      AND observed_at BETWEEN $3 AND $4
                    """,
                    ph_id,
                    identity_id,
                    start,
                    end,
                )
                await conn.execute(
                    """
                    UPDATE continuous_tracking.room_dwells
                    SET identity_id = NULL
                    WHERE ph_id = $1::uuid AND identity_id = $2
                      AND entered_at BETWEEN $3 AND $4
                    """,
                    ph_id,
                    identity_id,
                    start,
                    end,
                )
            log.info(
                "rollback applied: revision_id=%s ph_id=%s rows_restored=%d",
                revision_id,
                ph_id,
                traj_count + dwell_count,
            )
        else:
            log.info("DRY RUN -- no writes performed; pass --apply to execute")

        return report
    finally:
        await conn.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Roll back one Unknown-segment backfill revision. Default: dry-run.",
    )
    p.add_argument("revision_id", help="The inferred_backfill revision_id to roll back.")
    p.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Execute the rollback. Without this flag the script only reports counts.",
    )
    p.add_argument(
        "--report",
        metavar="PATH",
        default="",
        help="Write machine-readable JSON report to this path.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    try:
        dsn = _dsn_from_env()
        report = asyncio.run(run_rollback(dsn, args.revision_id, args.apply))
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        log.exception("rollback failed")
        return 1

    output = json.dumps(report, indent=2, default=str)
    print(output)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
