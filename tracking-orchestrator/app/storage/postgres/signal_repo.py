"""Postgres implementation of DementiaSignalRepository.

Uses asyncpg with $N positional placeholders and datetime.now(UTC)
throughout, consistent with the rest of the Postgres storage layer.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg

from ...domain import DementiaSignal
from ..base import DementiaSignalRepository

# The table is a hypertable with PRIMARY KEY (signal_id, emitted_at): the time
# dimension must be in the key. ``emitted_at`` advances every run, so a plain
# ``ON CONFLICT (signal_id, emitted_at)`` never collides for a re-emitted signal
# and inserts a duplicate row each cycle. Instead update by signal_id alone
# (advancing emitted_at in place — Timescale supports updating the partition
# column), and insert only when absent. This restores the design intent: one row
# per stable signal_id, refreshed in place and still ordered by latest emission.
_SQL_UPDATE_SIGNAL = """
UPDATE continuous_tracking.dementia_signals SET
    severity          = $2,
    value             = $3,
    baseline          = $4,
    z_score           = $5,
    window_start      = $6,
    window_end        = $7,
    context_json      = $8,
    emitted_at        = $9,
    algorithm_version = $10
WHERE signal_id = $1::uuid
"""

_SQL_INSERT_SIGNAL = """
INSERT INTO continuous_tracking.dementia_signals
    (signal_id, identity_id, signal_kind, severity, value,
     baseline, z_score, window_start, window_end, context_json, emitted_at,
     algorithm_version)
VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
ON CONFLICT (signal_id, emitted_at) DO NOTHING
"""

_SQL_LIST_SIGNALS = """
SELECT signal_id, identity_id, signal_kind, severity, value,
       baseline, z_score, window_start, window_end, context_json, emitted_at,
       algorithm_version
FROM continuous_tracking.dementia_signals
WHERE TRUE
"""


class PostgresDementiaSignalRepository(DementiaSignalRepository):
    """Postgres-backed DementiaSignalRepository.

    Requires a connected asyncpg.Pool injected at construction time.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_signal(self, signal: DementiaSignal) -> None:
        context_json = json.dumps(signal.context)
        async with self._pool.acquire() as conn, conn.transaction():
            status = await conn.execute(
                _SQL_UPDATE_SIGNAL,
                signal.signal_id,
                signal.severity,
                signal.value,
                signal.baseline,
                signal.z_score,
                signal.window_start,
                signal.window_end,
                context_json,
                signal.emitted_at,
                signal.algorithm_version,
            )
            # asyncpg returns "UPDATE <n>"; insert only when no row matched.
            if status.rsplit(" ", 1)[-1] == "0":
                await conn.execute(
                    _SQL_INSERT_SIGNAL,
                    signal.signal_id,
                    signal.identity_id,
                    signal.signal_kind,
                    signal.severity,
                    signal.value,
                    signal.baseline,
                    signal.z_score,
                    signal.window_start,
                    signal.window_end,
                    context_json,
                    signal.emitted_at,
                    signal.algorithm_version,
                )

    async def list_signals(
        self,
        identity_id: str | None = None,
        signal_kind: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 200,
    ) -> list[DementiaSignal]:
        sql = _SQL_LIST_SIGNALS
        args: list[Any] = []
        n = 1
        if identity_id is not None:
            sql += f" AND identity_id = ${n}"
            args.append(identity_id)
            n += 1
        if signal_kind is not None:
            sql += f" AND signal_kind = ${n}"
            args.append(signal_kind)
            n += 1
        if after is not None:
            sql += f" AND emitted_at >= ${n}"
            args.append(after)
            n += 1
        if before is not None:
            sql += f" AND emitted_at <= ${n}"
            args.append(before)
            n += 1
        sql += f" ORDER BY emitted_at DESC LIMIT ${n}"
        args.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_signal(r) for r in rows]


def _row_to_signal(row: Any) -> DementiaSignal:
    ctx_raw = row["context_json"]
    ctx: dict[str, Any] = json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw or {})
    return DementiaSignal(
        signal_id=str(row["signal_id"]),
        identity_id=row["identity_id"],
        signal_kind=row["signal_kind"],
        severity=row["severity"],
        value=float(row["value"]),
        baseline=float(row["baseline"]) if row["baseline"] is not None else None,
        z_score=float(row["z_score"]) if row["z_score"] is not None else None,
        window_start=row["window_start"],
        window_end=row["window_end"],
        context=ctx,
        emitted_at=row["emitted_at"],
        algorithm_version=int(row.get("algorithm_version", 1)),
    )
