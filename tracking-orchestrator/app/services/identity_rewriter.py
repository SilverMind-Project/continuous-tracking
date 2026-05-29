"""OrchestratorIdentityRewriter: retroactive cross-table identity relabelling.

When the identity resolver commits a high-confidence face anchor (or when the
committer's buffered window fires a changed decision), past trajectory and dwell
rows written under the old identity_id are relabelled to the new identity_id.
This implements the orchestrator side of CR-13 (the CC side handles
PersonLocationHistory).

Tables rewritten (all in continuous_tracking schema):
  - person_trajectories  — UPDATE identity_id WHERE ph_id + observed_at range
  - room_dwells          — UPDATE identity_id WHERE ph_id + entered_at range
  - dementia_signals     — DELETE old signal + INSERT new signal with new uuid5 PK
                            (PK is derived from identity_id so UPDATE is not viable)

The ``InMemoryIdentityRewriter`` no-ops for unit tests.  The
``PostgresIdentityRewriter`` requires an asyncpg pool wired at startup.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

import asyncpg  # type: ignore[import-untyped]

from ..observability import metrics as _metrics

_SIGNAL_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _stable_signal_id(
    identity_id: str, signal_kind: str, window_start: datetime, window_end: str
) -> str:
    key = f"{identity_id}\x00{signal_kind}\x00{window_start.isoformat()}\x00{window_end}"
    return str(uuid.uuid5(_SIGNAL_NS, key))


class IdentityRewriter(ABC):
    """Protocol for retroactive identity relabelling."""

    @abstractmethod
    async def rewrite(
        self,
        revision_id: str,
        ph_id: str,
        old_identity_id: str | None,
        new_identity_id: str,
        applies_from: datetime,
        applies_to: datetime,
    ) -> None:
        """Relabel all rows in the time window from old_identity_id to new_identity_id.

        Idempotency: ``person_trajectories`` and ``room_dwells`` use ``IS DISTINCT
        FROM`` so a retried rewrite is a no-op once rows are relabelled. The
        ``dementia_signals`` path deletes the old row and inserts a new one whose
        PK is ``uuid5(new_identity_id, kind, window_start, window_end)`` — once the
        old row is gone the retry's SELECT returns 0 rows and the path is a no-op.

        ``revision_id`` is currently advisory (logging/tracing only); it is NOT
        used for deduplication.
        *applies_from* / *applies_to* bound the window of rows to rewrite.
        """


class InMemoryIdentityRewriter(IdentityRewriter):
    """No-op rewriter for unit tests."""

    async def rewrite(
        self,
        revision_id: str,
        ph_id: str,
        old_identity_id: str | None,
        new_identity_id: str,
        applies_from: datetime,
        applies_to: datetime,
    ) -> None:
        pass


class PostgresIdentityRewriter(IdentityRewriter):
    """Postgres-backed rewriter using asyncpg.

    Rewrite happens in a single transaction. If the same revision_id is
    presented again (retry / at-least-once delivery), the UPDATE WHERE clauses
    are idempotent: rows already relabelled are filtered by
    ``identity_id IS DISTINCT FROM new_identity_id``.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def rewrite(
        self,
        revision_id: str,
        ph_id: str,
        old_identity_id: str | None,
        new_identity_id: str,
        applies_from: datetime,
        applies_to: datetime,
    ) -> None:
        if old_identity_id is None or old_identity_id == new_identity_id:
            return

        async with self._pool.acquire() as conn, conn.transaction():
            # -- person_trajectories --
            traj_rows = await conn.execute(
                """
                    UPDATE continuous_tracking.person_trajectories
                    SET identity_id = $1
                    WHERE ph_id = $2
                      AND identity_id IS DISTINCT FROM $1
                      AND observed_at BETWEEN $3 AND $4
                    """,
                new_identity_id,
                ph_id,
                applies_from,
                applies_to,
            )
            traj_count = _parse_rowcount(traj_rows)
            if traj_count > 0:
                _metrics.metrics.revision_rows_rewritten_total.labels(
                    table="person_trajectories"
                ).inc(traj_count)

            # -- room_dwells --
            dwell_rows = await conn.execute(
                """
                    UPDATE continuous_tracking.room_dwells
                    SET identity_id = $1
                    WHERE ph_id = $2
                      AND identity_id IS DISTINCT FROM $1
                      AND entered_at BETWEEN $3 AND $4
                    """,
                new_identity_id,
                ph_id,
                applies_from,
                applies_to,
            )
            dwell_count = _parse_rowcount(dwell_rows)
            if dwell_count > 0:
                _metrics.metrics.revision_rows_rewritten_total.labels(table="room_dwells").inc(
                    dwell_count
                )

            # -- dementia_signals --
            # PK is uuid5(identity_id, signal_kind, window_start, window_end)
            # so we must SELECT, compute new PK, INSERT, DELETE.
            old_signals = await conn.fetch(
                """
                    SELECT signal_id, signal_kind, severity, value, baseline, z_score,
                           window_start, window_end, context, emitted_at, algorithm_version
                    FROM continuous_tracking.dementia_signals
                    WHERE identity_id = $1
                      AND window_start BETWEEN $2 AND $3
                    """,
                old_identity_id,
                applies_from,
                applies_to,
            )
            for row in old_signals:
                window_end_val = row["window_end"]
                if not isinstance(window_end_val, datetime):
                    # Schema invariant: window_end is timestamptz. If this fires
                    # the schema has drifted; surface it loudly.
                    raise RuntimeError(
                        f"dementia_signals.window_end has unexpected type {type(window_end_val)!r}"
                    )

                new_signal_id = _stable_signal_id(
                    new_identity_id,
                    row["signal_kind"],
                    row["window_start"],
                    window_end_val.isoformat(),
                )

                raw_ctx = row["context"]
                if raw_ctx is None:
                    ctx_obj: dict[str, object] = {}
                elif isinstance(raw_ctx, str):
                    try:
                        ctx_obj = json.loads(raw_ctx)
                    except json.JSONDecodeError:
                        ctx_obj = {}
                else:
                    ctx_obj = dict(raw_ctx)
                context_json = json.dumps(ctx_obj)
                await conn.execute(
                    """
                        INSERT INTO continuous_tracking.dementia_signals (
                            signal_id, identity_id, signal_kind, severity, value,
                            baseline, z_score, window_start, window_end,
                            context, emitted_at, algorithm_version
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11,$12)
                        ON CONFLICT (signal_id) DO NOTHING
                        """,
                    new_signal_id,
                    new_identity_id,
                    row["signal_kind"],
                    row["severity"],
                    row["value"],
                    row["baseline"],
                    row["z_score"],
                    row["window_start"],
                    row["window_end"],
                    context_json,
                    row["emitted_at"],
                    row["algorithm_version"],
                )
                await conn.execute(
                    "DELETE FROM continuous_tracking.dementia_signals WHERE signal_id = $1",
                    row["signal_id"],
                )

            signal_count = len(old_signals)
            if signal_count > 0:
                _metrics.metrics.revision_rows_rewritten_total.labels(table="dementia_signals").inc(
                    signal_count
                )


def _parse_rowcount(status: str) -> int:
    """Parse asyncpg command tag like 'UPDATE 5' into the integer row count."""
    try:
        return int(status.split()[-1])
    except (ValueError, IndexError):
        return 0
