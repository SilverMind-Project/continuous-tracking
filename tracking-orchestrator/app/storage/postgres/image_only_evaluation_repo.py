"""Postgres implementation of :class:`ImageOnlyEvaluationRepository`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import asyncpg

from app.domain import ImageOnlyDailyTotal, WearableAttestationGrade

_SQL_ADD_TOTAL = """
    INSERT INTO continuous_tracking.image_only_evaluation_daily (
        day, identity_id, arm, attestation_grade, correct_s, wrong_s, unknown_s
    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (day, identity_id, arm, attestation_grade) DO UPDATE SET
        correct_s = continuous_tracking.image_only_evaluation_daily.correct_s
                    + EXCLUDED.correct_s,
        wrong_s = continuous_tracking.image_only_evaluation_daily.wrong_s + EXCLUDED.wrong_s,
        unknown_s = continuous_tracking.image_only_evaluation_daily.unknown_s
                    + EXCLUDED.unknown_s,
        updated_at = now()
"""

_SQL_LIST_TOTALS = """
    SELECT day, identity_id, arm, attestation_grade, correct_s, wrong_s, unknown_s
    FROM continuous_tracking.image_only_evaluation_daily
    WHERE ($1::date IS NULL OR day >= $1)
      AND ($2::date IS NULL OR day <= $2)
    ORDER BY day, identity_id, arm, attestation_grade
"""


class PostgresImageOnlyEvaluationRepository:
    """Accumulates daily totals with an additive upsert, so concurrent flushes never lose time."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add_totals(self, totals: Sequence[ImageOnlyDailyTotal]) -> None:
        if not totals:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                _SQL_ADD_TOTAL,
                [
                    (
                        t.day,
                        t.identity_id,
                        t.arm,
                        str(t.attestation_grade),
                        t.correct_s,
                        t.wrong_s,
                        t.unknown_s,
                    )
                    for t in totals
                ],
            )

    async def list_totals(
        self, *, since: date | None = None, until: date | None = None
    ) -> list[ImageOnlyDailyTotal]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_LIST_TOTALS, since, until)
        return [
            ImageOnlyDailyTotal(
                day=row["day"],
                identity_id=str(row["identity_id"]),
                arm=str(row["arm"]),
                attestation_grade=WearableAttestationGrade(str(row["attestation_grade"])),
                correct_s=float(row["correct_s"]),
                wrong_s=float(row["wrong_s"]),
                unknown_s=float(row["unknown_s"]),
            )
            for row in rows
        ]

    async def purge_identity(self, identity_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM continuous_tracking.image_only_evaluation_daily "
                "WHERE identity_id = $1",
                identity_id,
            )
        return int(result.split()[-1])
