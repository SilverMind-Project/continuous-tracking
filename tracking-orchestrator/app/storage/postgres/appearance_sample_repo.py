"""Postgres implementation of :class:`AppearanceSampleRepository`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import asyncpg

from app.domain import AppearanceLabelSource, AppearanceTrainingSample, OrientationBin

_COLUMNS = """
    sample_id, observed_at, day, identity_id, label_source, camera_id,
    orientation, quality, feature_version, embedding::text AS embedding,
    seed_association_id, revoked_at
"""

_SQL_INSERT = """
    INSERT INTO continuous_tracking.appearance_training_samples (
        sample_id, observed_at, day, identity_id, label_source, camera_id,
        orientation, quality, feature_version, embedding,
        seed_association_id, revoked_at
    ) VALUES (
        $1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11::uuid, $12
    )
    ON CONFLICT (sample_id, observed_at) DO NOTHING
"""


def _embedding_to_pgvector(embedding: Sequence[float]) -> str:
    return f"[{','.join(f'{v:.8f}' for v in embedding)}]"


def _pgvector_to_tuple(vector_value: str | Sequence[float] | bytes) -> tuple[float, ...]:
    if isinstance(vector_value, bytes):
        vector_value = vector_value.decode("utf-8")
    if isinstance(vector_value, str):
        inner = vector_value.strip("[]")
        if not inner:
            return ()
        return tuple(float(x) for x in inner.split(",") if x.strip())
    return tuple(float(v) for v in vector_value)


def _sample_from_row(row: asyncpg.Record) -> AppearanceTrainingSample:
    assoc_raw = row["seed_association_id"]
    ident_raw = row["identity_id"]
    return AppearanceTrainingSample(
        sample_id=str(row["sample_id"]),
        observed_at=row["observed_at"],
        day=row["day"],
        identity_id=str(ident_raw) if ident_raw is not None else None,
        label_source=AppearanceLabelSource(str(row["label_source"])),
        camera_id=str(row["camera_id"]),
        orientation=OrientationBin(int(row["orientation"])),
        quality=float(row["quality"]),
        feature_version=str(row["feature_version"]),
        embedding=_pgvector_to_tuple(row["embedding"]),
        seed_association_id=str(assoc_raw) if assoc_raw is not None else None,
        revoked_at=row["revoked_at"],
    )


class PostgresAppearanceSampleRepository:
    """Postgres/TimescaleDB-backed :class:`AppearanceSampleRepository`."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append_many(self, samples: Sequence[AppearanceTrainingSample]) -> int:
        if not samples:
            return 0
        inserted = 0
        async with self._pool.acquire() as conn, conn.transaction():
            for sample in samples:
                result = await conn.execute(
                    _SQL_INSERT,
                    sample.sample_id,
                    sample.observed_at,
                    sample.day,
                    sample.identity_id,
                    str(sample.label_source),
                    sample.camera_id,
                    int(sample.orientation),
                    sample.quality,
                    sample.feature_version,
                    _embedding_to_pgvector(sample.embedding),
                    sample.seed_association_id,
                    sample.revoked_at,
                )
                if str(result).endswith(" 1"):
                    inserted += 1
        return inserted

    async def list_samples(
        self,
        *,
        feature_version: str | None = None,
        since_day: date | None = None,
        until_day: date | None = None,
        include_revoked: bool = False,
        label_sources: frozenset[AppearanceLabelSource] | None = None,
    ) -> list[AppearanceTrainingSample]:
        sources_list = (
            [str(s) for s in sorted(label_sources)] if label_sources is not None else None
        )
        async with self._pool.acquire() as conn:
            # _COLUMNS is a module-level constant; all filter values are $N params.
            rows = await conn.fetch(
                f"""
                SELECT {_COLUMNS}
                FROM continuous_tracking.appearance_training_samples
                WHERE ($1::text IS NULL OR feature_version = $1)
                  AND ($2::date IS NULL OR day >= $2)
                  AND ($3::date IS NULL OR day <= $3)
                  AND ($4::boolean OR revoked_at IS NULL)
                  AND ($5::text[] IS NULL OR label_source = ANY($5::text[]))
                ORDER BY observed_at ASC, sample_id ASC
                """,  # noqa: S608  # module constant _COLUMNS
                feature_version,
                since_day,
                until_day,
                include_revoked,
                sources_list,
            )
        return [_sample_from_row(r) for r in rows]

    async def count_negatives_for_day(
        self,
        day: date,
        *,
        feature_version: str | None = None,
    ) -> int:
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT count(*)
                FROM continuous_tracking.appearance_training_samples
                WHERE day = $1
                  AND label_source = 'negative'
                  AND revoked_at IS NULL
                  AND ($2::text IS NULL OR feature_version = $2)
                """,
                day,
                feature_version,
            )
        return int(count or 0)

    async def revoke_by_association(
        self,
        association_id: str,
        *,
        revoked_at: datetime,
    ) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE continuous_tracking.appearance_training_samples
                SET revoked_at = $2
                WHERE seed_association_id = $1::uuid
                  AND revoked_at IS NULL
                """,
                association_id,
                revoked_at,
            )
        return int(str(result).split()[-1])

    async def purge_identity(self, identity_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM continuous_tracking.appearance_training_samples
                WHERE identity_id = $1
                """,
                identity_id,
            )
        return int(str(result).split()[-1])
