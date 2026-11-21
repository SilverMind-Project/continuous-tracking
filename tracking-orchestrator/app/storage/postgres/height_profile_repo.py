"""Postgres implementations of IdentityHeightProfileRepository and PHHeightSampleRepository."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

import asyncpg

from app.domain import HeightProfileSource, IdentityHeightProfile, PHHeightSample

_PROFILE_COLUMNS = """
    identity_id, height_m, height_sigma_m, source, samples, cameras, updated_at
"""

_SAMPLE_COLUMNS = """
    observed_at, ph_id, camera_id, height_m, height_sigma_m
"""


def _profile_from_row(row: asyncpg.Record) -> IdentityHeightProfile:
    return IdentityHeightProfile(
        identity_id=str(row["identity_id"]),
        height_m=float(row["height_m"]),
        height_sigma_m=float(row["height_sigma_m"]),
        source=cast(HeightProfileSource, str(row["source"])),
        samples=int(row["samples"]),
        cameras=int(row["cameras"]),
        updated_at=row["updated_at"],
    )


def _sample_from_row(row: asyncpg.Record) -> PHHeightSample:
    return PHHeightSample(
        ph_id=str(row["ph_id"]),
        camera_id=str(row["camera_id"]),
        observed_at=row["observed_at"],
        height_m=float(row["height_m"]),
        height_sigma_m=float(row["height_sigma_m"]),
    )


class PostgresIdentityHeightProfileRepository:
    """Postgres-backed :class:`IdentityHeightProfileRepository`."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, identity_id: str) -> IdentityHeightProfile | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_PROFILE_COLUMNS} FROM continuous_tracking.identity_height_profiles "
                "WHERE identity_id = $1",
                identity_id,
            )
        return _profile_from_row(row) if row is not None else None

    async def list_all(self) -> list[IdentityHeightProfile]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_PROFILE_COLUMNS} FROM continuous_tracking.identity_height_profiles "
                "ORDER BY identity_id ASC"
            )
        return [_profile_from_row(r) for r in rows]

    async def upsert(self, profile: IdentityHeightProfile) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.identity_height_profiles (
                    identity_id, height_m, height_sigma_m, source, samples, cameras, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (identity_id) DO UPDATE SET
                    height_m = EXCLUDED.height_m,
                    height_sigma_m = EXCLUDED.height_sigma_m,
                    source = EXCLUDED.source,
                    samples = EXCLUDED.samples,
                    cameras = EXCLUDED.cameras,
                    updated_at = EXCLUDED.updated_at
                """,
                profile.identity_id,
                profile.height_m,
                profile.height_sigma_m,
                profile.source,
                profile.samples,
                profile.cameras,
                profile.updated_at,
            )

    async def upsert_operator_if_not_learned(
        self,
        identity_id: str,
        *,
        height_m: float,
        height_sigma_m: float,
        updated_at: datetime,
    ) -> bool:
        async with self._pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM continuous_tracking.identities WHERE identity_id = $1",
                identity_id,
            )
            if not exists:
                return False
            row = await conn.fetchrow(
                """
                INSERT INTO continuous_tracking.identity_height_profiles (
                    identity_id, height_m, height_sigma_m, source, samples, cameras, updated_at
                ) VALUES ($1, $2, $3, 'operator', 0, 0, $4)
                ON CONFLICT (identity_id) DO UPDATE SET
                    height_m = EXCLUDED.height_m,
                    height_sigma_m = EXCLUDED.height_sigma_m,
                    samples = 0,
                    cameras = 0,
                    updated_at = EXCLUDED.updated_at
                WHERE continuous_tracking.identity_height_profiles.source != 'learned'
                RETURNING identity_id
                """,
                identity_id,
                height_m,
                height_sigma_m,
                updated_at,
            )
        return row is not None

    async def delete_operator_profile(self, identity_id: str) -> bool:
        async with self._pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM continuous_tracking.identity_height_profiles "
                "WHERE identity_id = $1 AND source = 'operator'",
                identity_id,
            )
        return res.endswith(" 1")

    async def delete(self, identity_id: str) -> bool:
        async with self._pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM continuous_tracking.identity_height_profiles WHERE identity_id = $1",
                identity_id,
            )
        return res.endswith(" 1")


class PostgresPHHeightSampleRepository:
    """Postgres-backed :class:`PHHeightSampleRepository`."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append_many(self, rows: Sequence[PHHeightSample]) -> None:
        if not rows:
            return
        args: list[tuple[Any, ...]] = [
            (r.observed_at, r.ph_id, r.camera_id, r.height_m, r.height_sigma_m) for r in rows
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"INSERT INTO continuous_tracking.ph_height_samples ({_SAMPLE_COLUMNS}) "
                "VALUES ($1, $2::uuid, $3, $4, $5)",
                args,
            )

    async def list_between(
        self,
        start: datetime,
        end: datetime,
        *,
        ph_id: str | None = None,
        camera_id: str | None = None,
    ) -> list[PHHeightSample]:
        async with self._pool.acquire() as conn:
            records = await conn.fetch(
                f"SELECT {_SAMPLE_COLUMNS} FROM continuous_tracking.ph_height_samples "
                "WHERE observed_at >= $1 AND observed_at < $2 "
                "AND ($3::uuid IS NULL OR ph_id = $3::uuid) "
                "AND ($4::text IS NULL OR camera_id = $4) "
                "ORDER BY observed_at ASC, ph_id ASC, camera_id ASC",
                start,
                end,
                ph_id,
                camera_id,
            )
        return [_sample_from_row(r) for r in records]
