"""Storage protocol and InMemory peer for household appearance training samples."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime
from typing import Protocol

from app.domain import AppearanceLabelSource, AppearanceTrainingSample


class AppearanceSampleRepository(Protocol):
    """Durable store for labeled body-appearance training samples (60-day retention)."""

    async def append_many(self, samples: Sequence[AppearanceTrainingSample]) -> int:
        """Persist *samples*, skipping duplicate ``(sample_id, observed_at)`` rows.

        Returns the number of newly inserted rows.
        """
        ...

    async def list_samples(
        self,
        *,
        feature_version: str | None = None,
        since_day: date | None = None,
        until_day: date | None = None,
        include_revoked: bool = False,
        label_sources: frozenset[AppearanceLabelSource] | None = None,
    ) -> list[AppearanceTrainingSample]:
        """Return matching training samples ordered by ``(observed_at ASC, sample_id ASC)``.

        Shared defaults (``include_revoked=False``) are enforced identically across the
        InMemory and Postgres peers so revoked wearable segments never enter training.
        """
        ...

    async def count_negatives_for_day(
        self,
        day: date,
        *,
        feature_version: str | None = None,
    ) -> int:
        """Count non-revoked ``negative`` samples already recorded for local calendar *day*."""
        ...

    async def revoke_by_association(
        self,
        association_id: str,
        *,
        revoked_at: datetime,
    ) -> int:
        """Mark all unrevoked samples sourced from *association_id* as revoked."""
        ...

    async def purge_identity(self, identity_id: str) -> int:
        """Delete all samples belonging to *identity_id* and return the removed count."""
        ...


class InMemoryAppearanceSampleRepository:
    """In-memory peer of :class:`PostgresAppearanceSampleRepository`."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, datetime], AppearanceTrainingSample] = {}
        self._lock = asyncio.Lock()

    async def append_many(self, samples: Sequence[AppearanceTrainingSample]) -> int:
        if not samples:
            return 0
        inserted = 0
        async with self._lock:
            for sample in samples:
                key = (sample.sample_id, sample.observed_at)
                if key not in self._rows:
                    self._rows[key] = sample
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
        async with self._lock:
            rows = [
                row
                for row in self._rows.values()
                if (feature_version is None or row.feature_version == feature_version)
                and (since_day is None or row.day >= since_day)
                and (until_day is None or row.day <= until_day)
                and (include_revoked or row.revoked_at is None)
                and (label_sources is None or row.label_source in label_sources)
            ]
        rows.sort(key=lambda s: (s.observed_at, s.sample_id))
        return rows

    async def count_negatives_for_day(
        self,
        day: date,
        *,
        feature_version: str | None = None,
    ) -> int:
        async with self._lock:
            return sum(
                1
                for row in self._rows.values()
                if row.day == day
                and row.label_source == AppearanceLabelSource.NEGATIVE
                and row.revoked_at is None
                and (feature_version is None or row.feature_version == feature_version)
            )

    async def revoke_by_association(
        self,
        association_id: str,
        *,
        revoked_at: datetime,
    ) -> int:
        updated = 0
        async with self._lock:
            for key, row in list(self._rows.items()):
                if row.seed_association_id == association_id and row.revoked_at is None:
                    self._rows[key] = replace(row, revoked_at=revoked_at)
                    updated += 1
        return updated

    async def purge_identity(self, identity_id: str) -> int:
        async with self._lock:
            doomed = [key for key, row in self._rows.items() if row.identity_id == identity_id]
            for key in doomed:
                del self._rows[key]
            return len(doomed)
