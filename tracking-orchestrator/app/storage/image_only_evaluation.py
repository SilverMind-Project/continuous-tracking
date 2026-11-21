"""Storage protocol and InMemory peer for daily image-only evaluation totals."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from typing import Protocol

from app.domain import ImageOnlyDailyTotal, WearableAttestationGrade

_Key = tuple[date, str, str, WearableAttestationGrade]


class ImageOnlyEvaluationRepository(Protocol):
    """Durable per-identity, per-day totals of image-only seconds against wearable truth."""

    async def add_totals(self, totals: Sequence[ImageOnlyDailyTotal]) -> None:
        """Add each total's seconds to its (day, identity, arm, grade) row, creating it if new."""
        ...

    async def list_totals(
        self, *, since: date | None = None, until: date | None = None
    ) -> list[ImageOnlyDailyTotal]:
        """Rows with ``since <= day <= until`` (open bounds when None), oldest day first."""
        ...

    async def purge_identity(self, identity_id: str) -> int: ...


def _key(total: ImageOnlyDailyTotal) -> _Key:
    return (total.day, total.identity_id, total.arm, total.attestation_grade)


class InMemoryImageOnlyEvaluationRepository:
    """In-memory peer of :class:`PostgresImageOnlyEvaluationRepository`."""

    def __init__(self) -> None:
        self._rows: dict[_Key, ImageOnlyDailyTotal] = {}
        self._lock = asyncio.Lock()

    async def add_totals(self, totals: Sequence[ImageOnlyDailyTotal]) -> None:
        async with self._lock:
            for total in totals:
                current = self._rows.get(_key(total))
                self._rows[_key(total)] = (
                    total
                    if current is None
                    else replace(
                        current,
                        correct_s=current.correct_s + total.correct_s,
                        wrong_s=current.wrong_s + total.wrong_s,
                        unknown_s=current.unknown_s + total.unknown_s,
                    )
                )

    async def list_totals(
        self, *, since: date | None = None, until: date | None = None
    ) -> list[ImageOnlyDailyTotal]:
        rows = [
            row
            for row in self._rows.values()
            if (since is None or row.day >= since) and (until is None or row.day <= until)
        ]
        rows.sort(key=lambda r: (r.day, r.identity_id, r.arm, str(r.attestation_grade)))
        return rows

    async def purge_identity(self, identity_id: str) -> int:
        async with self._lock:
            doomed = [key for key in self._rows if key[1] == identity_id]
            for key in doomed:
                del self._rows[key]
            return len(doomed)
