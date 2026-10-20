"""Face observation statistics storage: Protocol + InMemory implementation.

Rows describe how large and how well-matched each face result was, per camera, so operators can
size cameras for face recognition from live data. The Postgres peer lives in
``storage/postgres/face_stats_repo.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from ..domain import FaceObservationStat


class FaceObservationStatsRepository(Protocol):
    """Append-only store of face measurement rows."""

    async def append_many(self, rows: Sequence[FaceObservationStat]) -> None:
        """Persist rows. An empty sequence is a no-op."""
        ...

    async def list_between(
        self,
        start: datetime,
        end: datetime,
        camera_id: str | None = None,
    ) -> list[FaceObservationStat]:
        """Rows with ``start <= observed_at < end``, oldest first."""
        ...


class InMemoryFaceObservationStatsRepository:
    """In-memory peer used by unit tests."""

    def __init__(self) -> None:
        self._rows: list[FaceObservationStat] = []

    async def append_many(self, rows: Sequence[FaceObservationStat]) -> None:
        self._rows.extend(rows)

    async def list_between(
        self,
        start: datetime,
        end: datetime,
        camera_id: str | None = None,
    ) -> list[FaceObservationStat]:
        rows = [
            r
            for r in self._rows
            if start <= r.observed_at < end and (camera_id is None or r.camera_id == camera_id)
        ]
        return sorted(rows, key=lambda r: r.observed_at)
