"""Postgres implementation of FaceObservationStatsRepository."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import asyncpg

from ...domain import FaceGateOutcome, FaceObservationStat

_COLUMNS = """
    observed_at, camera_id, detection_id, crop_width_px, crop_height_px,
    face_width_px, face_height_px, interocular_px, det_score, yaw_deg, pitch_deg,
    raw_similarity, recognition_state, best_candidate_id, gate_outcome,
    quality_score, quality_version
"""

_SQL_INSERT = f"""
INSERT INTO continuous_tracking.face_observation_stats ({_COLUMNS})
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
"""

_SQL_LIST = f"""
SELECT {_COLUMNS}
FROM continuous_tracking.face_observation_stats
WHERE observed_at >= $1 AND observed_at < $2
  AND ($3::text IS NULL OR camera_id = $3)
ORDER BY observed_at
"""


class PostgresFaceObservationStatsRepository:
    """Postgres-backed FaceObservationStatsRepository."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append_many(self, rows: Sequence[FaceObservationStat]) -> None:
        if not rows:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(_SQL_INSERT, [_to_args(r) for r in rows])

    async def list_between(
        self,
        start: datetime,
        end: datetime,
        camera_id: str | None = None,
    ) -> list[FaceObservationStat]:
        async with self._pool.acquire() as conn:
            records = await conn.fetch(_SQL_LIST, start, end, camera_id)
        return [_from_record(r) for r in records]


def _to_args(row: FaceObservationStat) -> tuple[Any, ...]:
    return (
        row.observed_at,
        row.camera_id,
        row.detection_id,
        row.crop_width_px,
        row.crop_height_px,
        row.face_width_px,
        row.face_height_px,
        row.interocular_px,
        row.det_score,
        row.yaw_deg,
        row.pitch_deg,
        row.raw_similarity,
        row.recognition_state,
        row.best_candidate_id,
        row.gate_outcome.value,
        row.quality_score,
        row.quality_version,
    )


def _from_record(r: asyncpg.Record) -> FaceObservationStat:
    return FaceObservationStat(
        observed_at=r["observed_at"],
        camera_id=r["camera_id"],
        detection_id=r["detection_id"],
        crop_width_px=r["crop_width_px"],
        crop_height_px=r["crop_height_px"],
        face_width_px=float(r["face_width_px"]),
        face_height_px=float(r["face_height_px"]),
        interocular_px=None if r["interocular_px"] is None else float(r["interocular_px"]),
        det_score=float(r["det_score"]),
        yaw_deg=float(r["yaw_deg"]),
        pitch_deg=float(r["pitch_deg"]),
        raw_similarity=float(r["raw_similarity"]),
        recognition_state=r["recognition_state"],
        best_candidate_id=r["best_candidate_id"],
        gate_outcome=FaceGateOutcome(r["gate_outcome"]),
        quality_score=None if r["quality_score"] is None else float(r["quality_score"]),
        quality_version=r["quality_version"],
    )
