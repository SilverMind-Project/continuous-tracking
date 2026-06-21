"""Postgres implementation of KeyframeRepository.

Uses asyncpg with $N positional placeholders.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from structlog import get_logger

from ...domain import BboxAnnotation, TaggedKeyframe
from ..base import KeyframeRepository

logger = get_logger(__name__)

_SQL_INSERT_KEYFRAME = """
INSERT INTO continuous_tracking.tagged_keyframes
    (id, ph_id, camera_id,
     minio_key, captured_at, annotations, tag_reason, expires_at)
VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8)
ON CONFLICT (id) DO NOTHING
"""

_SQL_LIST_KEYFRAMES = """
SELECT id, ph_id, camera_id,
       minio_key, captured_at, annotations, tag_reason, expires_at
FROM continuous_tracking.tagged_keyframes
WHERE TRUE
"""

_SQL_GET_KEYFRAME = """
SELECT id, ph_id, camera_id,
       minio_key, captured_at, annotations, tag_reason, expires_at
FROM continuous_tracking.tagged_keyframes
WHERE id = $1::uuid
"""

_SQL_UPDATE_RETENTION = """
UPDATE continuous_tracking.tagged_keyframes
SET expires_at = $2
WHERE id = $1::uuid
"""


class PostgresKeyframeRepository(KeyframeRepository):
    """Postgres-backed KeyframeRepository."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_keyframe(self, keyframe: TaggedKeyframe) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_INSERT_KEYFRAME,
                keyframe.keyframe_id,
                keyframe.ph_id or None,
                keyframe.camera_id,
                keyframe.minio_key,
                keyframe.captured_at,
                json.dumps(keyframe.annotations),
                keyframe.tag_reason,
                keyframe.expires_at,
            )

    async def save_keyframe_with_bbox_annotations(
        self,
        keyframe: TaggedKeyframe,
        bbox_annotations: list[BboxAnnotation],
    ) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                _SQL_INSERT_KEYFRAME,
                keyframe.keyframe_id,
                keyframe.ph_id or None,
                keyframe.camera_id,
                keyframe.minio_key,
                keyframe.captured_at,
                json.dumps(keyframe.annotations),
                keyframe.tag_reason,
                keyframe.expires_at,
            )
            if bbox_annotations:
                await conn.executemany(
                    """
                    INSERT INTO continuous_tracking.keyframe_bbox_annotations
                        (keyframe_id, ph_id, camera_id,
                         x1, y1, x2, y2, detection_confidence,
                         frame_width, frame_height, identity_id, created_at,
                         bbox_age_frames)
                    VALUES ($1,$2::uuid,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        (
                            ann.keyframe_id,
                            ann.ph_id or None,
                            ann.camera_id,
                            ann.x1,
                            ann.y1,
                            ann.x2,
                            ann.y2,
                            ann.detection_confidence,
                            ann.frame_width,
                            ann.frame_height,
                            ann.identity_id,
                            ann.created_at,
                            ann.bbox_age_frames,
                        )
                        for ann in bbox_annotations
                    ],
                )

    async def get_keyframe(self, keyframe_id: str) -> TaggedKeyframe | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_KEYFRAME, keyframe_id)
        return _row_to_keyframe(row) if row is not None else None

    async def update_retention(self, keyframe_id: str, expires_at: datetime) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(_SQL_UPDATE_RETENTION, keyframe_id, expires_at)
        return not result.endswith(" 0")

    async def list_for_read_model(
        self,
        *,
        camera_id: str | None = None,
        tag_reason: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 5000,
    ) -> list[TaggedKeyframe]:
        sql = _SQL_LIST_KEYFRAMES
        args: list[Any] = []
        n = 1
        if camera_id is not None:
            sql += f" AND camera_id = ${n}"
            args.append(camera_id)
            n += 1
        if tag_reason is not None:
            sql += f" AND tag_reason = ${n}"
            args.append(tag_reason)
            n += 1
        if after is not None:
            sql += f" AND captured_at >= ${n}"
            args.append(after)
            n += 1
        if before is not None:
            sql += f" AND captured_at <= ${n}"
            args.append(before)
            n += 1
        sql += f" ORDER BY captured_at DESC LIMIT ${n}"
        args.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_keyframe(r) for r in rows]

    async def list_keyframes(
        self,
        ph_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[TaggedKeyframe]:
        sql = _SQL_LIST_KEYFRAMES
        args: list[Any] = []
        n = 1
        if ph_id is not None:
            sql += f" AND ph_id = ${n}::uuid"
            args.append(ph_id)
            n += 1
        if after is not None:
            sql += f" AND captured_at >= ${n}"
            args.append(after)
            n += 1
        sql += f" ORDER BY captured_at DESC LIMIT ${n}"
        args.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_keyframe(r) for r in rows]


def _row_to_keyframe(row: Any) -> TaggedKeyframe:
    annotations_raw = row["annotations"]
    annotations: dict[str, Any] = (
        json.loads(annotations_raw)
        if isinstance(annotations_raw, str)
        else dict(annotations_raw or {})
    )
    return TaggedKeyframe(
        keyframe_id=str(row["id"]),
        ph_id=str(row["ph_id"]) if row["ph_id"] else "",
        camera_id=row["camera_id"],
        minio_key=row["minio_key"],
        captured_at=row["captured_at"],
        annotations=annotations,
        tag_reason=row["tag_reason"],
        expires_at=row["expires_at"],
    )
