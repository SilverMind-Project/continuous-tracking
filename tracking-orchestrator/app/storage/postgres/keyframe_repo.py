"""Postgres implementation of KeyframeRepository.

Uses asyncpg with $N positional placeholders.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from structlog import get_logger

from ...domain import TaggedKeyframe
from ..base import KeyframeRepository

logger = get_logger(__name__)

_SQL_INSERT_KEYFRAME = """
INSERT INTO tagged_keyframes
    (id, tracklet_id, global_track_id, camera_id,
     minio_key, captured_at, annotations, tag_reason, expires_at)
VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9)
ON CONFLICT (id) DO NOTHING
"""

_SQL_LIST_KEYFRAMES = """
SELECT id, tracklet_id, global_track_id, camera_id,
       minio_key, captured_at, annotations, tag_reason, expires_at
FROM tagged_keyframes
WHERE TRUE
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
                keyframe.tracklet_id or None,
                keyframe.global_track_id or None,
                keyframe.camera_id,
                keyframe.minio_key,
                keyframe.captured_at,
                json.dumps(keyframe.annotations),
                keyframe.tag_reason,
                keyframe.expires_at,
            )

    async def list_keyframes(
        self,
        tracklet_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[TaggedKeyframe]:
        sql = _SQL_LIST_KEYFRAMES
        args: list[Any] = []
        n = 1
        if tracklet_id is not None:
            sql += f" AND tracklet_id = ${n}::uuid"
            args.append(tracklet_id)
            n += 1
        if global_track_id is not None:
            sql += f" AND global_track_id = ${n}::uuid"
            args.append(global_track_id)
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
        tracklet_id=str(row["tracklet_id"]) if row["tracklet_id"] else "",
        global_track_id=str(row["global_track_id"]) if row["global_track_id"] else "",
        camera_id=row["camera_id"],
        minio_key=row["minio_key"],
        captured_at=row["captured_at"],
        annotations=annotations,
        tag_reason=row["tag_reason"],
        expires_at=row["expires_at"],
    )
