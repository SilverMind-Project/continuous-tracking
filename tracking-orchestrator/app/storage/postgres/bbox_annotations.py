"""Postgres implementation of BboxAnnotationRepository.

Uses asyncpg with $N positional placeholders.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg  # type: ignore[import-untyped]

from ...domain import BboxAnnotation


class PostgresBboxAnnotationRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_bbox_annotations(self, annotations: list[BboxAnnotation]) -> None:
        if not annotations:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO continuous_tracking.keyframe_bbox_annotations
                    (keyframe_id, tracklet_id, camera_id,
                     x1, y1, x2, y2, detection_confidence,
                     frame_width, frame_height, identity_id, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT DO NOTHING
                """,
                [
                    (ann.keyframe_id, ann.tracklet_id, ann.camera_id,
                     ann.x1, ann.y1, ann.x2, ann.y2,
                     ann.detection_confidence, ann.frame_width,
                     ann.frame_height, ann.identity_id, ann.created_at)
                    for ann in annotations
                ],
            )

    async def get_bbox_annotations_for_keyframe(
        self, keyframe_id: str
    ) -> list[BboxAnnotation]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, keyframe_id, tracklet_id::text, camera_id,
                       x1, y1, x2, y2, detection_confidence,
                       frame_width, frame_height, identity_id, created_at,
                       override_x1, override_y1, override_x2, override_y2,
                       override_by, override_at
                FROM continuous_tracking.keyframe_bbox_annotations
                WHERE keyframe_id = $1
                ORDER BY created_at
                """,
                keyframe_id,
            )
        return [_row_to_domain(r) for r in rows]

    async def get_bbox_annotations_for_tracklet(
        self, tracklet_id: str
    ) -> list[BboxAnnotation]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, keyframe_id, tracklet_id::text, camera_id,
                       x1, y1, x2, y2, detection_confidence,
                       frame_width, frame_height, identity_id, created_at,
                       override_x1, override_y1, override_x2, override_y2,
                       override_by, override_at
                FROM continuous_tracking.keyframe_bbox_annotations
                WHERE tracklet_id = $1::uuid
                ORDER BY created_at
                """,
                tracklet_id,
            )
        return [_row_to_domain(r) for r in rows]

    async def update_identity_id(self, tracklet_id: str, identity_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE continuous_tracking.keyframe_bbox_annotations
                SET identity_id = $1
                WHERE tracklet_id = $2::uuid
                """,
                identity_id, tracklet_id,
            )

    async def save_override_bbox(
        self, annotation_id: str,
        x1: float, y1: float, x2: float, y2: float,
        override_by: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE continuous_tracking.keyframe_bbox_annotations
                SET override_x1 = $1, override_y1 = $2,
                    override_x2 = $3, override_y2 = $4,
                    override_by = $5, override_at = $6
                WHERE id = $7::uuid
                """,
                x1, y1, x2, y2, override_by, datetime.now(UTC), annotation_id,
            )


    async def get_annotation_by_id(
        self, annotation_id: str
    ) -> BboxAnnotation | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, keyframe_id, tracklet_id::text, camera_id,
                       x1, y1, x2, y2, detection_confidence,
                       frame_width, frame_height, identity_id, created_at,
                       override_x1, override_y1, override_x2, override_y2,
                       override_by, override_at
                FROM continuous_tracking.keyframe_bbox_annotations
                WHERE id = $1::uuid
                """,
                annotation_id,
            )
        return _row_to_domain(row) if row is not None else None


def _row_to_domain(row: asyncpg.Record) -> BboxAnnotation:
    return BboxAnnotation(
        keyframe_id=row["keyframe_id"],
        tracklet_id=row["tracklet_id"],
        camera_id=row["camera_id"],
        x1=row["x1"],
        y1=row["y1"],
        x2=row["x2"],
        y2=row["y2"],
        detection_confidence=row["detection_confidence"],
        frame_width=row["frame_width"],
        frame_height=row["frame_height"],
        identity_id=row["identity_id"],
        created_at=row["created_at"],
        id=str(row["id"]) if row["id"] is not None else None,
        override_x1=row["override_x1"],
        override_y1=row["override_y1"],
        override_x2=row["override_x2"],
        override_y2=row["override_y2"],
        override_by=row["override_by"],
        override_at=row["override_at"],
    )
