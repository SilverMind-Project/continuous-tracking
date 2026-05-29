"""Postgres implementation of BboxAnnotationRepository.

Uses asyncpg with $N positional placeholders.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

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
                    (keyframe_id, ph_id, camera_id,
                     x1, y1, x2, y2, detection_confidence,
                     frame_width, frame_height, identity_id, created_at,
                     bbox_age_frames)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                ON CONFLICT DO NOTHING
                """,
                [
                    (
                        ann.keyframe_id,
                        ann.ph_id,
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
                    for ann in annotations
                ],
            )

    async def get_bbox_annotations_for_keyframe(self, keyframe_id: str) -> list[BboxAnnotation]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, keyframe_id, ph_id::text, camera_id,
                       x1, y1, x2, y2, detection_confidence,
                       frame_width, frame_height, identity_id, created_at,
                       bbox_age_frames,
                       override_x1, override_y1, override_x2, override_y2,
                       override_by, override_at
                FROM continuous_tracking.keyframe_bbox_annotations
                WHERE keyframe_id = $1
                ORDER BY created_at
                """,
                keyframe_id,
            )
        return [_row_to_domain(r) for r in rows]

    async def get_bbox_annotations_for_ph(self, ph_id: str) -> list[BboxAnnotation]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, keyframe_id, ph_id::text, camera_id,
                       x1, y1, x2, y2, detection_confidence,
                       frame_width, frame_height, identity_id, created_at,
                       bbox_age_frames,
                       override_x1, override_y1, override_x2, override_y2,
                       override_by, override_at
                FROM continuous_tracking.keyframe_bbox_annotations
                WHERE ph_id = $1::uuid
                ORDER BY created_at
                """,
                ph_id,
            )
        return [_row_to_domain(r) for r in rows]

    async def update_identity_id(self, ph_id: str, identity_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE continuous_tracking.keyframe_bbox_annotations
                SET identity_id = $1
                WHERE ph_id = $2::uuid
                """,
                identity_id,
                ph_id,
            )

    async def save_override_bbox(
        self,
        annotation_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
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
                x1,
                y1,
                x2,
                y2,
                override_by,
                datetime.now(UTC),
                annotation_id,
            )

    async def get_annotation_by_id(self, annotation_id: str) -> BboxAnnotation | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, keyframe_id, ph_id::text, camera_id,
                       x1, y1, x2, y2, detection_confidence,
                       frame_width, frame_height, identity_id, created_at,
                       bbox_age_frames,
                       override_x1, override_y1, override_x2, override_y2,
                       override_by, override_at
                FROM continuous_tracking.keyframe_bbox_annotations
                WHERE id = $1::uuid
                """,
                annotation_id,
            )
        return _row_to_domain(row) if row is not None else None

    async def tag_annotation(self, annotation_id: str, identity_id: str | None) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE continuous_tracking.keyframe_bbox_annotations
                SET identity_id = $1
                WHERE id = $2::uuid
                """,
                identity_id,
                annotation_id,
            )

    async def delete_annotation(self, annotation_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM continuous_tracking.keyframe_bbox_annotations
                WHERE id = $1::uuid
                """,
                annotation_id,
            )

    # --- Batch methods ---

    async def delete_annotation_if_exists(self, annotation_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM continuous_tracking.keyframe_bbox_annotations WHERE id = $1::uuid",
                annotation_id,
            )
        # asyncpg returns "DELETE N" — parse the count.
        from contextlib import suppress

        deleted = 0
        with suppress(ValueError, IndexError):
            deleted = int(result.split()[-1])
        return deleted > 0

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        async with self._pool.acquire() as conn, conn.transaction():
            self._tx_conn = conn
            try:
                yield
            finally:
                self._tx_conn = None

    async def apply_bbox_batch(
        self,
        keyframe_id: str,
        operations: list[Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        async with self._pool.acquire() as conn, conn.transaction():
            for op in operations:
                if op.op == "delete" and op.annotation_id:
                    await conn.execute(
                        "DELETE FROM continuous_tracking.keyframe_bbox_annotations "
                        "WHERE id = $1::uuid",
                        op.annotation_id,
                    )
                    results.append({"op": "delete", "annotation_id": op.annotation_id, "ok": True})
                elif op.op == "update" and op.annotation_id and op.data:
                    d = op.data
                    await conn.execute(
                        "UPDATE continuous_tracking.keyframe_bbox_annotations "
                        "SET x1 = $2, y1 = $3, x2 = $4, y2 = $5 WHERE id = $1::uuid",
                        op.annotation_id,
                        float(d.get("x1", 0)),
                        float(d.get("y1", 0)),
                        float(d.get("x2", 0)),
                        float(d.get("y2", 0)),
                    )
                    results.append({"op": "update", "annotation_id": op.annotation_id, "ok": True})
                elif op.op == "create" and op.data:
                    import uuid as _uuid

                    new_id = str(_uuid.uuid4())
                    d = op.data
                    await conn.execute(
                        """INSERT INTO continuous_tracking.keyframe_bbox_annotations
                            (id, keyframe_id, ph_id, camera_id,
                             x1, y1, x2, y2, detection_confidence,
                             frame_width, frame_height, identity_id, created_at)
                            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""",
                        new_id,
                        keyframe_id,
                        "",
                        "",
                        float(d.get("x1", 0)),
                        float(d.get("y1", 0)),
                        float(d.get("x2", 0)),
                        float(d.get("y2", 0)),
                        float(d.get("detection_confidence", 0.5)),
                        int(d.get("frame_width", 0)),
                        int(d.get("frame_height", 0)),
                        d.get("identity_id"),
                        datetime.now(UTC),
                    )
                    results.append({"op": "create", "annotation_id": new_id, "ok": True})
        return results

    async def delete_annotations_below_confidence(self, threshold: float, since: datetime) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM continuous_tracking.keyframe_bbox_annotations "
                "WHERE detection_confidence < $1 AND created_at >= $2",
                threshold,
                since,
            )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0


def _row_to_domain(row: asyncpg.Record) -> BboxAnnotation:
    return BboxAnnotation(
        keyframe_id=row["keyframe_id"],
        ph_id=row["ph_id"],
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
        bbox_age_frames=row["bbox_age_frames"],
        override_x1=row["override_x1"],
        override_y1=row["override_y1"],
        override_x2=row["override_x2"],
        override_y2=row["override_y2"],
        override_by=row["override_by"],
        override_at=row["override_at"],
    )
