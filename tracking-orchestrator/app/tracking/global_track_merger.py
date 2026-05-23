"""Merges source GlobalTrack into target GlobalTrack.

All rows that reference source_id are rewritten to reference target_id.
Source is then tombstoned (merged_into_id = target_id).

This is executed inside a single database transaction to prevent partial state.

Uses asyncpg.Pool directly (not a repository) because it needs a single
transaction across multiple tables. This is a deliberate exception to the
repository pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg  # type: ignore[import-untyped]
import structlog

log = structlog.get_logger(__name__)

# Every table in the continuous_tracking schema that has a global_track_id
# column. Built by reading all migration .up.sql files — do not edit without
# re-verifying against the current schema.
_TABLES_WITH_GLOBAL_TRACK_ID: list[tuple[str, str]] = [
    ("continuous_tracking.detections", "global_track_id"),
    ("continuous_tracking.identity_revisions", "global_track_id"),
    ("continuous_tracking.person_trajectories", "global_track_id"),
    ("continuous_tracking.room_dwells", "global_track_id"),
    ("continuous_tracking.tagged_keyframes", "global_track_id"),
    ("continuous_tracking.global_track_identity", "global_track_id"),
    ("continuous_tracking.do_not_fuse_hints", "global_track_id"),
]


class GlobalTrackMerger:
    """Merges source GlobalTrack into target GlobalTrack."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def merge(
        self,
        source_id: str,
        target_id: str,
        merged_by: str,
    ) -> None:
        """Merge source global track into target.

        Raises ValueError if source == target or either does not exist or
        source is already tombstoned.
        """
        if source_id == target_id:
            raise ValueError("Cannot merge a global track with itself")

        async with self._pool.acquire() as conn, conn.transaction():
            # Validate both tracks exist and source is not already merged
            source = await conn.fetchrow(
                "SELECT id, merged_into_id FROM continuous_tracking.global_tracks "
                "WHERE global_track_id = $1::uuid",
                source_id,
            )
            if source is None:
                raise ValueError(f"Source global track {source_id} not found")
            if source["merged_into_id"] is not None:
                raise ValueError(
                    f"Source global track {source_id} is already merged "
                    f"into {source['merged_into_id']}"
                )

            target = await conn.fetchrow(
                "SELECT id FROM continuous_tracking.global_tracks WHERE global_track_id = $1::uuid",
                target_id,
            )
            if target is None:
                raise ValueError(f"Target global track {target_id} not found")

            now = datetime.now(UTC)

            # Rewrite global_track_id in all referencing tables.
            for table, col in _TABLES_WITH_GLOBAL_TRACK_ID:
                await conn.execute(
                    f"UPDATE {table} SET {col} = $1::uuid WHERE {col} = $2::uuid",
                    target_id,
                    source_id,
                )

            # Tombstone the source track
            await conn.execute(
                """
                    UPDATE continuous_tracking.global_tracks
                    SET merged_into_id = $1::uuid, merged_at = $2, merged_by = $3
                    WHERE global_track_id = $4::uuid
                    """,
                target_id,
                now,
                merged_by,
                source_id,
            )

        log.info(
            "global_track_merged",
            source_id=source_id,
            target_id=target_id,
            merged_by=merged_by,
        )
