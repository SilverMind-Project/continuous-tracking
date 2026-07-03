"""Postgres/TimescaleDB implementation of the GalleryRepository protocol.

Handles identities, gallery embeddings, and ANN search via pgvector HNSW.
"""

from __future__ import annotations

import json
from datetime import datetime

import asyncpg
from structlog import get_logger

from ...domain import GalleryEmbedding, Identity, ReviewCandidate, ReviewEvent
from ..base import GalleryRepository
from ..gallery import ReviewConflictError, ReviewNotFoundError

logger = get_logger(__name__)

# Columns the M09 review queue projects from reid_gallery.
_REVIEW_COLUMNS = """
    id, identity_id, proposed_identity_id, effective_identity_id, state,
    label_source, candidate_reason, model_version, preprocessing_version,
    dimension, crop_key, source_frame_key, crop_hash, frame_hash, bbox,
    crop_width, crop_height, ph_id, observation_id, keyframe_id, camera_id,
    capture_time, confidence, orientation, quality, is_truncated, is_occluded,
    source_episode_id, created_actor, created_at, seen_at, reviewed_actor,
    reviewed_time, review_reason, review_note, audit_version
"""

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_SQL_UPSERT_IDENTITY = """
    INSERT INTO continuous_tracking.identities (identity_id, display_name, metadata, is_active)
    VALUES ($1, $2, $3::jsonb, $4)
    ON CONFLICT (identity_id) DO UPDATE SET
        display_name = EXCLUDED.display_name,
        metadata = EXCLUDED.metadata,
        is_active = EXCLUDED.is_active,
        updated_at = now()
"""

_SQL_GET_IDENTITY = """
    SELECT identity_id, display_name, metadata, is_active, enrolled_at
    FROM continuous_tracking.identities
    WHERE identity_id = $1
"""

_SQL_LIST_IDENTITIES_ACTIVE = """
    SELECT identity_id, display_name, metadata, is_active, enrolled_at
    FROM continuous_tracking.identities
    WHERE is_active = true
    ORDER BY enrolled_at DESC
"""

_SQL_LIST_IDENTITIES_ALL = """
    SELECT identity_id, display_name, metadata, is_active, enrolled_at
    FROM continuous_tracking.identities
    ORDER BY enrolled_at DESC
"""

_SQL_UPSERT_GALLERY_ENTRY = """
    INSERT INTO continuous_tracking.reid_gallery
        (id, identity_id, embedding, quality, origin_tracklet_id, seen_at,
         face_confirmed, orientation, camera_id, state)
    VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9, $10)
    ON CONFLICT (id) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        quality = EXCLUDED.quality,
        seen_at = EXCLUDED.seen_at,
        face_confirmed = EXCLUDED.face_confirmed,
        orientation = EXCLUDED.orientation,
        camera_id = EXCLUDED.camera_id,
        state = EXCLUDED.state,
        updated_at = now()
"""

_SQL_GET_GALLERY_ENTRY = """
    SELECT id, identity_id, embedding, quality, origin_tracklet_id, seen_at,
           face_confirmed, orientation
    FROM continuous_tracking.reid_gallery
    WHERE id = $1
"""

_SQL_LIST_GALLERY_ENTRIES = """
    SELECT rg.id, rg.identity_id, rg.embedding, rg.quality, rg.origin_tracklet_id,
           rg.seen_at, rg.face_confirmed, rg.orientation, rg.state,
           rg.source_episode_id, rg.camera_id
    FROM continuous_tracking.reid_gallery rg
    INNER JOIN continuous_tracking.identities i ON rg.identity_id = i.identity_id
    WHERE ($1::text IS NULL OR rg.identity_id = $1)
      AND ($2 IS TRUE OR i.is_active = TRUE)
      AND rg.state = 'operator_verified'
    ORDER BY rg.seen_at DESC
    LIMIT 100
"""

_SQL_SEARCH_SIMILAR = """
    SELECT rg.id, rg.identity_id, rg.embedding, rg.quality,
           rg.origin_tracklet_id, rg.seen_at, rg.face_confirmed,
           rg.orientation, rg.camera_id, rg.state, rg.source_episode_id,
           1.0 - (rg.embedding <=> $3::vector) AS similarity
    FROM continuous_tracking.reid_gallery rg
    WHERE ($1::text IS NULL OR rg.identity_id = $1)
      AND rg.identity_id IS NOT NULL AND rg.identity_id != ''
      AND ($2 IS TRUE
           OR (
               SELECT is_active
               FROM continuous_tracking.identities
               WHERE identity_id = rg.identity_id
           ))
      AND ($4::integer IS NULL OR rg.seen_at > now() - ($5::integer || 'seconds')::interval)
      AND rg.state = 'operator_verified'
    ORDER BY rg.embedding <=> $3::vector
    LIMIT $6
"""

_SQL_LIST_GALLERY_FOR_TRACKLETS = """
    SELECT rg.id, rg.identity_id, rg.embedding, rg.quality, rg.origin_tracklet_id,
           rg.seen_at, rg.face_confirmed, rg.orientation, rg.state,
           rg.source_episode_id, rg.camera_id
    FROM continuous_tracking.reid_gallery rg
    WHERE rg.origin_tracklet_id = ANY($1::uuid[])
      AND rg.state = 'operator_verified'
    ORDER BY rg.seen_at DESC
    LIMIT $2
"""

_SQL_UPDATE_IDENTITY_FOR_TRACKLETS = """
    UPDATE continuous_tracking.reid_gallery
    SET identity_id = $2, updated_at = now()
    WHERE origin_tracklet_id = ANY($1::uuid[])
      AND (identity_id = '' OR identity_id IS NULL)
"""


class PostgresGalleryRepository(GalleryRepository):
    """Postgres implementation of the GalleryRepository.

    Uses pgvectorscale's StreamingDiskANN index for ANN search. The index
    must be created by the migration (see migrations/0001_init.up.sql).
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_identity(self, identity: Identity) -> str:
        metadata_json = json.dumps(identity.metadata)
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_IDENTITY,
                identity.identity_id,
                identity.display_name,
                metadata_json,
                identity.is_active,
            )
        return identity.identity_id

    async def get_identity(self, identity_id: str) -> Identity | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_IDENTITY, identity_id)
        if row is None:
            return None
        return Identity(
            identity_id=row["identity_id"],
            display_name=row["display_name"],
            metadata=json.loads(row["metadata"]),
            is_active=row["is_active"],
            enrolled_at=row["enrolled_at"],
        )

    async def list_identities(self, active_only: bool = True) -> list[Identity]:
        sql = _SQL_LIST_IDENTITIES_ACTIVE if active_only else _SQL_LIST_IDENTITIES_ALL
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
        return [
            Identity(
                identity_id=row["identity_id"],
                display_name=row["display_name"],
                metadata=json.loads(row["metadata"]),
                is_active=row["is_active"],
                enrolled_at=row["enrolled_at"],
            )
            for row in rows
        ]

    async def upsert_gallery_entry(self, entry: GalleryEmbedding) -> str:
        embedding_str = _embedding_to_pgvector(entry.embedding)
        identity_id = entry.identity_id if entry.identity_id else None
        # origin_tracklet_id is a nullable UUID column; the domain default is ""
        # (online multi-view seeding has no originating tracklet). Coerce empty
        # to None so asyncpg does not reject "" as an invalid UUID.
        origin_tracklet_id = entry.origin_tracklet_id if entry.origin_tracklet_id else None
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_GALLERY_ENTRY,
                entry.gallery_entry_id,
                identity_id,
                embedding_str,
                entry.quality,
                origin_tracklet_id,
                entry.seen_at,
                entry.face_confirmed,
                entry.orientation,
                entry.camera_id,
                entry.state,
            )
        return entry.gallery_entry_id

    async def get_gallery_entry(self, gallery_entry_id: str) -> GalleryEmbedding | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_GALLERY_ENTRY, gallery_entry_id)
        if row is None:
            return None
        return GalleryEmbedding(
            gallery_entry_id=row["id"],
            identity_id=row["identity_id"],
            embedding=_pgvector_to_list(row["embedding"]),
            quality=row["quality"],
            seen_at=row["seen_at"],
            origin_tracklet_id=row["origin_tracklet_id"] or "",
            face_confirmed=row["face_confirmed"],
            orientation=row.get("orientation", 4),
            camera_id=row.get("camera_id") or "",
            state=row.get("state", "pending_review"),
            source_episode_id=(
                str(row["source_episode_id"]) if row.get("source_episode_id") else None
            ),
        )

    async def phs_with_pending_reid(self, ph_ids: list[str]) -> set[str]:
        if not ph_ids:
            return set()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ph_id::text AS ph_id
                FROM continuous_tracking.reid_gallery
                WHERE state = 'pending_review'
                  AND ph_id = ANY($1::uuid[])
                """,
                ph_ids,
            )
        return {row["ph_id"] for row in rows if row["ph_id"]}

    async def list_gallery_entries(
        self, identity_id: str | None = None, active_only: bool = True
    ) -> list[GalleryEmbedding]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _SQL_LIST_GALLERY_ENTRIES,
                identity_id,
                active_only,
            )
        return [
            GalleryEmbedding(
                gallery_entry_id=row["id"],
                identity_id=row["identity_id"],
                embedding=_pgvector_to_list(row["embedding"]),
                quality=row["quality"],
                seen_at=row["seen_at"],
                origin_tracklet_id=row["origin_tracklet_id"] or "",
                face_confirmed=row["face_confirmed"],
                orientation=row.get("orientation", 4),
                camera_id=row.get("camera_id") or "",
                state=row.get("state", "pending_review"),
                source_episode_id=(
                    str(row["source_episode_id"]) if row.get("source_episode_id") else None
                ),
            )
            for row in rows
        ]

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
    ) -> list[tuple[GalleryEmbedding, float]]:
        embedding_str = _embedding_to_pgvector(embedding)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _SQL_SEARCH_SIMILAR,
                None,  # $1: identity_id filter
                True,  # $2: active_only filter
                embedding_str,  # $3: query embedding
                max_age_seconds,  # $4: IS NULL check
                max_age_seconds,  # $5: interval seconds value
                limit,  # $6: LIMIT
            )
        return [
            (
                GalleryEmbedding(
                    gallery_entry_id=row["id"],
                    identity_id=row["identity_id"],
                    embedding=_pgvector_to_list(row["embedding"]),
                    quality=row["quality"],
                    seen_at=row["seen_at"],
                    origin_tracklet_id=row["origin_tracklet_id"] or "",
                    face_confirmed=row["face_confirmed"],
                    camera_id=row.get("camera_id") or "",
                    orientation=row.get("orientation", 4),
                    state=row.get("state", "pending_review"),
                    source_episode_id=(
                        str(row["source_episode_id"]) if row.get("source_episode_id") else None
                    ),
                ),
                float(row["similarity"]),
            )
            for row in rows
        ]

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> list[GalleryEmbedding]:
        # The SQL already hard-restricts to operator_verified rows, matching the
        # resolver's default {"operator_verified"} state set, so the extra
        # state/version parameters are accepted for protocol parity.
        if not tracklet_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _SQL_LIST_GALLERY_FOR_TRACKLETS,
                list(tracklet_ids),
                limit,
            )
        return [
            GalleryEmbedding(
                gallery_entry_id=row["id"],
                identity_id=row["identity_id"],
                embedding=_pgvector_to_list(row["embedding"]),
                quality=row["quality"],
                seen_at=row["seen_at"],
                origin_tracklet_id=row["origin_tracklet_id"] or "",
                face_confirmed=row["face_confirmed"],
                orientation=row.get("orientation", 4),
            )
            for row in rows
        ]

    async def update_identity_for_tracklets(
        self,
        tracklet_ids: set[str],
        identity_id: str,
    ) -> int:
        """Backfill identity_id on gallery entries for the given tracklets."""
        if not tracklet_ids or not identity_id:
            return 0
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                _SQL_UPDATE_IDENTITY_FOR_TRACKLETS,
                list(tracklet_ids),
                identity_id,
            )
        updated = int(result.split()[-1]) if result else 0
        if updated:
            logger.debug(
                "gallery_identity_backfilled",
                tracklet_count=len(tracklet_ids),
                updated_rows=updated,
                identity_id=identity_id,
            )
        return updated

    async def gallery_similarity(
        self,
        tracklet_ids_a: set[str],
        tracklet_ids_b: set[str],
        limit: int = 20,
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> float:
        entries_a = await self.list_gallery_entries_for_tracklets(
            tracklet_ids_a, limit, allowed_states, model_versions
        )
        entries_b = await self.list_gallery_entries_for_tracklets(
            tracklet_ids_b, limit, allowed_states, model_versions
        )
        if not entries_a and not entries_b:
            return 0.0
        if not entries_a or not entries_b:
            return 0.5
        return self._cosine_between_centroids(entries_a, entries_b)

    # -- M09 ReID review queue ------------------------------------------------

    async def list_review_candidates(
        self,
        *,
        state: str = "pending_review",
        identity_id: str | None = None,
        camera_id: str | None = None,
        model_version: str | None = None,
        source_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ReviewCandidate], int]:
        # One filtered window shared by the page query and the COUNT so the
        # total reflects the same predicate the page was drawn from.
        where = """
            WHERE state = $1
              AND ($2::text IS NULL OR identity_id = $2)
              AND ($3::text IS NULL OR camera_id = $3)
              AND ($4::text IS NULL OR model_version = $4)
              AND ($5::text IS NULL OR candidate_reason = $5)
              AND ($6::timestamptz IS NULL OR capture_time >= $6)
              AND ($7::timestamptz IS NULL OR capture_time <= $7)
        """
        params = [state, identity_id, camera_id, model_version, source_type, since, until]
        async with self._pool.acquire() as conn:
            total = await conn.fetchval(
                f"SELECT count(*) FROM continuous_tracking.reid_gallery {where}",
                *params,
            )
            rows = await conn.fetch(
                f"""
                SELECT {_REVIEW_COLUMNS}
                FROM continuous_tracking.reid_gallery
                {where}
                ORDER BY created_at ASC NULLS LAST, seen_at ASC
                LIMIT $8 OFFSET $9
                """,
                *params,
                limit,
                offset,
            )
        return ([_row_to_review_candidate(r) for r in rows], int(total or 0))

    async def get_review_candidate(self, candidate_id: str) -> ReviewCandidate | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_REVIEW_COLUMNS} FROM continuous_tracking.reid_gallery WHERE id = $1",
                candidate_id,
            )
        return _row_to_review_candidate(row) if row is not None else None

    async def list_review_events(self, candidate_id: str) -> list[ReviewEvent]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_id, entry_id, previous_state, new_state, actor,
                       reason, note, event_time, audit_version
                FROM continuous_tracking.gallery_review_events
                WHERE entry_id = $1
                ORDER BY event_time ASC, audit_version ASC
                """,
                candidate_id,
            )
        return [
            ReviewEvent(
                event_id=str(r["event_id"]),
                entry_id=str(r["entry_id"]),
                previous_state=r["previous_state"],
                new_state=r["new_state"],
                actor=r["actor"],
                reason=r["reason"],
                note=r["note"],
                event_time=r["event_time"],
                audit_version=r["audit_version"],
            )
            for r in rows
        ]

    async def count_review_queue(self) -> dict[str, int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT state, count(*) AS n
                FROM continuous_tracking.reid_gallery
                GROUP BY state
                """
            )
        counts = {"pending_review": 0, "operator_verified": 0, "rejected": 0}
        for r in rows:
            counts[str(r["state"])] = int(r["n"])
        return counts

    async def apply_review_action(
        self,
        candidate_id: str,
        *,
        action: str,
        actor: str,
        base_audit_version: int,
        reason: str | None = None,
        note: str | None = None,
        new_identity_id: str | None = None,
    ) -> ReviewCandidate:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                    SELECT state, audit_version
                    FROM continuous_tracking.reid_gallery
                    WHERE id = $1
                    FOR UPDATE
                    """,
                candidate_id,
            )
            if row is None:
                raise ReviewNotFoundError(candidate_id)
            if row["state"] != "pending_review":
                raise ReviewConflictError(f"{candidate_id} already reviewed (state={row['state']})")
            if row["audit_version"] != base_audit_version:
                raise ReviewConflictError(
                    f"{candidate_id} stale audit_version "
                    f"(have {row['audit_version']}, sent {base_audit_version})"
                )
            prev_state = row["state"]
            next_version = base_audit_version + 1

            if action == "reject":
                # Vector bytes are nulled here; the crop object is removed by
                # the service after this transaction commits. The crop_key,
                # hashes, and audit metadata are retained as a fingerprint.
                await conn.execute(
                    """
                        UPDATE continuous_tracking.reid_gallery
                        SET state = 'rejected', embedding = NULL,
                            reviewed_actor = $2, reviewed_time = now(),
                            review_reason = $3, review_note = $4,
                            audit_version = $5
                        WHERE id = $1
                        """,
                    candidate_id,
                    actor,
                    reason,
                    note,
                    next_version,
                )
                new_state = "rejected"
            elif action == "relabel":
                await conn.execute(
                    """
                        UPDATE continuous_tracking.reid_gallery
                        SET state = 'operator_verified', identity_id = $2,
                            reviewed_actor = $3, reviewed_time = now(),
                            review_reason = $4, review_note = $5,
                            audit_version = $6
                        WHERE id = $1
                        """,
                    candidate_id,
                    new_identity_id,
                    actor,
                    reason,
                    note,
                    next_version,
                )
                new_state = "operator_verified"
            elif action == "approve":
                await conn.execute(
                    """
                        UPDATE continuous_tracking.reid_gallery
                        SET state = 'operator_verified',
                            reviewed_actor = $2, reviewed_time = now(),
                            review_reason = $3, review_note = $4,
                            audit_version = $5
                        WHERE id = $1
                        """,
                    candidate_id,
                    actor,
                    reason,
                    note,
                    next_version,
                )
                new_state = "operator_verified"
            else:
                raise ValueError(f"unknown review action: {action}")

            await conn.execute(
                """
                    INSERT INTO continuous_tracking.gallery_review_events
                    (entry_id, previous_state, new_state, actor, reason, note, audit_version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                candidate_id,
                prev_state,
                new_state,
                actor,
                reason,
                note,
                next_version,
            )

        updated = await self.get_review_candidate(candidate_id)
        if updated is None:  # pragma: no cover - row cannot vanish mid-call
            raise ReviewNotFoundError(candidate_id)
        return updated

    async def compensate_review(
        self,
        candidate_id: str,
        *,
        actor: str,
        base_audit_version: int,
    ) -> ReviewCandidate:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                    SELECT state, audit_version
                    FROM continuous_tracking.reid_gallery
                    WHERE id = $1
                    FOR UPDATE
                    """,
                candidate_id,
            )
            if row is None:
                raise ReviewNotFoundError(candidate_id)
            if row["state"] != "operator_verified":
                raise ReviewConflictError(
                    f"{candidate_id} is not operator_verified (state={row['state']})"
                )
            if row["audit_version"] != base_audit_version:
                raise ReviewConflictError(f"{candidate_id} stale audit_version")
            next_version = base_audit_version + 1
            await conn.execute(
                """
                    UPDATE continuous_tracking.reid_gallery
                    SET state = 'pending_review', reviewed_actor = $2,
                        reviewed_time = now(), review_reason = 'compensated',
                        audit_version = $3
                    WHERE id = $1
                    """,
                candidate_id,
                actor,
                next_version,
            )
            await conn.execute(
                """
                    INSERT INTO continuous_tracking.gallery_review_events
                    (entry_id, previous_state, new_state, actor, reason, note, audit_version)
                    VALUES ($1, 'operator_verified', 'pending_review', $2, 'compensated', NULL, $3)
                    """,
                candidate_id,
                actor,
                next_version,
            )
        updated = await self.get_review_candidate(candidate_id)
        if updated is None:  # pragma: no cover
            raise ReviewNotFoundError(candidate_id)
        return updated


def _row_to_review_candidate(row: asyncpg.Record) -> ReviewCandidate:
    bbox = row["bbox"]
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except (json.JSONDecodeError, TypeError):
            bbox = None
    return ReviewCandidate(
        candidate_id=str(row["id"]),
        identity_id=row["identity_id"],
        proposed_identity_id=row["proposed_identity_id"],
        effective_identity_id=row["effective_identity_id"],
        state=row["state"],
        label_source=row["label_source"],
        candidate_reason=row["candidate_reason"],
        model_version=row["model_version"],
        preprocessing_version=row["preprocessing_version"],
        dimension=row["dimension"],
        crop_key=row["crop_key"],
        source_frame_key=row["source_frame_key"],
        crop_hash=row["crop_hash"],
        frame_hash=row["frame_hash"],
        bbox=bbox if isinstance(bbox, dict) else None,
        crop_width=row["crop_width"],
        crop_height=row["crop_height"],
        ph_id=str(row["ph_id"]) if row["ph_id"] else None,
        observation_id=str(row["observation_id"]) if row["observation_id"] else None,
        keyframe_id=str(row["keyframe_id"]) if row["keyframe_id"] else None,
        camera_id=row["camera_id"],
        capture_time=row["capture_time"],
        confidence=row["confidence"],
        orientation=row["orientation"] if row["orientation"] is not None else 4,
        quality=row["quality"] if row["quality"] is not None else 0.0,
        is_truncated=bool(row["is_truncated"]),
        is_occluded=bool(row["is_occluded"]),
        source_episode_id=str(row["source_episode_id"]) if row["source_episode_id"] else None,
        created_actor=row["created_actor"],
        created_at=row["created_at"],
        seen_at=row["seen_at"],
        reviewed_actor=row["reviewed_actor"],
        reviewed_time=row["reviewed_time"],
        review_reason=row["review_reason"],
        review_note=row["review_note"],
        audit_version=row["audit_version"],
    )


def _embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert a Python list of floats to a pgvector string literal.

    Returns a string like '[0.1,0.2,0.3,...]' that PostgreSQL's vector type
    accepts directly.
    """
    return f"[{','.join(f'{v:.8f}' for v in embedding)}]"


def _pgvector_to_list(vector_value: str | list[float] | bytes) -> list[float]:
    """Convert a pgvector value to a Python list of floats.

    Handles both the PostgreSQL vector text representation ('[0.1,0.2,...]')
    and native Python list types returned by newer pgvector/asyncpg versions.
    """
    if vector_value is None or vector_value == "[]" or vector_value == "":
        return []
    if isinstance(vector_value, list):
        return [float(v) for v in vector_value]
    if isinstance(vector_value, bytes):
        vector_value = vector_value.decode("utf-8")
    # Remove brackets and parse
    inner = vector_value.strip("[]")
    return [float(x) for x in inner.split(",") if x.strip()]
