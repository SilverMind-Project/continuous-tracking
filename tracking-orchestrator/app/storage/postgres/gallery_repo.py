"""Postgres/TimescaleDB implementation of the GalleryRepository protocol.

Handles identities, gallery embeddings, and ANN search via pgvector HNSW.
"""

from __future__ import annotations

import json

import asyncpg  # type: ignore[import-untyped]
from structlog import get_logger

from ...domain import GalleryEmbedding, Identity
from ..base import GalleryRepository

logger = get_logger(__name__)

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
        (id, identity_id, embedding, quality, origin_tracklet_id, seen_at, face_confirmed)
    VALUES ($1, $2, $3::vector, $4, $5, $6, $7)
    ON CONFLICT (id) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        quality = EXCLUDED.quality,
        seen_at = EXCLUDED.seen_at,
        face_confirmed = EXCLUDED.face_confirmed,
        updated_at = now()
"""

_SQL_GET_GALLERY_ENTRY = """
    SELECT id, identity_id, embedding, quality, origin_tracklet_id, seen_at, face_confirmed
    FROM continuous_tracking.reid_gallery
    WHERE id = $1
"""

_SQL_LIST_GALLERY_ENTRIES = """
    SELECT rg.id, rg.identity_id, rg.embedding, rg.quality, rg.origin_tracklet_id,
           rg.seen_at, rg.face_confirmed
    FROM continuous_tracking.reid_gallery rg
    INNER JOIN continuous_tracking.identities i ON rg.identity_id = i.identity_id
    WHERE ($1::text IS NULL OR rg.identity_id = $1)
      AND ($2 IS TRUE OR i.is_active = TRUE)
    ORDER BY rg.seen_at DESC
    LIMIT 100
"""

_SQL_SEARCH_SIMILAR = """
    SELECT rg.id, rg.identity_id, rg.embedding, rg.quality,
           rg.origin_tracklet_id, rg.seen_at, rg.face_confirmed,
           t.camera_id,
           1.0 - (rg.embedding <=> $3::vector) AS similarity
    FROM continuous_tracking.reid_gallery rg
    LEFT JOIN continuous_tracking.tracklets t ON rg.origin_tracklet_id = t.tracklet_id
    WHERE ($1::text IS NULL OR rg.identity_id = $1)
      AND rg.identity_id IS NOT NULL AND rg.identity_id != ''
      AND ($2 IS TRUE
           OR (
               SELECT is_active
               FROM continuous_tracking.identities
               WHERE identity_id = rg.identity_id
           ))
      AND ($4::text IS NULL OR t.camera_id = $4)
      AND ($5::integer IS NULL OR rg.seen_at > now() - ($6::integer || 'seconds')::interval)
    ORDER BY rg.embedding <=> $3::vector
    LIMIT $7
"""

_SQL_LIST_GALLERY_FOR_TRACKLETS = """
    SELECT rg.id, rg.identity_id, rg.embedding, rg.quality, rg.origin_tracklet_id,
           rg.seen_at, rg.face_confirmed
    FROM continuous_tracking.reid_gallery rg
    WHERE rg.origin_tracklet_id = ANY($1::uuid[])
    ORDER BY rg.seen_at DESC
    LIMIT $2
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
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_GALLERY_ENTRY,
                entry.gallery_entry_id,
                identity_id,
                embedding_str,
                entry.quality,
                entry.origin_tracklet_id,
                entry.seen_at,
                entry.face_confirmed,
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
        )

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
                camera_id,  # $4: camera_id filter
                max_age_seconds,  # $5: IS NULL check
                max_age_seconds,  # $6: interval seconds value
                limit,  # $7: LIMIT
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
                    camera_id=row["camera_id"] or "",
                ),
                float(row["similarity"]),
            )
            for row in rows
        ]

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
    ) -> list[GalleryEmbedding]:
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
            )
            for row in rows
        ]


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
