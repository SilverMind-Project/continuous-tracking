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
    INSERT INTO identities (identity_id, display_name, metadata, is_active)
    VALUES ($1, $2, $3::jsonb, $4)
    ON CONFLICT (identity_id) DO UPDATE SET
        display_name = EXCLUDED.display_name,
        metadata = EXCLUDED.metadata,
        is_active = EXCLUDED.is_active,
        updated_at = now()
"""

_SQL_GET_IDENTITY = """
    SELECT identity_id, display_name, metadata, is_active, enrolled_at
    FROM identities
    WHERE identity_id = $1
"""

_SQL_LIST_IDENTITIES = """
    SELECT identity_id, display_name, metadata, is_active, enrolled_at
    FROM identities
    WHERE is_active = $1
    ORDER BY enrolled_at DESC
"""

_SQL_UPSERT_GALLERY_ENTRY = """
    INSERT INTO reid_gallery (id, identity_id, embedding, quality, origin_tracklet_id,
                              seen_at, face_confirmed)
    VALUES ($1, $2, $3::vector, $4, $5, $6, $7)
    ON CONFLICT (id) DO NOTHING
"""

_SQL_GET_GALLERY_ENTRY = """
    SELECT id, identity_id, embedding, quality, origin_tracklet_id, seen_at, face_confirmed
    FROM reid_gallery
    WHERE id = $1
"""

_SQL_LIST_GALLERY_ENTRIES = """
    SELECT rg.id, rg.identity_id, rg.embedding, rg.quality, rg.origin_tracklet_id,
           rg.seen_at, rg.face_confirmed
    FROM reid_gallery rg
    INNER JOIN identities i ON rg.identity_id = i.identity_id
    WHERE ($1::text IS NULL OR rg.identity_id = $1)
      AND ($2 IS TRUE OR i.is_active = TRUE)
    ORDER BY rg.seen_at DESC
    LIMIT 100
"""

_SQL_SEARCH_SIMILAR = """
    SELECT rg.id, rg.identity_id, rg.embedding, rg.quality,
           rg.origin_tracklet_id, rg.seen_at, rg.face_confirmed,
           t.camera_id
    FROM reid_gallery rg
    LEFT JOIN tracklets t ON rg.origin_tracklet_id = t.tracklet_id
    WHERE ($1::text IS NULL OR rg.identity_id = $1)
      AND ($2 IS TRUE
           OR (
               SELECT is_active
               FROM identities
               WHERE identity_id = rg.identity_id
           ))
      AND ($4::text IS NULL OR t.camera_id = $4)
      AND ($5 IS NULL OR rg.seen_at > now() - ($6::integer || 'seconds')::interval)
    ORDER BY rg.embedding <=> $3::vector
    LIMIT $7
"""


class PostgresGalleryRepository(GalleryRepository):
    """Postgres implementation of the GalleryRepository.

    Uses pgvector's HNSW index for ANN search. The index must be created
    by the migration (see migrations/0001_init.sql).
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
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_LIST_IDENTITIES, active_only)
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
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_GALLERY_ENTRY,
                entry.gallery_entry_id,
                entry.identity_id,
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
    ) -> list[GalleryEmbedding]:
        embedding_str = _embedding_to_pgvector(embedding)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _SQL_SEARCH_SIMILAR,
                None,  # $1: identity_id filter
                True,  # $2: active_only filter
                embedding_str,  # $3: query embedding
                camera_id,  # $4: camera_id filter
                max_age_seconds,  # $5: max_age_seconds filter
                limit,  # $6: limit
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
                camera_id=row["camera_id"] or "",
            )
            for row in rows
        ]


def _embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert a Python list of floats to a pgvector string literal.

    Returns a string like '[0.1,0.2,0.3,...]' that PostgreSQL's vector type
    accepts directly.
    """
    return f"[{','.join(f'{v:.8f}' for v in embedding)}]"


def _pgvector_to_list(vector_str: str) -> list[float]:
    """Convert a pgvector string literal to a Python list of floats.

    Handles the PostgreSQL vector text representation.
    """
    if not vector_str or vector_str == "[]":
        return []
    # Remove brackets and parse
    inner = vector_str.strip("[]")
    return [float(x) for x in inner.split(",") if x.strip()]
