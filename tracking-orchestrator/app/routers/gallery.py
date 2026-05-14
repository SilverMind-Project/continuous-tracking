"""Gallery enrollment endpoint: seed the ReID gallery with named identities.

The Bayesian identity resolver's ``search_similar`` uses an INNER JOIN with
the ``identities`` table, so gallery entries whose ``identity_id=""`` are
invisible.  This endpoint is the surgical tool that makes tracklet embeddings
visible: it takes a known ``identity_id`` + ``tracklet_id``, upserts the
:class:`Identity` record, and creates fresh :class:`GalleryEmbedding` entries
(new UUIDs) with the correct ``identity_id``.  The orphaned empty entries
remain in the DB but are unreachable by the resolver.

Endpoint
--------
``POST /internal/gallery/enroll``
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from structlog import get_logger

from ..domain import GalleryEmbedding, Identity
from ..storage.base import GalleryRepository, InMemoryGalleryRepository

logger = get_logger(__name__)

router = APIRouter(tags=["gallery-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


@dataclass
class _GalleryContext:
    gallery_repo: GalleryRepository


_ctx: _GalleryContext = _GalleryContext(gallery_repo=InMemoryGalleryRepository())


def get_context() -> _GalleryContext:
    return _ctx


def set_context(gallery_repo: GalleryRepository) -> None:
    """Override with production repository (called from lifespan)."""
    global _ctx
    _ctx = _GalleryContext(gallery_repo=gallery_repo)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EnrollRequest(BaseModel):
    """Body for ``POST /internal/gallery/enroll``.

    ``tracklet_id`` identifies which gallery embeddings to promote.
    ``display_name`` is used when the identity record does not yet exist.
    """

    identity_id: str = Field(..., min_length=1, max_length=128)
    tracklet_id: str = Field(..., min_length=1, max_length=128)
    display_name: str | None = Field(default=None, max_length=128)


class EnrollResponse(BaseModel):
    identity_id: str
    enrolled_count: int
    enrolled_at: str


# ---------------------------------------------------------------------------
# POST /internal/gallery/enroll
# ---------------------------------------------------------------------------


@router.post("/internal/gallery/enroll", response_model=EnrollResponse)
async def enroll_tracklet(
    body: EnrollRequest,
    ctx: _GalleryContext = Depends(get_context),
) -> EnrollResponse:
    """Promote a tracklet's gallery embeddings to a named identity.

    Workflow
    --------
    1. Upsert the :class:`Identity` record (creates if absent; preserves
       ``enrolled_at`` on the existing row since ``upsert_identity`` keeps
       whichever ``enrolled_at`` the caller supplies — send ``now()`` only
       when the identity is being created for the first time).
    2. Fetch all :class:`GalleryEmbedding` rows whose
       ``origin_tracklet_id == tracklet_id``.
    3. For each embedding create a *new* row with a fresh UUID and the
       supplied ``identity_id``; the old empty-identity rows are left in
       place but remain invisible to ``search_similar``.

    Returns 404 when no gallery embeddings exist for the given tracklet so
    the caller can surface a meaningful error rather than silently enrolling
    zero vectors.
    """
    now = datetime.now(UTC)

    existing = await ctx.gallery_repo.get_identity(body.identity_id)
    identity = Identity(
        identity_id=body.identity_id,
        display_name=body.display_name or body.identity_id,
        enrolled_at=existing.enrolled_at if existing else now,
        metadata=existing.metadata if existing else {},
        is_active=True,
    )
    await ctx.gallery_repo.upsert_identity(identity)

    source_entries = await ctx.gallery_repo.list_gallery_entries_for_tracklets(
        tracklet_ids={body.tracklet_id},
        limit=50,
    )
    if not source_entries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "gallery.no_embeddings_for_tracklet",
                "message": (
                    f"No gallery embeddings found for tracklet {body.tracklet_id!r}. "
                    "Ensure the tracklet has been observed before enrolling."
                ),
            },
        )

    for entry in source_entries:
        named = GalleryEmbedding(
            gallery_entry_id=str(uuid.uuid4()),
            identity_id=body.identity_id,
            embedding=entry.embedding,
            seen_at=entry.seen_at,
            quality=entry.quality,
            origin_tracklet_id=entry.origin_tracklet_id,
            face_confirmed=entry.face_confirmed,
            camera_id=entry.camera_id,
        )
        await ctx.gallery_repo.upsert_gallery_entry(named)

    logger.info(
        "gallery_enrollment_applied",
        identity_id=body.identity_id,
        tracklet_id=body.tracklet_id,
        enrolled_count=len(source_entries),
    )

    return EnrollResponse(
        identity_id=body.identity_id,
        enrolled_count=len(source_entries),
        enrolled_at=now.isoformat(),
    )
