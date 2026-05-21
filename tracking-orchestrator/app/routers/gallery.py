"""Gallery enrollment endpoints: seed the ReID gallery with named identities and add single crops.

The Bayesian identity resolver's ``search_similar`` uses an INNER JOIN with
the ``identities`` table, so gallery entries whose ``identity_id=""`` are
invisible.  The enroll endpoint creates named gallery entries from tracklet
embeddings; the add_crop endpoint receives a raw JPEG crop, embeds it, and
saves it directly.

Endpoints
---------
``POST /internal/gallery/enroll``
``POST /internal/gallery/add_crop``
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from structlog import get_logger

from ..domain import GalleryEmbedding, Identity
from ..storage.base import GalleryRepository, InMemoryGalleryRepository

if TYPE_CHECKING:
    from ..inference.reid_embedder import ReidEmbedder

logger = get_logger(__name__)

router = APIRouter(tags=["gallery-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


@dataclass
class _GalleryContext:
    gallery_repo: GalleryRepository
    reid_embedder: ReidEmbedder | None = field(default=None)


_ctx: _GalleryContext = _GalleryContext(gallery_repo=InMemoryGalleryRepository())


def get_context() -> _GalleryContext:
    return _ctx


def set_context(
    gallery_repo: GalleryRepository,
    reid_embedder: ReidEmbedder | None = None,
) -> None:
    """Override with production repository and embedder (called from lifespan)."""
    global _ctx
    _ctx = _GalleryContext(gallery_repo=gallery_repo, reid_embedder=reid_embedder)


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


# ---------------------------------------------------------------------------
# POST /internal/gallery/add_crop
# ---------------------------------------------------------------------------


@router.post("/internal/gallery/add_crop", status_code=204)
async def add_crop_to_gallery(
    request: Request,
    identity_id: str = Header(alias="X-Identity-Id"),
    ctx: _GalleryContext = Depends(get_context),
) -> None:
    """Embed a raw JPEG crop and save it to the identity's ReID gallery.

    Receives JPEG bytes in the request body, decodes them with OpenCV, runs the
    ReID embedder to produce an embedding vector, and upserts a
    :class:`GalleryEmbedding` entry for the given identity.

    Headers
    -------
    X-Identity-Id: the identity to associate this crop with.

    Returns 204 on success, 400 if the image cannot be decoded, 422 if the
    embedder is not available.
    """
    if ctx.reid_embedder is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "gallery.reid_embedder_unavailable",
                "message": "ReID embedder is not available. Is Triton connected?",
            },
        )

    image_bytes = await request.body()
    arr = np.frombuffer(image_bytes, np.uint8)
    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if crop is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "gallery.crop_decode_failed",
                "message": "Failed to decode crop image bytes.",
            },
        )

    try:
        embeddings = await ctx.reid_embedder.embed_batch([crop])
    except Exception as err:
        logger.exception("gallery_add_crop_embed_error", identity_id=identity_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "gallery.embed_failed",
                "message": "ReID embedding failed.",
            },
        ) from err

    embedding = embeddings[0]
    now = datetime.now(UTC)
    gallery_entry = GalleryEmbedding(
        gallery_entry_id=str(uuid.uuid4()),
        identity_id=identity_id,
        embedding=embedding,
        seen_at=now,
        quality=1.0,
        face_confirmed=False,
        camera_id="",
    )

    await ctx.gallery_repo.upsert_gallery_entry(gallery_entry)

    logger.info(
        "gallery_add_crop_saved",
        identity_id=identity_id,
        gallery_entry_id=gallery_entry.gallery_entry_id,
    )
