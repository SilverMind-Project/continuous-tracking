"""Gallery endpoints: seed the ReID gallery with named identities from single crops.

The Bayesian identity resolver's ``search_similar`` uses an INNER JOIN with
the ``identities`` table, so gallery entries whose ``identity_id=""`` are
invisible.  The add_crop endpoint receives a raw JPEG crop, embeds it, and
saves it directly. Legacy tracklet promotion is intentionally not exposed.

Endpoints
---------
``POST /internal/gallery/add_crop``
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from structlog import get_logger

from ..domain import GalleryEmbedding
from ..storage.base import GalleryRepository, InMemoryGalleryRepository

if TYPE_CHECKING:
    from ..inference.reid_embedder import ReidEmbedder
    from ..services.identity_roster_sync import IdentityRosterSyncService

logger = get_logger(__name__)

router = APIRouter(tags=["gallery-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


@dataclass
class _GalleryContext:
    gallery_repo: GalleryRepository
    reid_embedder: ReidEmbedder | None = field(default=None)
    roster_sync: IdentityRosterSyncService | None = field(default=None)


_ctx: _GalleryContext = _GalleryContext(gallery_repo=InMemoryGalleryRepository())


def get_context() -> _GalleryContext:
    return _ctx


def set_context(
    gallery_repo: GalleryRepository,
    reid_embedder: ReidEmbedder | None = None,
    roster_sync: IdentityRosterSyncService | None = None,
) -> None:
    """Override with production repository and embedder (called from lifespan)."""
    global _ctx
    _ctx = _GalleryContext(
        gallery_repo=gallery_repo,
        reid_embedder=reid_embedder,
        roster_sync=roster_sync,
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
        embeddings = await ctx.reid_embedder.embed_batch([cast(npt.NDArray[np.uint8], crop)])
    except Exception as err:
        logger.exception("gallery_add_crop_embed_error", identity_id=identity_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "gallery.embed_failed",
                "message": "ReID embedding failed.",
            },
        ) from err

    embedding = embeddings[0].tolist()
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


@router.delete("/internal/gallery/identities/{identity_id}")
async def purge_gallery_identity(
    identity_id: str,
    ctx: _GalleryContext = Depends(get_context),
) -> dict[str, object]:
    """Purge an unenrolled identity and all associated anchors/gallery rows."""
    if ctx.roster_sync is not None:
        removed = await ctx.roster_sync.purge_identity(identity_id)
    else:
        removed = await ctx.gallery_repo.purge_identity(identity_id)
    return {"purged": removed}


@router.post("/internal/gallery/identities/sync")
async def sync_gallery_identities(
    ctx: _GalleryContext = Depends(get_context),
) -> dict[str, int]:
    """Reconcile CTS identities against the enrolled household roster."""
    if ctx.roster_sync is None:
        return {"upserted": 0, "purged": 0}
    upserted, purged = await ctx.roster_sync.sync_once()
    return {"upserted": upserted, "purged": purged}
