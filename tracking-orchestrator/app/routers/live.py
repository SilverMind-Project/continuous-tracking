"""Live view + health internal endpoints consumed by the CC BFF.

These endpoints back the Live view and the admin health widgets. Heavier,
low-rate reads live here (the hot tracking-event path is the Redis stream).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Depends, Query
from structlog import get_logger

from ..storage.base import (
    GalleryRepository,
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    PHRepositoryProtocol,
)

logger = get_logger(__name__)

router = APIRouter(tags=["live-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


@dataclass
class _LiveContext:
    ph_repo: PHRepositoryProtocol
    gallery_repo: GalleryRepository
    feature_flags: dict[str, bool] = field(
        default_factory=lambda: {
            "retroactive_revision_enabled": True,
            "cross_camera_association_enabled": True,
            "manual_corrections_enabled": True,
        }
    )


_ctx: _LiveContext = _LiveContext(
    ph_repo=InMemoryPHRepository(),
    gallery_repo=InMemoryGalleryRepository(),
)


def get_context() -> _LiveContext:
    return _ctx


def set_context(
    ph_repo: PHRepositoryProtocol,
    keyframe_repo: object | None = None,
    gallery_repo: GalleryRepository | None = None,
    feature_flags: dict[str, bool] | None = None,
) -> None:
    _ = keyframe_repo
    global _ctx
    _ctx = _LiveContext(
        ph_repo=ph_repo,
        gallery_repo=gallery_repo or InMemoryGalleryRepository(),
        feature_flags=feature_flags or _ctx.feature_flags,
    )


# ---------------------------------------------------------------------------
# GET /internal/identities
# ---------------------------------------------------------------------------


@router.get("/internal/identities")
async def list_identities(
    active_only: bool = Query(True, description="Only return active identities"),
    ctx: _LiveContext = Depends(get_context),
) -> dict[str, Any]:
    """Return all known named identities from the ReID gallery."""
    identities = await ctx.gallery_repo.list_identities(active_only=active_only)
    return {
        "identities": [
            {
                "identity_id": i.identity_id,
                "display_name": i.display_name,
                "enrolled_at": i.enrolled_at.isoformat(),
                "is_active": i.is_active,
            }
            for i in identities
        ],
        "count": len(identities),
    }


# ---------------------------------------------------------------------------
# GET /internal/health
# ---------------------------------------------------------------------------


@router.get("/internal/health")
async def get_health() -> dict[str, Any]:
    """Lightweight liveness ping."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /internal/features
# ---------------------------------------------------------------------------


@router.get("/internal/features")
async def get_feature_flags(
    ctx: _LiveContext = Depends(get_context),
) -> dict[str, Any]:
    """Return boolean feature flags for the orchestrator."""
    return {"flags": dict(ctx.feature_flags)}
