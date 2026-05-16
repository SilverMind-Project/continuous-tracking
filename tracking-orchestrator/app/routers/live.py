"""Live view + health internal endpoints consumed by the CC BFF.

These endpoints back the Live view and the admin health widgets. Heavier,
low-rate reads live here (the hot tracking-event path is the Redis stream).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from structlog import get_logger

from ..storage.base import (
    GalleryRepository,
    GlobalTrackRepository,
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
    InMemoryKeyframeRepository,
    KeyframeRepository,
)

logger = get_logger(__name__)

router = APIRouter(tags=["live-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


@dataclass
class _LiveContext:
    global_track_repo: GlobalTrackRepository
    keyframe_repo: KeyframeRepository
    gallery_repo: GalleryRepository
    feature_flags: dict[str, bool] = field(
        default_factory=lambda: {
            "retroactive_revision_enabled": True,
            "cross_camera_association_enabled": True,
            "manual_corrections_enabled": True,
        }
    )


_ctx: _LiveContext = _LiveContext(
    global_track_repo=InMemoryGlobalTrackRepository(),
    keyframe_repo=InMemoryKeyframeRepository(),
    gallery_repo=InMemoryGalleryRepository(),
)


def get_context() -> _LiveContext:
    return _ctx


def set_context(
    global_track_repo: GlobalTrackRepository,
    keyframe_repo: KeyframeRepository | None = None,
    gallery_repo: GalleryRepository | None = None,
    feature_flags: dict[str, bool] | None = None,
) -> None:
    global _ctx
    _ctx = _LiveContext(
        global_track_repo=global_track_repo,
        keyframe_repo=keyframe_repo or InMemoryKeyframeRepository(),
        gallery_repo=gallery_repo or InMemoryGalleryRepository(),
        feature_flags=feature_flags or _ctx.feature_flags,
    )


# ---------------------------------------------------------------------------
# GET /internal/global_tracks
# ---------------------------------------------------------------------------


@router.get("/internal/global_tracks")
async def list_global_tracks(
    open_only: bool = Query(True, description="Only return tracks with state='active'"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    camera_id: str | None = Query(None),
    status: str | None = Query(None, pattern="^(committed|UNKNOWN)$"),
    search: str | None = Query(None),
    ctx: _LiveContext = Depends(get_context),
) -> dict[str, Any]:
    """Return a summary of global tracks for the Live and Corrections views."""
    tracks = await ctx.global_track_repo.list_active()
    if not open_only:
        # list_active() is the only protocol accessor; until a list-all method
        # is added we simply honor the request shape for callers that will
        # later filter server-side. Today this is equivalent to open_only.
        pass
    if camera_id:
        tracks = [t for t in tracks if camera_id in t.camera_ids]
    if status == "committed":
        tracks = [t for t in tracks if t.current_identity_id is not None]
    elif status == "UNKNOWN":
        tracks = [t for t in tracks if t.current_identity_id is None]
    if search:
        needle = search.casefold()
        identities = await ctx.gallery_repo.list_identities(active_only=True)
        display_by_id = {i.identity_id: i.display_name for i in identities}
        tracks = [
            t
            for t in tracks
            if t.current_identity_id is not None
            and (
                needle in t.current_identity_id.casefold()
                or needle in display_by_id.get(t.current_identity_id, "").casefold()
            )
        ]

    tracks.sort(key=lambda t: t.last_seen_at, reverse=True)
    total = len(tracks)
    page = tracks[offset : offset + limit]

    result = []
    for t in page:
        result.append(await _track_to_dict(t, ctx.keyframe_repo))
    return {"tracks": result, "count": total, "limit": limit, "offset": offset}


@router.get("/internal/global_tracks/{global_track_id}")
async def get_global_track(
    global_track_id: str,
    ctx: _LiveContext = Depends(get_context),
) -> dict[str, Any]:
    """Return details for a single global track."""
    track = await ctx.global_track_repo.get(global_track_id)
    if track is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "global_track.not_found",
                "message": f"GlobalTrack {global_track_id} not found.",
            },
        )
    return await _track_to_dict(track, ctx.keyframe_repo)


# ---------------------------------------------------------------------------
# GET /internal/identities
# ---------------------------------------------------------------------------


@router.get("/internal/identities")
async def list_identities(
    active_only: bool = Query(True, description="Only return active identities"),
    ctx: _LiveContext = Depends(get_context),
) -> dict[str, Any]:
    """Return all known named identities from the ReID gallery.

    Used by the CC identity-corrections UI to populate the identity picker
    so caregivers select from names rather than typing raw UUIDs.
    """
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
    """Lightweight liveness ping. Distinct from ``/health`` which reflects
    the pipeline run state; this endpoint is what the CC BFF polls."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _track_to_dict(track: Any, keyframe_repo: KeyframeRepository) -> dict[str, Any]:
    keyframes = await keyframe_repo.list_keyframes(global_track_id=track.global_track_id, limit=1)
    latest_minio_key: str | None = keyframes[0].minio_key if keyframes else None
    return {
        "global_track_id": track.global_track_id,
        "camera_ids": list(track.camera_ids),
        "tracklet_ids": list(track.tracklet_ids),
        "current_identity_id": track.current_identity_id,
        "started_at": track.started_at.isoformat(),
        "last_seen_at": track.last_seen_at.isoformat(),
        "state": track.state,
        "latest_keyframe_minio_key": latest_minio_key,
    }
