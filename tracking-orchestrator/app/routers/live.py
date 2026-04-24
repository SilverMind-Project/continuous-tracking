"""Live view + health internal endpoints consumed by the CC BFF.

These endpoints back the Live view and the admin health widgets. Heavier,
low-rate reads live here (the hot tracking-event path is the Redis stream).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from structlog import get_logger

from ..storage.base import (
    GlobalTrackRepository,
    InMemoryGlobalTrackRepository,
)

logger = get_logger(__name__)

router = APIRouter(tags=["live-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


@dataclass
class _LiveContext:
    global_track_repo: GlobalTrackRepository
    feature_flags: dict[str, bool]


_ctx: _LiveContext = _LiveContext(
    global_track_repo=InMemoryGlobalTrackRepository(),
    feature_flags={
        "retroactive_revision_enabled": True,
        "cross_camera_association_enabled": True,
        "manual_corrections_enabled": True,
    },
)


def get_context() -> _LiveContext:
    return _ctx


def set_context(
    global_track_repo: GlobalTrackRepository,
    feature_flags: dict[str, bool] | None = None,
) -> None:
    global _ctx
    _ctx = _LiveContext(
        global_track_repo=global_track_repo,
        feature_flags=feature_flags or _ctx.feature_flags,
    )


# ---------------------------------------------------------------------------
# GET /internal/global_tracks
# ---------------------------------------------------------------------------


@router.get("/internal/global_tracks")
async def list_global_tracks(
    open_only: bool = Query(True, description="Only return tracks with state='active'"),
    ctx: _LiveContext = Depends(get_context),
) -> dict[str, Any]:
    """Return a summary of global tracks for the Live and Corrections views."""
    tracks = await ctx.global_track_repo.list_active()
    if not open_only:
        # list_active() is the only protocol accessor; until a list-all method
        # is added we simply honor the request shape for callers that will
        # later filter server-side.  Today this is equivalent to open_only.
        pass
    return {"tracks": [_track_to_dict(t) for t in tracks], "count": len(tracks)}


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
    return _track_to_dict(track)


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


def _track_to_dict(track: Any) -> dict[str, Any]:
    return {
        "global_track_id": track.global_track_id,
        "camera_ids": list(track.camera_ids),
        "tracklet_ids": list(track.tracklet_ids),
        "current_identity_id": track.current_identity_id,
        "started_at": track.started_at.isoformat(),
        "last_seen_at": track.last_seen_at.isoformat(),
        "state": track.state,
    }
