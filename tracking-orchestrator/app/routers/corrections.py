"""Manual identity-correction endpoints consumed by the CC BFF.

Provides a single authenticated call surface for caregivers to override the
Bayesian identity assignment on a GlobalTrack. Each correction is expressed
as a synthetic :class:`IdentityRevision` with ``reason="manual"`` so the
downstream revision-stream consumers handle it identically to an automated
revision. This preserves the audit invariant that *every* identity change
flows through a revision.

The endpoint never mutates gallery entries or re-runs inference: it simply
records operator intent, updates the GlobalTrack pointer, persists the
revision, and emits it on the ``tracking.revisions`` stream. Downstream
consumers (CC ``IdentityRewriter`` + Vue ``CTSIdentityCorrectionsView``)
observe the same contract as for automatic revisions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from structlog import get_logger

from ..domain import IdentityRevision
from ..storage.base import (
    DoNotFuseRepository,
    GlobalTrackRepository,
    InMemoryDoNotFuseRepository,
    InMemoryGlobalTrackRepository,
    InMemoryTrackingRepository,
    TrackingRepository,
)
from ..transport.revision_publisher import RevisionPublisher

logger = get_logger(__name__)

router = APIRouter(tags=["corrections-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring: module-level singletons overridden at startup.
# ---------------------------------------------------------------------------


@dataclass
class _CorrectionContext:
    """Injected services needed to execute a manual correction."""

    tracking_repo: TrackingRepository
    global_track_repo: GlobalTrackRepository
    publisher: RevisionPublisher | None
    dnf_repo: DoNotFuseRepository | None = None
    merger: object | None = None  # N0: was GlobalTrackMerger, deleted


_ctx: _CorrectionContext = _CorrectionContext(
    tracking_repo=InMemoryTrackingRepository(),
    global_track_repo=InMemoryGlobalTrackRepository(),
    publisher=None,
    dnf_repo=InMemoryDoNotFuseRepository(),
    merger=None,
)


def get_context() -> _CorrectionContext:
    return _ctx


def set_context(
    tracking_repo: TrackingRepository,
    global_track_repo: GlobalTrackRepository,
    publisher: RevisionPublisher | None,
    dnf_repo: DoNotFuseRepository | None = None,
    merger: object | None = None,  # N0: was GlobalTrackMerger, deleted
) -> None:
    """Wire production repositories + publisher at startup (called from lifespan)."""
    global _ctx
    _ctx = _CorrectionContext(
        tracking_repo=tracking_repo,
        global_track_repo=global_track_repo,
        publisher=publisher,
        dnf_repo=dnf_repo,
        merger=merger,
    )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CorrectionRequest(BaseModel):
    """Body for ``POST /internal/corrections``.

    ``new_identity_id=None`` expresses "clear identity to UNKNOWN".
    """

    global_track_id: str = Field(..., min_length=1, max_length=128)
    new_identity_id: str | None = Field(default=None, max_length=128)
    actor: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(default="manual", max_length=512)
    display_name: str | None = Field(default=None, max_length=128)
    evidence: dict[str, Any] = Field(default_factory=dict)


class CorrectionResponse(BaseModel):
    revision_id: str
    global_track_id: str
    previous_identity_id: str | None
    new_identity_id: str | None
    applied_at: str


# ---------------------------------------------------------------------------
# POST /internal/corrections
# ---------------------------------------------------------------------------


@router.post("/internal/corrections", response_model=CorrectionResponse)
async def apply_correction(
    body: CorrectionRequest,
    ctx: _CorrectionContext = Depends(get_context),
) -> CorrectionResponse:
    """Apply a manual identity override for one GlobalTrack.

    Semantics
    ---------
    - The GlobalTrack is loaded; if it doesn't exist we return 404 so the UI
      can refresh its list.
    - A synthetic :class:`IdentityRevision` is constructed with the top
      candidate at ``probability=1.0`` for the target identity (caregiver
      authority supersedes the Bayesian posterior).
    - The GlobalTrack pointer is updated via ``assign_identity``.
    - The revision is persisted and published. Publishing is best-effort:
      if Redis is unreachable the HTTP call still succeeds (the local state
      is already consistent); the caller is logged for investigation.
    """
    gt = await ctx.global_track_repo.get(body.global_track_id)
    if gt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "global_track.not_found",
                "message": f"GlobalTrack {body.global_track_id} not found.",
            },
        )

    previous_identity_id = gt.current_identity_id
    new_identity_id = body.new_identity_id
    now = datetime.now(UTC)

    revision = IdentityRevision(
        revision_id=str(uuid.uuid4()),
        ph_id=body.global_track_id,  # N0: legacy global_track_id maps to ph_id
        previous_identity_id=previous_identity_id,
        new_identity_id=new_identity_id,
        actor=body.actor,
        reason=body.reason,
        applied_at=now,
        rewritten_rows=1,
        evidence=None,
    )

    # Update the GlobalTrack pointer first.
    await ctx.global_track_repo.assign_identity(
        global_track_id=body.global_track_id,
        identity_id=new_identity_id,
        candidates=None,
    )
    await ctx.tracking_repo.save_identity_revision(revision=revision)

    # Publish last so that all state is durable before any downstream consumer
    # reacts.  If Redis is unavailable we log and continue; the persistent
    # store is authoritative.
    if ctx.publisher is not None and ctx.publisher.is_connected:
        try:
            await ctx.publisher.publish(revision)
        except Exception as exc:
            logger.warning(
                "correction_publish_failed",
                revision_id=revision.revision_id,
                error=str(exc),
            )

    logger.info(
        "manual_identity_correction_applied",
        revision_id=revision.revision_id,
        global_track_id=body.global_track_id,
        previous_identity_id=previous_identity_id,
        new_identity_id=new_identity_id,
        actor=body.actor,
    )

    return CorrectionResponse(
        revision_id=revision.revision_id,
        global_track_id=body.global_track_id,
        previous_identity_id=previous_identity_id,
        new_identity_id=new_identity_id,
        applied_at=now.isoformat(),
    )


# ---------------------------------------------------------------------------
# POST /internal/corrections/unmerge_tracklet
# ---------------------------------------------------------------------------


class TrackletUnmergeRequest(BaseModel):
    """Body for ``POST /internal/corrections/unmerge_tracklet``."""

    tracklet_id: str = Field(..., min_length=1, max_length=128)
    requested_by: str = Field(default="caregiver", max_length=128)


class TrackletUnmergeResponse(BaseModel):
    tracklet_id: str
    original_global_track_id: str
    new_global_track_id: str


@router.post(
    "/internal/corrections/unmerge_tracklet",
    response_model=TrackletUnmergeResponse,
)
async def unmerge_tracklet(
    body: TrackletUnmergeRequest,
    ctx: _CorrectionContext = Depends(get_context),
) -> TrackletUnmergeResponse:
    """N0: TrackletUnmergeService was deleted; this endpoint is no longer available.

    The world tracker uses Person Hypotheses (PHs) instead of tracklets
    and global tracks. A PH-native merge/unmerge surface will be delivered
    in N1 (PH API) plus N3 (admin UI).
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail={
            "code": "unmerge.deprecated",
            "message": (
                "Tracklet unmerge is no longer available. "
                "The world tracker uses Person Hypotheses (PHs). "
                "A PH-native correction surface is coming in N1/N3."
            ),
        },
    )


# ---------------------------------------------------------------------------
# POST /internal/corrections/merge_global_tracks
# ---------------------------------------------------------------------------


class GlobalTrackMergeRequest(BaseModel):
    source_id: str
    target_id: str
    merged_by: str = "caregiver"


class GlobalTrackMergeResponse(BaseModel):
    source_id: str
    target_id: str
    merged_at: datetime


@router.post(
    "/internal/corrections/merge_global_tracks",
    response_model=GlobalTrackMergeResponse,
)
async def merge_global_tracks(
    body: GlobalTrackMergeRequest,
    ctx: _CorrectionContext = Depends(get_context),
) -> GlobalTrackMergeResponse:
    if ctx.merger is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "merge.unavailable",
                "message": "Global track merge is not configured.",
            },
        )
    await ctx.merger.merge(  # type: ignore[attr-defined]
        source_id=body.source_id,
        target_id=body.target_id,
        merged_by=body.merged_by,
    )
    return GlobalTrackMergeResponse(
        source_id=body.source_id,
        target_id=body.target_id,
        merged_at=datetime.now(UTC),
    )
