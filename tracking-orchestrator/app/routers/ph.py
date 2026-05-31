"""Person Hypothesis HTTP API.

Provides read, correction, merge, split, and audit endpoints for
Person Hypotheses.  Replaces the deleted ``/identity/global_tracks*``
and ``/identity/decisions*`` surfaces.

Actor extracted from X-Actor-Subject header. Idempotency keys
enforced for all mutation endpoints.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from structlog import get_logger

from ..domain import IdentityRevision
from ..observability.metrics import metrics
from ..storage.base import PHRepositoryProtocol
from .ph_schemas import (
    BatchCorrectRequest,
    BatchCorrectResponse,
    CorrectIdentityRequest,
    CorrectIdentityResponse,
    KeyframeResponse,
    MergeRequest,
    MergeResponse,
    ObservationResponse,
    PaginatedPHList,
    PHCoPresentItem,
    PHCoPresentResponse,
    PHDetail,
    PHKeyframesResponse,
    PHObservationsList,
    PHSummary,
    RevisionResponse,
    RevisionsFeedResponse,
    SplitRequest,
    SplitResponse,
    TrailPointResponse,
)

logger = get_logger(__name__)

router = APIRouter(tags=["ph"])


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


_repo: PHRepositoryProtocol | None = None
_revision_publisher: object | None = None  # RevisionPublisher


def set_ph_repository(repo: PHRepositoryProtocol) -> None:
    global _repo
    _repo = repo


def set_revision_publisher(publisher: object) -> None:
    """Inject RevisionPublisher for manual correction publishing."""
    global _revision_publisher
    _revision_publisher = publisher


async def get_repo() -> PHRepositoryProtocol:
    if _repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "ph.repository.not_wired", "message": "PH repository not configured"},
        )
    return _repo


async def _publish_manual_revision(revision: IdentityRevision, kind: str) -> None:
    """Publish a manual correction revision through the same revision stream."""
    if _revision_publisher is not None:
        try:
            await _revision_publisher.publish(revision)  # type: ignore[attr-defined]
            logger.info("manual_revision_published", kind=kind, revision_id=revision.revision_id)
        except Exception:
            logger.exception("manual_revision_publish_failed", revision_id=revision.revision_id)


# ---------------------------------------------------------------------------
# Static-path routes (must precede /ph/{ph_id})
# ---------------------------------------------------------------------------


@router.get("/ph", response_model=PaginatedPHList)
async def list_phs(
    since: str | None = Query(default=None, description="ISO-8601 lower bound on last_seen_at"),
    until: str | None = Query(default=None, description="ISO-8601 upper bound on first_seen_at"),
    room_id: str | None = Query(default=None),
    identity_id: str | None = Query(default=None),
    state: Literal["active", "coasting", "ended"] | None = Query(default=None),
    include_transient: bool = Query(default=False, description="Include PHs with duration < 2s"),
    min_duration_s: float | None = Query(
        default=None, ge=0, description="Minimum PH duration in seconds"
    ),
    search: str | None = Query(
        default=None, max_length=200, description="Search by identity display name (ILIKE)"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> PaginatedPHList:
    start = time.monotonic()
    try:
        since_dt = _parse_iso(since) if since else None
        until_dt = _parse_iso(until) if until else None
        items, total = await repo.list_active(
            since=since_dt,
            until=until_dt,
            room_id=room_id,
            identity_id=identity_id,
            state=state,
            include_transient=include_transient,
            min_duration_s=min_duration_s,
            search=search,
            limit=limit,
            offset=offset,
        )
        return PaginatedPHList(
            items=[PHSummary.from_domain(ph) for ph in items],
            total=total,
            limit=limit,
            offset=offset,
        )
    finally:
        metrics.cts_ph_api_latency_seconds.labels(endpoint="list").observe(time.monotonic() - start)


@router.get("/ph/revisions", response_model=RevisionsFeedResponse)
async def list_revisions(
    ph_id: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    before_id: str | None = Query(default=None),
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> RevisionsFeedResponse:
    kind_val: str | None = None
    if kind is not None:
        valid_kinds = {"auto", "manual_correct", "manual_merge", "manual_split"}
        if kind not in valid_kinds:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "ph.revisions.invalid_kind",
                    "message": f"kind must be one of {sorted(valid_kinds)}",
                },
            )
        kind_val = kind

    items, has_more = await repo.list_revisions(
        ph_id=ph_id,
        kind=kind_val,
        limit=limit,
        before_id=before_id,
    )
    return RevisionsFeedResponse(
        items=[RevisionResponse.from_domain(rev, kind=kind or "auto") for rev in items],
        has_more=has_more,
    )


@router.post("/ph/merge", response_model=MergeResponse)
async def merge_phs(
    body: MergeRequest,
    request: Request,
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> MergeResponse:
    if body.source_ph_id == body.target_ph_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "cts.ph.merge.same_ph", "message": "Source and target PH are the same"},
        )
    actor = _actor_from_request(request)
    idem_key = _idempotency_key_from_request(request)
    try:
        revision = await repo.merge(
            source_ph_id=body.source_ph_id,
            target_ph_id=body.target_ph_id,
            actor=actor,
            reason=body.reason,
            idempotency_key=idem_key,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "ph.merge.invalid", "message": str(exc)},
        ) from exc
    metrics.cts_ph_merges_total.labels(actor="operator").inc()
    await _publish_manual_revision(revision, "manual_merge")
    return MergeResponse(
        revision=RevisionResponse.from_domain(revision, kind="manual_merge"),
        source_ph_id=body.source_ph_id,
        target_ph_id=body.target_ph_id,
    )


@router.post("/ph/batch_correct", response_model=BatchCorrectResponse)
async def batch_correct(
    body: BatchCorrectRequest,
    request: Request,
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> BatchCorrectResponse:
    actor = _actor_from_request(request)
    idem_key = _idempotency_key_from_request(request)
    ph_ids = [item.ph_id for item in body.corrections]
    new_identity_ids = [item.new_identity_id for item in body.corrections]
    reasons = [item.reason for item in body.corrections]
    try:
        revs = await repo.batch_correct(
            ph_ids=ph_ids,
            new_identity_ids=new_identity_ids,
            actor=actor,
            reasons=reasons,
            idempotency_key=idem_key,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "ph.batch_correct.invalid", "message": str(exc)},
        ) from exc
    metrics.cts_ph_corrections_total.labels(actor="batch").inc(len(revs))
    for rev in revs:
        await _publish_manual_revision(rev, "manual_correct")
    return BatchCorrectResponse(
        revisions=[RevisionResponse.from_domain(r, kind="manual_correct") for r in revs],
        applied=len(revs),
        errors=[],
    )


# ---------------------------------------------------------------------------
# Dynamic-path routes (/ph/{ph_id}/...)
# ---------------------------------------------------------------------------


@router.get("/ph/{ph_id}", response_model=PHDetail)
async def get_ph(
    ph_id: str,
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> PHDetail:
    ph = await repo.get_by_id(ph_id)
    if ph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ph.not_found", "message": f"PH {ph_id} not found"},
        )
    return PHDetail.from_domain(ph)


@router.get("/ph/{ph_id}/observations", response_model=PHObservationsList)
async def list_ph_observations(
    ph_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> PHObservationsList:
    ph = await repo.get_by_id(ph_id)
    if ph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ph.not_found", "message": f"PH {ph_id} not found"},
        )
    observations = await repo.list_observations_by_ph(ph_id, limit=limit)
    return PHObservationsList(
        ph_id=ph_id,
        items=[ObservationResponse.from_domain(obs) for obs in observations],
        count=len(observations),
    )


@router.get("/ph/{ph_id}/keyframes", response_model=PHKeyframesResponse)
async def list_ph_keyframes(
    ph_id: str,
    limit: int = Query(default=24, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> PHKeyframesResponse:
    ph = await repo.get_by_id(ph_id)
    if ph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ph.not_found", "message": f"PH {ph_id} not found"},
        )
    kfs, total = await repo.get_keyframes(ph_id, limit=limit, offset=offset)
    return PHKeyframesResponse(
        ph_id=ph_id,
        items=[KeyframeResponse.from_domain(kf) for kf in kfs],
        count=total,
    )


@router.get("/ph/{ph_id}/trail", response_model=dict)
async def get_ph_trail(
    ph_id: str,
    since: str | None = Query(default=None, description="ISO-8601 lower bound"),
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> dict[str, Any]:
    ph = await repo.get_by_id(ph_id)
    if ph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ph.not_found", "message": f"PH {ph_id} not found"},
        )
    since_dt = _parse_iso(since) if since else datetime.now(UTC) - timedelta(hours=1)
    observations = await repo.list_observations_by_ph(ph_id, limit=500)
    trail = [
        TrailPointResponse.from_observation(obs)
        for obs in observations
        if obs.captured_at >= since_dt
    ]
    return {"ph_id": ph_id, "points": [t.model_dump() for t in trail], "count": len(trail)}


@router.get("/ph/{ph_id}/co_present", response_model=PHCoPresentResponse)
async def get_co_present(
    ph_id: str,
    radius_m: float = Query(default=5.0, ge=0.5, le=50.0),
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> PHCoPresentResponse:
    ph = await repo.get_by_id(ph_id)
    if ph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "ph.not_found", "message": f"PH {ph_id} not found"},
        )
    co_present = await repo.get_co_present(ph_id, radius_m=radius_m)
    return PHCoPresentResponse(
        ph_id=ph_id,
        co_present=[PHCoPresentItem.from_domain(p) for p in co_present],
        radius_m=radius_m,
    )


# ---------------------------------------------------------------------------
# Correction endpoints (dynamic)
# ---------------------------------------------------------------------------


def _actor_from_request(request: Request) -> str:
    """Extract actor identity from request headers."""
    actor = request.headers.get("X-Actor-Subject", "").strip()
    if actor:
        return actor
    # Fallback: check if there's an auth context
    auth = getattr(request.state, "auth_context", None)
    if auth:
        sub = getattr(auth, "subject", None)
        if sub:
            return str(sub)
    return "system"


def _idempotency_key_from_request(request: Request) -> str | None:
    """Extract idempotency key from request headers."""
    key = request.headers.get("X-Idempotency-Key", "").strip()
    return key if key else None


@router.post("/ph/{ph_id}/correct", response_model=CorrectIdentityResponse)
async def correct_identity(
    ph_id: str,
    body: CorrectIdentityRequest,
    request: Request,
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> CorrectIdentityResponse:
    actor = _actor_from_request(request)
    idem_key = _idempotency_key_from_request(request)
    try:
        revision = await repo.correct_identity(
            ph_id=ph_id,
            new_identity_id=body.new_identity_id,
            reason=body.reason,
            actor=actor,
            idempotency_key=idem_key,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "ph.correct.invalid", "message": str(exc)},
        ) from exc
    metrics.cts_ph_corrections_total.labels(actor="operator").inc()
    await _publish_manual_revision(revision, "manual_correct")
    return CorrectIdentityResponse(
        revision=RevisionResponse.from_domain(revision, kind="manual_correct"),
    )


@router.post("/ph/{ph_id}/split", response_model=SplitResponse)
async def split_ph(
    ph_id: str,
    body: SplitRequest,
    request: Request,
    repo: PHRepositoryProtocol = Depends(get_repo),
) -> SplitResponse:
    actor = _actor_from_request(request)
    idem_key = _idempotency_key_from_request(request)
    try:
        original_id, new_id = await repo.split(
            ph_id=ph_id,
            at_observation_id=body.at_observation_id,
            actor=actor,
            reason=body.reason,
            idempotency_key=idem_key,
        )
    except ValueError as exc:
        msg = str(exc)
        if "first observation" in msg.lower():
            raise HTTPException(
                status_code=422,
                detail={"code": "cts.ph.split.cannot_split_at_first", "message": msg},
            ) from exc
        raise HTTPException(
            status_code=422,
            detail={"code": "ph.split.invalid", "message": msg},
        ) from exc
    metrics.cts_ph_splits_total.labels(actor="operator").inc()
    # Publish a split revision (same stream as automatic corrections).
    split_revision = IdentityRevision(
        revision_id=f"manual-split-{original_id}-{new_id}",
        ph_id=original_id,
        previous_identity_id=None,
        new_identity_id=None,
        actor=actor,
        reason=body.reason,
        applied_at=datetime.now(UTC),
        rewritten_rows=1,
        evidence=None,
    )
    await _publish_manual_revision(split_revision, "manual_split")
    return SplitResponse(original_ph_id=original_id, new_ph_id=new_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
