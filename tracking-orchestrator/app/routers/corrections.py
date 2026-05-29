"""Manual identity-correction endpoints consumed by the CC BFF.

Provides a single authenticated call surface for caregivers to override the
Bayesian identity assignment on a PersonHypothesis. Each correction is expressed
as a synthetic IdentityRevision with reason="manual" so the downstream
revision-stream consumers handle it identically to an automated revision.

The endpoint calls ph_repo.correct_identity(), which atomically updates the PH's
identity and persists the IdentityRevision, then publishes the revision to Redis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from structlog import get_logger

from ..domain import IdentityRevision
from ..storage.base import (
    InMemoryPHRepository,
    PHRepositoryProtocol,
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

    ph_repo: PHRepositoryProtocol
    publisher: RevisionPublisher | None


_ctx: _CorrectionContext = _CorrectionContext(
    ph_repo=InMemoryPHRepository(),
    publisher=None,
)


def get_context() -> _CorrectionContext:
    return _ctx


def set_context(
    ph_repo: PHRepositoryProtocol,
    publisher: RevisionPublisher | None,
) -> None:
    """Wire production repositories + publisher at startup (called from lifespan)."""
    global _ctx
    _ctx = _CorrectionContext(ph_repo=ph_repo, publisher=publisher)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CorrectionRequest(BaseModel):
    """Body for POST /internal/corrections."""

    ph_id: str = Field(..., min_length=1, max_length=128)
    new_identity_id: str | None = Field(default=None, max_length=128)
    actor: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(default="manual", max_length=512)
    display_name: str | None = Field(default=None, max_length=128)
    evidence: dict[str, Any] = Field(default_factory=dict)


class CorrectionResponse(BaseModel):
    revision_id: str
    ph_id: str
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
    """Apply a manual identity override for one PersonHypothesis.

    Semantics
    ---------
    - The PH is loaded; if it doesn't exist we return 404 so the UI can refresh.
    - ph_repo.correct_identity() atomically updates the PH identity and persists
      the IdentityRevision.
    - The revision is published to Redis. Publishing is best-effort: if Redis is
      unreachable the HTTP call still succeeds (local state is already consistent).
    """
    ph = await ctx.ph_repo.get(body.ph_id)
    if ph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "ph.not_found",
                "message": f"PH {body.ph_id} not found.",
            },
        )

    previous_identity_id = ph.current_identity_id

    try:
        revision: IdentityRevision = await ctx.ph_repo.correct_identity(
            body.ph_id,
            new_identity_id=body.new_identity_id,
            reason=body.reason,
            actor=body.actor,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "correction.rejected", "message": str(exc)},
        ) from exc

    if ctx.publisher is not None and ctx.publisher.is_connected:
        try:
            await ctx.publisher.publish(revision)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "correction_publish_failed",
                revision_id=revision.revision_id,
                error=str(exc),
            )

    logger.info(
        "manual_identity_correction_applied",
        revision_id=revision.revision_id,
        ph_id=body.ph_id,
        previous_identity_id=previous_identity_id,
        new_identity_id=body.new_identity_id,
        actor=body.actor,
    )

    return CorrectionResponse(
        revision_id=revision.revision_id,
        ph_id=body.ph_id,
        previous_identity_id=previous_identity_id,
        new_identity_id=body.new_identity_id,
        applied_at=revision.applied_at.isoformat(),
    )
