"""Internal read-only endpoints for per-identity standing height profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.domain import HeightProfileSource, IdentityHeightProfile
from app.storage.height_profiles import (
    IdentityHeightProfileRepository,
    InMemoryIdentityHeightProfileRepository,
)

router = APIRouter(prefix="/internal/identity", tags=["identity-height-profiles-internal"])


@dataclass
class _HeightProfileContext:
    repo: IdentityHeightProfileRepository


_ctx = _HeightProfileContext(repo=InMemoryIdentityHeightProfileRepository())


def set_context(repo: IdentityHeightProfileRepository) -> None:
    """Wire production repository from lifespan."""
    global _ctx
    _ctx = _HeightProfileContext(repo=repo)


def get_context() -> _HeightProfileContext:
    return _ctx


class HeightProfileResponse(BaseModel):
    identity_id: str
    height_m: float
    height_sigma_m: float
    source: HeightProfileSource
    samples: int
    cameras: int
    updated_at: datetime


class HeightProfileListResponse(BaseModel):
    profiles: list[HeightProfileResponse]


def _to_response(p: IdentityHeightProfile) -> HeightProfileResponse:
    return HeightProfileResponse(
        identity_id=p.identity_id,
        height_m=p.height_m,
        height_sigma_m=p.height_sigma_m,
        source=p.source,
        samples=p.samples,
        cameras=p.cameras,
        updated_at=p.updated_at,
    )


@router.get("/height-profiles", response_model=HeightProfileListResponse)
async def list_height_profiles(
    ctx: _HeightProfileContext = Depends(get_context),
) -> HeightProfileListResponse:
    """List every stored identity height profile (learned or operator-entered)."""
    profiles = await ctx.repo.list_all()
    return HeightProfileListResponse(profiles=[_to_response(p) for p in profiles])


@router.get("/height-profiles/{identity_id}", response_model=HeightProfileResponse)
async def get_height_profile(
    identity_id: str,
    ctx: _HeightProfileContext = Depends(get_context),
) -> HeightProfileResponse:
    """Fetch the effective standing height profile for one identity."""
    profile = await ctx.repo.get(identity_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "height_profile.not_found",
                "message": "No height profile recorded for this identity.",
            },
        )
    return _to_response(profile)
