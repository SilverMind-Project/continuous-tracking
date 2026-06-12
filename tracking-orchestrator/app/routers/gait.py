"""Gait daily aggregate read API consumed by the CC BFF for mobility trends."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from structlog import get_logger

from ..storage.gait import GaitDailyRepository, InMemoryGaitDailyRepository

logger = get_logger(__name__)

router = APIRouter(tags=["gait-internal"])


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class GaitDailyRow(BaseModel):
    identity_id: str
    local_date: str = Field(description="ISO-8601 date (YYYY-MM-DD)")
    bout_count: int
    total_walking_s: float
    total_distance_m: float
    median_speed_m_s: float
    mad_speed_m_s: float
    p95_speed_m_s: float
    computed_at: str = Field(description="ISO-8601 UTC datetime")


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


@dataclass
class _GaitContext:
    gait_daily_repo: GaitDailyRepository


_ctx: _GaitContext = _GaitContext(gait_daily_repo=InMemoryGaitDailyRepository())


def get_context() -> _GaitContext:
    return _ctx


def set_context(gait_daily_repo: GaitDailyRepository) -> None:
    global _ctx
    _ctx = _GaitContext(gait_daily_repo=gait_daily_repo)


# ---------------------------------------------------------------------------
# GET /internal/gait/daily
# ---------------------------------------------------------------------------


@router.get("/internal/gait/daily", response_model=list[GaitDailyRow])
async def list_gait_daily(
    identity_id: str = Query(..., description="Identity (resident) ID"),
    since: str | None = Query(
        None,
        description="ISO-8601 date (YYYY-MM-DD); defaults to 56 days ago",
    ),
    until: str | None = Query(
        None,
        description="ISO-8601 date (YYYY-MM-DD); defaults to yesterday",
    ),
    ctx: _GaitContext = Depends(get_context),
) -> list[GaitDailyRow]:
    """Return gait daily aggregate rows for one resident.

    Consumed by cognitive-companion to build the mobility trend panel.
    """
    today = datetime.now(UTC).date()
    since_date: date = date.fromisoformat(since) if since else today - timedelta(days=56)
    until_date: date = date.fromisoformat(until) if until else today - timedelta(days=1)

    records = await ctx.gait_daily_repo.list_days(
        identity_id=identity_id,
        since=since_date,
        until=until_date,
    )

    return [
        GaitDailyRow(
            identity_id=r.identity_id,
            local_date=r.local_date.isoformat(),
            bout_count=r.bout_count,
            total_walking_s=r.total_walking_s,
            total_distance_m=r.total_distance_m,
            median_speed_m_s=r.median_speed_m_s,
            mad_speed_m_s=r.mad_speed_m_s,
            p95_speed_m_s=r.p95_speed_m_s,
            computed_at=r.computed_at.isoformat(),
        )
        for r in records
    ]
