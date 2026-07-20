"""Trajectory read API consumed by the CC BFF for past-track annotation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from structlog import get_logger

from ..storage.base import (
    InMemoryTrajectoryRepository,
    TrajectoryRepository,
)

logger = get_logger(__name__)

router = APIRouter(tags=["trajectory-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


@dataclass
class _TrajectoryContext:
    trajectory_repo: TrajectoryRepository


_ctx: _TrajectoryContext = _TrajectoryContext(
    trajectory_repo=InMemoryTrajectoryRepository(),
)


def get_context() -> _TrajectoryContext:
    return _ctx


def set_context(trajectory_repo: TrajectoryRepository) -> None:
    global _ctx
    _ctx = _TrajectoryContext(trajectory_repo=trajectory_repo)


# ---------------------------------------------------------------------------
# GET /internal/trajectory/recent
# ---------------------------------------------------------------------------


@router.get("/internal/trajectory/recent")
async def list_recent_trajectory(
    identity_id: str | None = Query(None, description="Filter by identity"),
    ph_id: str | None = Query(None, description="Filter by PH"),
    since: str | None = Query(
        None,
        description="ISO-8601 UTC datetime; defaults to 30 minutes ago",
    ),
    limit: int = Query(200, ge=1, le=2000, description="Max points to return"),
    ctx: _TrajectoryContext = Depends(get_context),
) -> dict[str, Any]:
    """Return recent trajectory points with posture and motion energy.

    Consumed by cognitive-companion to render past-track annotation
    in the live view and the floor-plan view.
    """
    since_dt = datetime.fromisoformat(since) if since else datetime.now(UTC) - timedelta(minutes=30)

    points = await ctx.trajectory_repo.list_trajectory_points(
        identity_id=identity_id,
        ph_id=ph_id,
        after=since_dt,
        limit=limit,
    )

    return {
        "points": [
            {
                "observed_at": p.observed_at.isoformat(),
                "identity_id": p.identity_id,
                "ph_id": p.ph_id,
                "room_name": p.room_name,
                "ground_x": p.ground_x,
                "ground_y": p.ground_y,
                "posture": p.posture,
                "motion_energy": p.motion_energy,
                "identity_confidence": p.identity_confidence,
            }
            for p in points
        ],
        "count": len(points),
    }


# ---------------------------------------------------------------------------
# GET /internal/trajectory/dwells
# ---------------------------------------------------------------------------


class RoomDwellModel(BaseModel):
    room_name: str
    entered_at: str
    exited_at: str | None
    identity_id: str | None
    ph_id: str
    entry_confidence: float


class RoomDwellRangeResponse(BaseModel):
    dwells: list[RoomDwellModel]


@router.get("/internal/trajectory/dwells", response_model=RoomDwellRangeResponse)
async def list_dwells_for_ph(
    ph_id: str = Query(..., description="PH to fetch dwells for"),
    start: datetime = Query(..., description="ISO-8601 UTC range start (entered_at >=)"),
    end: datetime = Query(..., description="ISO-8601 UTC range end (entered_at <=)"),
    ctx: _TrajectoryContext = Depends(get_context),
) -> RoomDwellRangeResponse:
    """Return one PH's room dwells within an explicit time range.

    Consumed by cognitive-companion (identity-continuity M05) to project an
    ``inferred_backfill`` revision's range into presence segments.
    """
    dwells = await ctx.trajectory_repo.list_room_dwells(
        ph_id=ph_id,
        after=start,
        before=end,
        limit=1000,
    )
    return RoomDwellRangeResponse(
        dwells=[
            RoomDwellModel(
                room_name=d.room_name,
                entered_at=d.entered_at.isoformat(),
                exited_at=d.exited_at.isoformat() if d.exited_at is not None else None,
                identity_id=d.identity_id,
                ph_id=d.ph_id,
                entry_confidence=d.entry_confidence,
            )
            for d in dwells
        ]
    )
