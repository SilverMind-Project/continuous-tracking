"""Trajectory read API consumed by the CC BFF for past-track annotation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
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
    global_track_id: str | None = Query(None, description="Filter by global track"),
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
        global_track_id=global_track_id,
        after=since_dt,
        limit=limit,
    )

    return {
        "points": [
            {
                "observed_at": p.observed_at.isoformat(),
                "identity_id": p.identity_id,
                "global_track_id": p.global_track_id,
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
