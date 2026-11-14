"""Learned camera-to-camera handoff statistics, read-only.

Every accepted cross-camera revival or continuation records the delay between the earlier
PH's close and the new sighting (``camera_topology_edges``). The delay is measured from
``closed_at``, which comes at least ``ph_close_grace_s`` after the last sighting, so it says
how often and how soon tracking hands a person from one camera to another; it is not a walk
time. This endpoint lets the CC admin show it next to the operator-declared overlap groups.

Endpoints
---------
``GET /internal/topology/transitions``
"""

from __future__ import annotations

import math
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..storage.base import CameraTopologyRepository, InMemoryCameraTopologyRepository

router = APIRouter(prefix="/internal/topology", tags=["topology-internal"])

_topology_repo: CameraTopologyRepository = InMemoryCameraTopologyRepository()


def get_topology_repo() -> CameraTopologyRepository:
    return _topology_repo


def set_topology_repo(repo: CameraTopologyRepository) -> None:
    """Install the production repository (called once from ``app.main``)."""
    global _topology_repo
    _topology_repo = repo


class LearnedTransitionOut(BaseModel):
    from_camera: str
    to_camera: str
    observation_count: int
    mean_transit_s: float
    std_transit_s: float
    last_updated_at: datetime | None


class LearnedTransitionsResponse(BaseModel):
    transitions: list[LearnedTransitionOut]


@router.get("/transitions", response_model=LearnedTransitionsResponse)
async def list_transitions(
    repo: CameraTopologyRepository = Depends(get_topology_repo),
) -> LearnedTransitionsResponse:
    """Return every learned directed camera pair, most observed first."""
    edges = sorted(
        await repo.list_edges(),
        key=lambda e: (-e.observation_count, e.from_camera, e.to_camera),
    )
    return LearnedTransitionsResponse(
        transitions=[
            LearnedTransitionOut(
                from_camera=e.from_camera,
                to_camera=e.to_camera,
                observation_count=e.observation_count,
                mean_transit_s=e.mean_transit_s,
                std_transit_s=math.sqrt(max(e.variance_transit_s2, 0.0)),
                last_updated_at=e.last_updated_at,
            )
            for e in edges
        ],
    )
