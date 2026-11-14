"""Learned camera-to-camera transitions, read-only.

CTS learns how long people take to walk between two cameras from accepted cross-camera
handoffs (``camera_topology_edges``). The cross-camera revival gate uses those statistics; this
endpoint lets the CC admin show them next to the operator-declared overlap groups.

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
from ..tracking.world.topology import MIN_OBSERVATIONS_FOR_LEARNED

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
    # True once enough handoffs exist for the revival gate to reject a too-fast crossing.
    gates_revival: bool


class LearnedTransitionsResponse(BaseModel):
    transitions: list[LearnedTransitionOut]
    min_observations_to_gate: int


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
                gates_revival=e.observation_count >= MIN_OBSERVATIONS_FOR_LEARNED,
            )
            for e in edges
        ],
        min_observations_to_gate=MIN_OBSERVATIONS_FOR_LEARNED,
    )
