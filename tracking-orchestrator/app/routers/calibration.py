"""Internal calibration endpoints consumed by the Cognitive Companion BFF.

These endpoints are NOT exposed publicly.  The CC backend is the only
authorized caller, authenticated via a short-lived service JWT.

Routes:
    POST /internal/calibration/homography
    POST /internal/calibration/privacy_zones
    POST /internal/calibration/camera_adjacency
    POST /internal/calibration/reload
    GET  /internal/calibration/status
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.calibration.state import AdjacencyEdge, CalibrationState, PrivacyZoneConfig
from app.calibration.state import calibration_state as _default_state

router = APIRouter(prefix="/internal/calibration", tags=["calibration-internal"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class HomographyRequest(BaseModel):
    camera_id: str = Field(..., min_length=1)
    matrix: list[list[float]] = Field(
        ...,
        description="3x3 homography matrix, row-major (9 floats in 3 rows of 3)",
    )
    points: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("matrix")
    @classmethod
    def _validate_matrix(cls, m: list[list[float]]) -> list[list[float]]:
        if len(m) != 3 or any(len(row) != 3 for row in m):
            raise ValueError("matrix must be 3x3")
        return m


class PrivacyZoneIn(BaseModel):
    zone_id: str = Field(..., min_length=1)
    name: str = ""
    polygon: list[list[float]] = Field(..., min_length=3)
    policy: str = Field(..., pattern=r"^(drop_detection|blur_region|mask_region)$")
    enabled: bool = True

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, pts: list[list[float]]) -> list[list[float]]:
        for pt in pts:
            if len(pt) != 2:
                raise ValueError("each polygon point must be [x, y]")
            if not all(0.0 <= v <= 1.0 for v in pt):
                raise ValueError("polygon coordinates must be normalized to [0, 1]")
        return pts


class PrivacyZonesRequest(BaseModel):
    camera_id: str = Field(..., min_length=1)
    zones: list[PrivacyZoneIn]


class AdjacencyEdgeIn(BaseModel):
    from_camera: str = Field(..., alias="from", min_length=1)
    to_camera: str = Field(..., alias="to", min_length=1)
    min_transit_s: float = Field(default=0.5, ge=0.0)
    max_transit_s: float = Field(default=30.0, ge=0.0)
    overlap: bool = False

    model_config = {"populate_by_name": True}


class AdjacencyRequest(BaseModel):
    edges: list[AdjacencyEdgeIn]


class CalibrationStatusResponse(BaseModel):
    cameras_with_homography: int
    cameras_with_privacy_zones: int
    adjacency_edge_count: int
    last_reload_at: str | None
    adjacency_edges: list[AdjacencyEdgeIn] = []


# ---------------------------------------------------------------------------
# Dependency: injectable state (facilitates testing)
# ---------------------------------------------------------------------------


def _get_state() -> CalibrationState:
    return _default_state


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/homography",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Store a pre-computed homography for a camera",
)
async def post_homography(
    body: Annotated[HomographyRequest, Body()],
) -> None:
    state = _get_state()
    await state.set_homography(
        camera_id=body.camera_id,
        matrix=body.matrix,
        meta=body.meta or {},
        points=[[p.get("x", 0), p.get("y", 0)] for p in body.points] if body.points else None,
    )


@router.get(
    "/homography/{camera_id}",
    summary="Return the stored homography for a camera (for editing)",
)
async def get_homography(camera_id: str) -> dict[str, Any]:
    state = _get_state()
    matrix = state.homographies.get(camera_id)
    if matrix is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "homography.not_found",
                "message": f"No homography stored for camera {camera_id}",
            },
        )
    meta = state.camera_meta.get(camera_id, {})
    return {
        "camera_id": camera_id,
        "matrix": matrix,
        "points": meta.get("points", []),
        "meta": {k: v for k, v in meta.items() if k != "points"},
    }


@router.post(
    "/privacy_zones",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Replace privacy zones for a camera",
)
async def post_privacy_zones(
    body: Annotated[PrivacyZonesRequest, Body()],
) -> None:
    state = _get_state()
    zones = [
        PrivacyZoneConfig(
            zone_id=z.zone_id,
            polygon=z.polygon,
            policy=z.policy,
            name=z.name,
            enabled=z.enabled,
        )
        for z in body.zones
    ]
    await state.set_privacy_zones(body.camera_id, zones)


@router.post(
    "/camera_adjacency",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Replace the full camera adjacency graph",
)
async def post_camera_adjacency(
    body: Annotated[AdjacencyRequest, Body()],
) -> None:
    state = _get_state()
    edges = [
        AdjacencyEdge(
            from_camera=e.from_camera,
            to_camera=e.to_camera,
            min_transit_s=e.min_transit_s,
            max_transit_s=e.max_transit_s,
            overlap=e.overlap,
        )
        for e in body.edges
    ]
    if any(e.max_transit_s < e.min_transit_s for e in edges):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="max_transit_s must be >= min_transit_s for every edge",
        )
    await state.set_adjacency(edges)


@router.post(
    "/reload",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Trigger a hot-reload of all calibration config",
)
async def post_reload() -> None:
    state = _get_state()
    await state.reload()


@router.get(
    "/status",
    response_model=CalibrationStatusResponse,
    summary="Return a summary of current calibration state",
)
async def get_status() -> CalibrationStatusResponse:
    state = _get_state()
    edges = [
        AdjacencyEdgeIn(
            from_camera=e.from_camera,
            to_camera=e.to_camera,
            min_transit_s=e.min_transit_s,
            max_transit_s=e.max_transit_s,
            overlap=e.overlap,
        )
        for e in state.adjacency_edges
    ]
    return CalibrationStatusResponse(
        cameras_with_homography=len(state.homographies),
        cameras_with_privacy_zones=len(state.privacy_zones),
        adjacency_edge_count=len(state.adjacency_edges),
        last_reload_at=state.last_reload_at,
        adjacency_edges=edges,
    )
