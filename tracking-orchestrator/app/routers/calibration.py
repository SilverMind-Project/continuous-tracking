"""Internal calibration endpoints consumed by the Cognitive Companion BFF.

These endpoints are NOT exposed publicly.  The CC backend is the only
authorized caller, authenticated via a short-lived service JWT.

Routes:
    POST /internal/calibration/homography/fit     - compute matrix from raw points
    POST /internal/calibration/homography         - store a pre-computed matrix
    GET  /internal/calibration/homography/{id}    - retrieve stored matrix
    POST /internal/calibration/auto/{camera_id}   - depth-based auto-calibration
    POST /internal/calibration/privacy_zones
    POST /internal/calibration/camera_adjacency
    POST /internal/calibration/reload
    GET  /internal/calibration/status
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import numpy.typing as npt
from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from structlog import get_logger

from app.calibration.homography import (
    RESIDUAL_ERROR_M,
    compute_homography,
    residual_status,
)
from app.calibration.state import AdjacencyEdge, CalibrationState, PrivacyZoneConfig
from app.calibration.state import calibration_state as _default_state

if TYPE_CHECKING:
    from app.calibration.auto_calibrator import AutoCalibrator
    from app.transport.minio_frames import MinioFrameFetcher

logger = get_logger(__name__)

router = APIRouter(prefix="/internal/calibration", tags=["calibration-internal"])

# Module-level singletons wired from the app lifespan.
_auto_calibrator: AutoCalibrator | None = None
_frame_fetcher: MinioFrameFetcher | None = None


def set_auto_calibration_context(
    auto_calibrator: AutoCalibrator | None,
    frame_fetcher: MinioFrameFetcher | None,
) -> None:
    """Wire the auto-calibrator and MinIO fetcher from the app lifespan."""
    global _auto_calibrator, _frame_fetcher
    _auto_calibrator = auto_calibrator
    _frame_fetcher = frame_fetcher


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
    floor_plan_id: str = Field(default="", description="Shared floor-plan identifier")
    image_width: int = Field(default=0, ge=0)
    image_height: int = Field(default=0, ge=0)
    max_residual_m: float = Field(default=0.0, ge=0.0)
    mean_residual_m: float = Field(default=0.0, ge=0.0)
    quality_status: str = Field(default="ok", pattern=r"^(ok|warning|error)$")
    quality_point_count: int = Field(default=0, ge=0)

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
    homography_camera_ids: list[str] = []
    cameras_with_privacy_zones: int
    adjacency_edge_count: int
    last_reload_at: str | None
    adjacency_edges: list[AdjacencyEdgeIn] = []


class CalibrationPoint(BaseModel):
    """One pixel ↔ floor-metres correspondence used to fit a homography."""

    pixel: list[float] = Field(..., min_length=2, max_length=2)
    floor_m: list[float] = Field(..., min_length=2, max_length=2)


class HomographyFitRequest(BaseModel):
    """Compute a homography from raw calibration point pairs."""

    camera_id: str = Field(..., min_length=1)
    points: list[CalibrationPoint] = Field(..., min_length=4)


class HomographyFitResult(BaseModel):
    """Result of a homography computation."""

    camera_id: str
    matrix: list[list[float]]
    residuals_m: list[float]
    max_residual_m: float
    status: str  # "ok" | "warning" | "error"
    method: str = "manual"


class AutoCalibrateRequest(BaseModel):
    """Request body for depth-based auto-calibration.

    Exactly one of ``minio_key`` or ``snapshot_bytes`` must be provided.
    ``snapshot_bytes`` is a standard base64-encoded JPEG and avoids the MinIO
    round-trip when the caller already holds the image (e.g. a fresh ingress
    snapshot fetched on demand).
    """

    minio_key: str | None = Field(
        default=None, min_length=1, description="MinIO object key for the camera frame"
    )
    snapshot_bytes: str | None = Field(
        default=None,
        description="Base64-encoded JPEG frame (alternative to minio_key).",
    )
    fov_deg: float = Field(
        default=70.0,
        ge=20.0,
        le=180.0,
        description="Camera horizontal field of view in degrees (default 70°).",
    )

    @model_validator(mode="after")
    def _check_source(self) -> AutoCalibrateRequest:
        if not self.minio_key and not self.snapshot_bytes:
            raise ValueError("Provide either minio_key or snapshot_bytes.")
        if self.minio_key and self.snapshot_bytes:
            raise ValueError("Provide minio_key or snapshot_bytes, not both.")
        return self


class AutoCalibrateResult(BaseModel):
    """Result of depth-based automatic homography estimation."""

    camera_id: str
    draft_matrix: list[list[float]]
    suggested_points: list[dict[str, list[float]]]
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Plane-fit confidence [0, 1]. Below 0.4 is unreliable.",
    )
    inlier_count: int
    sample_count: int
    fov_deg: float
    image_width: int
    image_height: int
    method: str = "depth_auto_draft"
    warning: str | None = None
    floor_region_polygon: list[list[float]] | None = Field(
        default=None,
        description=(
            "Normalised [0,1] image-space polygon tracing the detected floor. "
            "Same coordinate space as visibility_polygon (NOT floor-plan metres). "
            "Null when fewer than 3 floor inliers were detected."
        ),
    )


# ---------------------------------------------------------------------------
# Dependency: injectable state (facilitates testing)
# ---------------------------------------------------------------------------


def _get_state() -> CalibrationState:
    return _default_state


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/homography/fit",
    response_model=HomographyFitResult,
    summary="Compute and store a homography from raw calibration point pairs",
)
async def post_homography_fit(
    body: Annotated[HomographyFitRequest, Body()],
) -> HomographyFitResult:
    """Fit a 3x3 homography from pixel-to-floor-metre correspondences.

    Uses OpenCV ``findHomography`` (RANSAC) server-side.  Returns the matrix
    and per-point reprojection errors.  Also stores the result in the in-memory
    calibration state so the frame-processing pipeline picks it up immediately.

    Returns 400 with ``code="calibration.residuals_too_high"`` when the maximum
    per-point error exceeds ``RESIDUAL_ERROR_M`` (0.5 m).
    """
    pixel_pts = [p.pixel for p in body.points]
    floor_pts = [p.floor_m for p in body.points]

    try:
        matrix, residuals = compute_homography(pixel_pts, floor_pts)
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "calibration.opencv_missing",
                "message": "opencv-python-headless is not installed on the orchestrator.",
            },
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "calibration.invalid_points", "message": str(exc)},
        ) from exc

    max_residual = max(residuals)
    if max_residual > RESIDUAL_ERROR_M:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "calibration.residuals_too_high",
                "message": (
                    f"Maximum reprojection error {max_residual:.3f} m exceeds the "
                    f"{RESIDUAL_ERROR_M} m threshold. Adjust calibration points."
                ),
                "max_residual_m": max_residual,
                "residuals_m": residuals,
            },
        )

    state = _get_state()
    mean_residual = sum(residuals) / len(residuals) if residuals else 0.0
    r_status = residual_status(max_residual)
    await state.set_homography(
        camera_id=body.camera_id,
        matrix=matrix,
        meta={"max_residual_m": max_residual, "method": "manual"},
        points=[[p.pixel[0], p.pixel[1]] for p in body.points],
        max_residual_m=max_residual,
        mean_residual_m=mean_residual,
        quality_status=r_status,
        quality_point_count=len(body.points),
    )

    logger.info(
        "homography_fit",
        camera_id=body.camera_id,
        max_residual_m=round(max_residual, 4),
        point_count=len(body.points),
    )

    return HomographyFitResult(
        camera_id=body.camera_id,
        matrix=matrix,
        residuals_m=residuals,
        max_residual_m=max_residual,
        status=residual_status(max_residual),
    )


@router.post(
    "/auto/{camera_id}",
    response_model=AutoCalibrateResult,
    summary="Auto-calibrate using monocular depth estimation",
)
async def post_auto_calibrate(
    camera_id: str,
    body: Annotated[AutoCalibrateRequest, Body()],
) -> AutoCalibrateResult:
    """Estimate a homography automatically from a single camera frame.

    Downloads *minio_key* from object storage, runs Depth Anything v2 to
    obtain a metric depth map, fits the floor plane with RANSAC, and derives
    a local pixel→camera-floor-metres draft. The result is not anchored to the
    shared floor plan and is never stored as an active homography.

    Returns 503 when the depth model is not loaded (Triton unavailable).
    Returns 409 when no valid floor plane is detected.
    """
    if _auto_calibrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "calibration.depth_model_unavailable",
                "message": (
                    "Auto-calibration requires the Depth Anything v2 model to be "
                    "loaded in Triton. Check that the 'depth-anything-v2' model "
                    "repository is present and the Triton server is reachable."
                ),
            },
        )

    if body.snapshot_bytes is not None:
        import base64

        try:
            jpeg_bytes = base64.b64decode(body.snapshot_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "calibration.invalid_snapshot_bytes",
                    "message": "snapshot_bytes is not valid base64.",
                },
            ) from exc
        import cv2
        import numpy as np

        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "calibration.invalid_snapshot_bytes",
                    "message": "snapshot_bytes could not be decoded as a JPEG image.",
                },
            )
        import numpy as np

        image: npt.NDArray[np.uint8] = np.asarray(
            cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )
    else:
        fetcher: MinioFrameFetcher | None = _frame_fetcher
        if fetcher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "calibration.minio_unavailable",
                    "message": "MinIO frame storage is not configured on the orchestrator.",
                },
            )

        assert body.minio_key is not None  # enforced by AutoCalibrateRequest validator
        try:
            image = await fetcher.fetch_rgb(body.minio_key)
        except Exception as exc:
            logger.warning(
                "auto_calibration_frame_fetch_failed",
                camera_id=camera_id,
                minio_key=body.minio_key,
                error=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "calibration.frame_not_found",
                    "message": (
                        f"Could not fetch frame '{body.minio_key}' from object storage. "
                        "Ensure the camera is active and producing frames."
                    ),
                },
            ) from exc

    result = await _auto_calibrator.calibrate(image, fov_deg=body.fov_deg)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "calibration.floor_not_detected",
                "message": (
                    "Could not detect a reliable floor plane in the frame. "
                    "Ensure the camera has an unobstructed view of the floor "
                    "with good lighting, or use manual calibration."
                ),
            },
        )

    logger.info(
        "auto_calibration_draft_complete",
        camera_id=camera_id,
        confidence=round(result.confidence, 3),
        inlier_count=result.inlier_count,
        suggested_point_count=len(result.suggested_points),
        fov_deg=result.fov_deg,
    )

    warning: str | None = None
    if result.confidence < 0.5:
        warning = (
            "Confidence is below 0.5 — the homography may be inaccurate. "
            "Consider verifying with a few manual calibration points."
        )
    if body.fov_deg == 70.0:
        fov_note = (
            " Default FoV of 70 deg was used; enter your camera's actual FoV for best accuracy."
        )
        warning = (warning or "") + fov_note

    return AutoCalibrateResult(
        camera_id=camera_id,
        draft_matrix=result.draft_matrix,
        suggested_points=result.suggested_points,
        confidence=result.confidence,
        inlier_count=result.inlier_count,
        sample_count=result.sample_count,
        fov_deg=result.fov_deg,
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        method=result.method,
        warning=warning or None,
        floor_region_polygon=result.floor_region_polygon,
    )


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
        floor_plan_id=body.floor_plan_id,
        image_width=body.image_width,
        image_height=body.image_height,
        max_residual_m=body.max_residual_m,
        mean_residual_m=body.mean_residual_m,
        quality_status=body.quality_status,
        quality_point_count=body.quality_point_count,
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
    cal = state.calibrations.get(camera_id)
    result: dict[str, Any] = {
        "camera_id": camera_id,
        "matrix": matrix,
        "points": meta.get("points", []),
        "meta": {k: v for k, v in meta.items() if k != "points"},
    }
    if cal is not None:
        result["floor_plan_id"] = cal.floor_plan_id
        result["image_width"] = cal.image_width
        result["image_height"] = cal.image_height
        result["quality"] = {
            "status": cal.quality.status,
            "max_residual_m": cal.quality.max_residual_m,
            "mean_residual_m": cal.quality.mean_residual_m,
            "point_count": cal.quality.point_count,
        }
        result["calibrated_at"] = cal.calibrated_at.isoformat()
    return result


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
        homography_camera_ids=sorted(state.homographies),
        cameras_with_privacy_zones=len(state.privacy_zones),
        adjacency_edge_count=len(state.adjacency_edges),
        last_reload_at=state.last_reload_at,
        adjacency_edges=edges,
    )
