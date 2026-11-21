"""Single-view standing height estimation stage.

Runs in ``pre_world_stages`` immediately after ``InferenceStage`` (so RTMPose keypoints exist)
and before ``WorldTrackingStage`` (so ``WorldObservation`` carries ``height_estimate_m`` and
``height_sigma_m`` into world tracking and identity resolution).

Skips uncalibrated detections, cameras whose recovered pose fails height or tilt validation
against their configured ``CameraProfile``, and detections that fail the upright standing gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from structlog import get_logger

from app.calibration.state import CalibrationState
from app.domain import ObservationGeometry, OrientationBin
from app.observability import metrics as _metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.base import FrameStage
from app.services.camera_profile_map import CameraProfileMap
from app.tracking.camera_pose import (
    CameraPoseValidation,
    is_upright_standing,
    standing_height_with_sigma,
    validate_camera_pose,
)
from app.tracking.world.observation_model import (
    footpoint_reliable,
    homography_jacobian,
    pixel_covariance,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class HeightEstimationConfig:
    """Geometry, gating, and uncertainty parameters for single-view standing height."""

    pose_height_tolerance_m: float = 0.25
    pose_tilt_tolerance_deg: float = 10.0
    sigma_head_px: float = 4.0
    sigma_model_m: float = 0.04
    top_margin_px: int = 4
    edge_margin_frac: float = 0.1
    max_torso_tilt_deg: float = 25.0
    min_aspect_ratio: float = 1.3
    min_height_m: float = 0.9
    max_height_m: float = 2.2
    base_footpoint_sigma_px: float = 6.0
    occluded_footpoint_inflation: float = 6.0


class CameraPoseCache:
    """Per-camera validated pose cache keyed by ``(cal_version, profile_version, w, h)``.

    Logs and counts unreliable cameras once per calibration/profile epoch so operators see
    misconfigured FOV, mount height, or tilt without flooding logs on every frame.
    """

    def __init__(
        self,
        *,
        calibration_state: CalibrationState,
        camera_profiles: CameraProfileMap | None = None,
        camera_profiles_getter: Callable[[], CameraProfileMap | None] | None = None,
        config: HeightEstimationConfig | None = None,
    ) -> None:
        self._cal = calibration_state
        self._profiles = camera_profiles
        self._profiles_getter = camera_profiles_getter
        self._config = config or HeightEstimationConfig()
        self._cache: dict[str, tuple[tuple[int, int, int, int], CameraPoseValidation]] = {}

    def _current_profiles(self) -> CameraProfileMap | None:
        if self._profiles_getter is not None:
            return self._profiles_getter()
        return self._profiles

    def get(self, camera_id: str, *, width_px: int, height_px: int) -> CameraPoseValidation:
        """Return cached validated pose for *camera_id* at ``(width_px, height_px)``."""
        profiles = self._current_profiles()
        cal_ver = self._cal.version
        prof_ver = profiles.version_sync() if profiles is not None else 0
        key = (cal_ver, prof_ver, width_px, height_px)
        cached = self._cache.get(camera_id)
        if cached is not None and cached[0] == key:
            return cached[1]

        profile = profiles.get_sync(camera_id) if profiles is not None else None
        homography = self._cal.homographies.get(camera_id)

        validation = validate_camera_pose(
            camera_id,
            homography,
            image_width_px=width_px,
            image_height_px=height_px,
            horizontal_fov_deg=profile.horizontal_fov_deg if profile is not None else None,
            mounting_height_m=profile.mounting_height_m if profile is not None else None,
            tilt_deg=profile.tilt_deg if profile is not None else None,
            pose_height_tolerance_m=self._config.pose_height_tolerance_m,
            pose_tilt_tolerance_deg=self._config.pose_tilt_tolerance_deg,
        )
        self._cache[camera_id] = (key, validation)

        if not validation.reliable and validation.reason is not None:
            _metrics.metrics.camera_height_pose_unreliable_total.labels(
                reason=validation.reason.value
            ).inc()
            logger.warning(
                "camera_height_pose_unreliable",
                camera_id=camera_id,
                reason=validation.reason.value,
                recovered_height_m=(
                    round(validation.recovered_height_m, 3)
                    if validation.recovered_height_m is not None
                    else None
                ),
                expected_height_m=validation.expected_height_m,
                recovered_tilt_deg=(
                    round(validation.recovered_tilt_deg, 2)
                    if validation.recovered_tilt_deg is not None
                    else None
                ),
                expected_tilt_deg=validation.expected_tilt_deg,
            )
        elif validation.reliable:
            logger.info(
                "camera_height_pose_validated",
                camera_id=camera_id,
                recovered_height_m=(
                    round(validation.recovered_height_m, 3)
                    if validation.recovered_height_m is not None
                    else None
                ),
                recovered_tilt_deg=(
                    round(validation.recovered_tilt_deg, 2)
                    if validation.recovered_tilt_deg is not None
                    else None
                ),
            )
        return validation

    def snapshot_all(self) -> dict[str, CameraPoseValidation]:
        """Return validation status for every calibrated camera (for diagnostics)."""
        result: dict[str, CameraPoseValidation] = {}
        profiles = self._current_profiles()
        for camera_id in sorted(self._cal.homographies):
            cal = self._cal.calibrations.get(camera_id)
            obs_size = self._cal.observed_stream_sizes.get(camera_id)
            profile = profiles.get_sync(camera_id) if profiles is not None else None
            w = 0
            h = 0
            if obs_size is not None:
                w, h = obs_size
            elif cal is not None and cal.image_width > 0 and cal.image_height > 0:
                w, h = cal.image_width, cal.image_height
            elif (
                profile is not None
                and profile.frame_natural_width_px
                and profile.frame_natural_height_px
            ):
                w, h = profile.frame_natural_width_px, profile.frame_natural_height_px
            result[camera_id] = self.get(camera_id, width_px=w, height_px=h)
        return result


class HeightEstimationStage(FrameStage):
    """Estimate per-detection standing height and uncertainty before world tracking."""

    name = "height_estimation"

    def __init__(
        self,
        *,
        calibration_state: CalibrationState,
        camera_profiles: CameraProfileMap | None = None,
        camera_profiles_getter: Callable[[], CameraProfileMap | None] | None = None,
        pose_cache: CameraPoseCache | None = None,
        config: HeightEstimationConfig | None = None,
    ) -> None:
        self._cal = calibration_state
        self._config = config or HeightEstimationConfig()
        self._pose_cache = pose_cache or CameraPoseCache(
            calibration_state=calibration_state,
            camera_profiles=camera_profiles,
            camera_profiles_getter=camera_profiles_getter,
            config=self._config,
        )

    @property
    def pose_cache(self) -> CameraPoseCache:
        return self._pose_cache

    async def run(self, ctx: FrameContext) -> None:
        ctx.height_by_detection = {}
        if not ctx.domain_detections:
            return

        camera_id = ctx.frame.camera_id
        validation = self._pose_cache.get(
            camera_id,
            width_px=ctx.effective_width,
            height_px=ctx.effective_height,
        )
        if not validation.reliable or validation.pose is None:
            return

        homography = self._cal.homographies.get(camera_id)
        if not homography:
            return
        h_mat = np.asarray(homography, dtype=np.float64)

        pose = validation.pose
        cfg = self._config

        for det in ctx.domain_detections:
            if not det.floor_point.calibrated:
                continue
            pose_res = ctx.det_pose_result.get(det.detection_id)
            if not is_upright_standing(
                pose_res,
                det.bbox,
                frame_width_px=ctx.effective_width,
                frame_height_px=ctx.effective_height,
                top_margin_px=cfg.top_margin_px,
                edge_margin_frac=cfg.edge_margin_frac,
                max_torso_tilt_deg=cfg.max_torso_tilt_deg,
                min_aspect_ratio=cfg.min_aspect_ratio,
            ):
                continue

            fx_px = (det.bbox.x_min + det.bbox.x_max) / 2.0
            fy_px = float(det.bbox.y_max)
            head_px = (fx_px, float(det.bbox.y_min))
            foot_floor_m = (det.floor_point.x_mm / 1000.0, det.floor_point.y_mm / 1000.0)

            orientation, orientation_conf = ctx.orientation_by_detection.get(
                det.detection_id, (OrientationBin.UNKNOWN, 0.0)
            )
            reliable_fp = footpoint_reliable(
                pose_res,
                (det.bbox.x_min, det.bbox.y_min, det.bbox.x_max, det.bbox.y_max),
                ctx.effective_width,
                ctx.effective_height,
            )
            geom = ObservationGeometry(
                footpoint_px=(fx_px, fy_px),
                floor_residual_m=det.floor_residual_m if det.floor_residual_m is not None else 0.0,
                footpoint_reliable=reliable_fp,
                detection_confidence=det.confidence,
                crop_quality=det.crop_quality,
                orientation=orientation,
                orientation_confidence=orientation_conf,
            )
            px_cov = pixel_covariance(
                geom,
                base_sigma_px=cfg.base_footpoint_sigma_px,
                occluded_inflation=cfg.occluded_footpoint_inflation,
            )
            foot_cov_m2 = None
            try:
                jac = homography_jacobian(h_mat, fx_px, fy_px)
                foot_cov_m2 = jac @ px_cov @ jac.T
            except ValueError:
                foot_cov_m2 = None

            estimate = standing_height_with_sigma(
                pose,
                foot_floor_m,
                foot_cov_m2,
                head_px,
                sigma_head_px=cfg.sigma_head_px,
                sigma_model_m=cfg.sigma_model_m,
                min_height_m=cfg.min_height_m,
                max_height_m=cfg.max_height_m,
            )
            if estimate is not None:
                ctx.height_by_detection[det.detection_id] = estimate
