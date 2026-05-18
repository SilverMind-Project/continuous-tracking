"""Calibration state and hot-reload for the tracking orchestrator."""

from .auto_calibrator import AutoCalibrationResult, AutoCalibrator
from .floor_plane import FloorPlaneFitter, FloorPlaneResult, floor_plane_to_homography
from .homography import RESIDUAL_ERROR_M, RESIDUAL_WARN_M, compute_homography, residual_status
from .state import CalibrationState, calibration_state

__all__ = [
    "AutoCalibrationResult",
    "AutoCalibrator",
    "CalibrationState",
    "FloorPlaneFitter",
    "FloorPlaneResult",
    "RESIDUAL_ERROR_M",
    "RESIDUAL_WARN_M",
    "calibration_state",
    "compute_homography",
    "floor_plane_to_homography",
    "residual_status",
]
