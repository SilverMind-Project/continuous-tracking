"""Homography matrix computation from calibration point correspondences.

Pure function: unit-testable without FastAPI or Triton.

``compute_homography`` is the canonical implementation for the whole CTS stack.
The Cognitive Companion BFF delegates to the orchestrator's
``/internal/calibration/homography/fit`` endpoint rather than running OpenCV
itself — keeping all spatial logic inside ``continuous-tracking``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Validation thresholds (metres).
#: Max per-point reprojection error before we reject the calibration.
RESIDUAL_ERROR_M: float = 0.5
#: Per-point error that triggers a "warning" status (still accepted).
RESIDUAL_WARN_M: float = 0.25


def compute_homography(
    pixel_points: list[list[float]],
    floor_points: list[list[float]],
) -> tuple[list[list[float]], list[float]]:
    """Fit a 3x3 homography H such that H @ [px, py, 1]' ∝ [fx, fy, 1]'.

    Uses OpenCV ``findHomography`` with the RANSAC method for robustness against
    outlier correspondences.  Each residual is the Euclidean distance (metres)
    between the reprojected pixel point and the provided floor point.

    Args:
        pixel_points: N≥4 pixel coordinates [[px, py], ...] in the camera
            frame's natural resolution (raw, unnormalised pixel values).
        floor_points: N≥4 floor-plan metre coordinates [[fx, fy], ...].
            Origin is the top-left corner of the floor-plan image; X increases
            right, Y increases downward — the same convention used by the
            floor-plan editor.

    Returns:
        ``(matrix, residuals)`` where *matrix* is the 3x3 homography as a
        nested list (row-major) and *residuals* is a per-point list of
        reprojection errors in metres.

    Raises:
        ImportError: if ``opencv-python-headless`` is not installed.
        ValueError: if fewer than 4 point pairs are provided, or if
            ``findHomography`` fails to converge.
    """
    import cv2

    if len(pixel_points) < 4 or len(floor_points) < 4:
        raise ValueError("At least 4 point pairs are required to fit a homography.")
    if len(pixel_points) != len(floor_points):
        raise ValueError(
            f"pixel_points and floor_points must have the same length "
            f"({len(pixel_points)} != {len(floor_points)})."
        )

    src: npt.NDArray[np.float64] = np.array(pixel_points, dtype=np.float64)
    dst: npt.NDArray[np.float64] = np.array(floor_points, dtype=np.float64)

    h_raw, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=0.05)
    if h_raw is None:
        raise ValueError(
            "findHomography did not converge: check that the points are not collinear "
            "and span a reasonable area of the camera view."
        )

    h: npt.NDArray[np.float64] = np.asarray(h_raw, dtype=np.float64)  # (3, 3)

    # Per-point reprojection error in floor-plan metres.
    ones = np.ones((len(src), 1), dtype=np.float64)
    src_h = np.hstack([src, ones])  # (N, 3) homogeneous pixel coords
    proj_h: npt.NDArray[np.float64] = (h @ src_h.T).T  # (N, 3)
    proj = proj_h[:, :2] / proj_h[:, 2:3]  # de-homogenise

    residuals: list[float] = [float(np.linalg.norm(proj[i] - dst[i])) for i in range(len(src))]

    matrix: list[list[float]] = h.tolist()
    return matrix, residuals


def residual_status(max_residual: float) -> str:
    """Return ``"ok"``, ``"warning"``, or ``"error"`` for a max residual value."""
    if max_residual <= RESIDUAL_WARN_M:
        return "ok"
    if max_residual <= RESIDUAL_ERROR_M:
        return "warning"
    return "error"
