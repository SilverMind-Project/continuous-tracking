"""2D similarity transform from vendor RTLS coordinates to the shared floor frame.

Fits ``p_floor = scale * R(rotation_rad) @ p_native + [tx_m, ty_m]`` by closed-form
least squares (2D Procrustes / Umeyama with proper rotation ``det(R) = +1``) over
two or more surveyed anchor point pairs, and propagates native 2x2 covariances
into the shared floor frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform2D:
    """Calibrated 2D similarity transform (metres to metres in floor frame)."""

    scale: float
    rotation_rad: float
    tx_m: float
    ty_m: float
    rms_residual_m: float
    point_count: int

    def transform_point(self, x_native_m: float, y_native_m: float) -> tuple[float, float]:
        """Map a native-frame point (metres) into the shared floor frame (metres)."""
        c = math.cos(self.rotation_rad)
        s = math.sin(self.rotation_rad)
        x_floor = self.scale * (c * x_native_m - s * y_native_m) + self.tx_m
        y_floor = self.scale * (s * x_native_m + c * y_native_m) + self.ty_m
        return (x_floor, y_floor)

    def transform_covariance(
        self,
        cov_xx_m2: float,
        cov_xy_m2: float,
        cov_yy_m2: float,
    ) -> tuple[float, float, float]:
        """Propagate a native-frame 2x2 covariance (m^2) into the floor frame."""
        c = math.cos(self.rotation_rad)
        s = math.sin(self.rotation_rad)
        a = self.scale * np.array([[c, -s], [s, c]], dtype=np.float64)
        cov_native = np.array(
            [[cov_xx_m2, cov_xy_m2], [cov_xy_m2, cov_yy_m2]],
            dtype=np.float64,
        )
        cov_floor = a @ cov_native @ a.T
        xx = float(cov_floor[0, 0])
        xy = float(0.5 * (cov_floor[0, 1] + cov_floor[1, 0]))
        yy = float(cov_floor[1, 1])
        return (xx, xy, yy)


def fit_similarity_transform(
    native_points_m: Sequence[tuple[float, float]],
    floor_points_m: Sequence[tuple[float, float]],
) -> SimilarityTransform2D:
    """Fit a 2D similarity transform from *native_points_m* to *floor_points_m*.

    Requires at least two corresponding point pairs with non-zero spatial spread.
    """
    if len(native_points_m) != len(floor_points_m):
        raise ValueError("native_points_m and floor_points_m must have identical length")
    if len(native_points_m) < 2:
        raise ValueError("At least 2 survey point pairs are required to fit a 2D transform")

    src = np.asarray(native_points_m, dtype=np.float64)
    dst = np.asarray(floor_points_m, dtype=np.float64)
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("Survey point coordinates must be finite")

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    denom = float(np.sum(src_c**2))
    if denom <= 1e-12:
        raise ValueError("Native survey points are coincident; cannot fit similarity transform")

    # Closed-form optimal 2D proper similarity: a = s*cos(theta), b = s*sin(theta).
    a = float(np.sum(src_c[:, 0] * dst_c[:, 0] + src_c[:, 1] * dst_c[:, 1])) / denom
    b = float(np.sum(src_c[:, 0] * dst_c[:, 1] - src_c[:, 1] * dst_c[:, 0])) / denom
    scale = math.hypot(a, b)
    if scale <= 1e-9:
        raise ValueError("Degenerate survey correspondences produced zero scale")

    rotation_rad = math.atan2(b, a)
    c = math.cos(rotation_rad)
    s = math.sin(rotation_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = dst_mean - scale * (rot @ src_mean)

    pred = (scale * (src @ rot.T)) + t[None, :]
    sq_err = np.sum((pred - dst) ** 2, axis=1)
    rms_residual_m = float(math.sqrt(float(np.mean(sq_err))))

    return SimilarityTransform2D(
        scale=scale,
        rotation_rad=rotation_rad,
        tx_m=float(t[0]),
        ty_m=float(t[1]),
        rms_residual_m=rms_residual_m,
        point_count=len(native_points_m),
    )
