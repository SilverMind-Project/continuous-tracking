"""Floor plane detection from a metric depth map via RANSAC.

Algorithm
---------
1. Back-project each pixel into camera space using the pinhole model and
   the absolute depth value.
2. Sample candidate floor pixels from the lower portion of the image
   (where the floor is most likely to appear).
3. Run RANSAC to fit the dominant flat plane within those 3-D points.
4. Return the plane normal, offset, inlier mask, and a confidence score.

The floor coordinate system produced by this module follows the same
convention as the manual calibration view:

* Origin at the floor point directly below the camera (or the centroid of
  inlier 3-D points projected onto the fitted plane if the camera is not
  directly above the floor centre).
* X axis: component of the camera's right vector projected onto the floor plane
  (→ increases in the image-right direction when the camera is roughly horizontal).
* Y axis: component of the camera's down vector projected onto the floor plane.

This matches the floor-plan editor where the origin is the top-left corner,
X increases right, and Y increases downward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorPlaneResult:
    """Output from :class:`FloorPlaneFitter`."""

    #: Unit normal of the fitted floor plane (camera frame, pointing upward).
    normal: npt.NDArray[np.float64]
    #: Plane offset: for any 3-D point X on the plane, ``normal @ X + d = 0``.
    d: float
    #: Boolean mask over the *sampled* candidate pixels (True = inlier).
    inlier_mask: npt.NDArray[np.bool_]
    #: Row/col indices of the sampled pixels, shape (N, 2).
    sample_indices: npt.NDArray[np.int64]
    #: 3-D points for each sampled pixel, shape (N, 3) in camera metres.
    points_3d: npt.NDArray[np.float64]
    #: Fraction of sampled points that are inliers [0, 1].
    inlier_ratio: float
    #: Mean absolute distance (metres) of inlier points from the plane.
    mean_inlier_distance: float

    @property
    def confidence(self) -> float:
        """Scalar confidence in [0, 1]: higher is more reliable.

        Combines inlier ratio with plane tightness.  Returns 0.0 when there
        are no inliers.
        """
        if self.inlier_ratio == 0.0:
            return 0.0
        # Tightness score: 1.0 at 0 m mean error, decays toward 0 at 0.2 m.
        tightness = max(0.0, 1.0 - self.mean_inlier_distance / 0.2)
        return float(self.inlier_ratio * tightness)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


class FloorPlaneFitter:
    """Detect the floor plane in a depth map using RANSAC.

    Parameters
    ----------
    fov_deg:
        Horizontal field of view of the camera in degrees.  Used to derive
        the focal length: ``fx = fy = W / (2 · tan(fov/2))``.  Typical
        surveillance cameras have 60-90°.  Defaults to 70°.
    floor_region_fraction:
        Fraction of the image height to use as the floor-candidate region,
        measured from the bottom of the image.  Defaults to 0.6 (lower 60%).
    min_depth_m:
        Minimum reliable depth value.  Pixels below this threshold are
        discarded before sampling.
    max_depth_m:
        Maximum reliable depth value.  Pixels above this threshold are
        discarded.
    ransac_iterations:
        Number of RANSAC trials.
    ransac_threshold_m:
        Maximum point-to-plane distance (metres) to count as an inlier.
    max_samples:
        Maximum number of candidate pixels to sample (uniform random).
        Reduces computation on high-resolution frames.
    """

    def __init__(
        self,
        fov_deg: float = 70.0,
        floor_region_fraction: float = 0.75,
        min_depth_m: float = 0.3,
        max_depth_m: float = 15.0,
        ransac_iterations: int = 256,
        ransac_threshold_m: float = 0.05,
        max_samples: int = 4096,
    ) -> None:
        self._fov_deg = fov_deg
        self._floor_region_fraction = floor_region_fraction
        self._min_depth = min_depth_m
        self._max_depth = max_depth_m
        self._ransac_iters = ransac_iterations
        self._ransac_threshold = ransac_threshold_m
        self._max_samples = max_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        depth_map: npt.NDArray[np.float32],
        rng: np.random.Generator | None = None,
    ) -> FloorPlaneResult | None:
        """Fit a floor plane to *depth_map* and return the result.

        Args:
            depth_map: ``(H, W)`` absolute metric depth in metres.
            rng: Optional seeded RNG for reproducible tests.

        Returns:
            :class:`FloorPlaneResult` on success, or ``None`` when the depth
            map contains too few valid pixels (fewer than 6 after filtering).
        """
        rng = rng or np.random.default_rng()

        h, w = depth_map.shape[:2]
        fx = fy = w / (2.0 * math.tan(math.radians(self._fov_deg / 2.0)))
        cx, cy = w / 2.0, h / 2.0

        pts3d, row_col = self._back_project(depth_map, fx, fy, cx, cy, h, w)
        if pts3d.shape[0] < 6:
            return None

        # Limit sample size for speed.
        if pts3d.shape[0] > self._max_samples:
            idx = rng.choice(pts3d.shape[0], self._max_samples, replace=False)
            pts3d = pts3d[idx]
            row_col = row_col[idx]

        normal, d = self._ransac(pts3d, rng)

        dists = np.abs(pts3d @ normal + d)
        inlier_mask: npt.NDArray[np.bool_] = dists < self._ransac_threshold
        inlier_count = int(inlier_mask.sum())

        inlier_ratio = inlier_count / len(pts3d)
        mean_inlier_dist = float(dists[inlier_mask].mean()) if inlier_count > 0 else 0.0

        return FloorPlaneResult(
            normal=normal,
            d=d,
            inlier_mask=inlier_mask,
            sample_indices=row_col,
            points_3d=pts3d,
            inlier_ratio=inlier_ratio,
            mean_inlier_distance=mean_inlier_dist,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _back_project(
        self,
        depth_map: npt.NDArray[np.float32],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        h: int,
        w: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Return valid 3-D points in the floor-candidate region."""
        floor_row_start = int(h * (1.0 - self._floor_region_fraction))

        rows_2d, cols_2d = np.meshgrid(
            np.arange(floor_row_start, h),
            np.arange(w),
            indexing="ij",
        )
        rows = rows_2d.ravel()
        cols = cols_2d.ravel()

        z: npt.NDArray[np.float64] = depth_map[rows, cols].astype(np.float64)
        valid = (z > self._min_depth) & (z < self._max_depth)
        rows, cols, z = rows[valid], cols[valid], z[valid]

        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy

        pts3d = np.stack([x, y, z], axis=1)
        row_col = np.stack([rows, cols], axis=1).astype(np.int64)
        return pts3d, row_col

    def _ransac(
        self,
        pts: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Return the (normal, d) of the plane with the most inliers."""
        n = pts.shape[0]
        best_count = -1
        best_normal = np.array([0.0, -1.0, 0.0])  # gravity direction fallback
        best_d = 0.0

        for _ in range(self._ransac_iters):
            idx = rng.choice(n, 3, replace=False)
            sample = pts[idx]
            normal, d = _fit_plane_3pts(sample[0], sample[1], sample[2])
            if normal is None or d is None:
                continue
            dists = np.abs(pts @ normal + d)
            count = int((dists < self._ransac_threshold).sum())
            if count > best_count:
                best_count = count
                best_normal = normal
                best_d = d

        # Refit using all inliers of the best plane.
        dists = np.abs(pts @ best_normal + best_d)
        inliers = pts[dists < self._ransac_threshold]
        if inliers.shape[0] >= 3:
            refined_normal, refined_d = _fit_plane_svd(inliers)
            if refined_normal is not None and refined_d is not None:
                return refined_normal, refined_d

        return best_normal, best_d


# ---------------------------------------------------------------------------
# Plane geometry helpers
# ---------------------------------------------------------------------------


def _fit_plane_3pts(
    p0: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float] | tuple[None, None]:
    """Fit a plane through three points.  Returns (unit_normal, d) or (None, None)."""
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    norm = float(np.linalg.norm(n))
    if norm < 1e-8:
        return None, None
    n = n / norm
    d = float(-n @ p0)
    return n, d


def _fit_plane_svd(
    pts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float] | tuple[None, None]:
    """Least-squares plane fit via SVD.  Returns (unit_normal, d) or (None, None)."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n: npt.NDArray[np.float64] = vt[-1]  # smallest singular value = normal
    norm = float(np.linalg.norm(n))
    if norm < 1e-8:
        return None, None
    n = n / norm
    d = float(-n @ centroid)
    return n, d


# ---------------------------------------------------------------------------
# Floor homography from plane result
# ---------------------------------------------------------------------------


def floor_plane_to_homography(
    result: FloorPlaneResult,
    image_h: int,
    image_w: int,
    fov_deg: float = 70.0,
) -> list[list[float]] | None:
    """Derive a pixel→floor-metres homography from a :class:`FloorPlaneResult`.

    Uses inlier 3-D points and their camera-frame pixel positions to compute
    a 2-D floor coordinate frame on the fitted plane.  The floor coordinate
    system origin is placed at the 3-D centroid of the inlier points projected
    onto the plane.  The X axis is the camera-right direction projected onto
    the plane; the Y axis is orthogonal (roughly camera-down on the plane).

    Returns the 3x3 homography as a nested list (row-major), or ``None`` if
    fewer than 4 inlier correspondences are available.
    """
    import cv2

    inlier_pts3d = result.points_3d[result.inlier_mask]
    inlier_rc = result.sample_indices[result.inlier_mask]
    if len(inlier_pts3d) < 4:
        return None

    # -- Build a 2-D floor coordinate frame on the plane --------------------

    # Project each inlier 3-D point onto the plane (remove component along normal).
    n = result.normal
    offsets: npt.NDArray[np.float64] = (inlier_pts3d @ n + result.d)[:, np.newaxis] * n
    projected: npt.NDArray[np.float64] = inlier_pts3d - offsets

    # Origin = centroid of projected points.
    origin = projected.mean(axis=0)

    # X axis: camera right ([1, 0, 0]) projected onto the plane and normalised.
    cam_right = np.array([1.0, 0.0, 0.0])
    x_axis = cam_right - float(cam_right @ n) * n
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-6:
        # Degenerate: camera looking straight down; use arbitrary tangent.
        x_axis = np.array([1.0, 0.0, 0.0]) - n[0] * n
        x_norm = float(np.linalg.norm(x_axis))
    x_axis = x_axis / x_norm

    # Y axis: cross product (normal x x_axis) gives the other in-plane direction.
    y_axis: npt.NDArray[np.float64] = np.cross(n, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-6:
        return None
    y_axis = y_axis / y_norm

    # -- Compute pixel ↔ floor-metres correspondences -----------------------

    rel: npt.NDArray[np.float64] = projected - origin
    floor_x: npt.NDArray[np.float64] = rel @ x_axis
    floor_y: npt.NDArray[np.float64] = rel @ y_axis

    # Pixel coords (unnormalised, in image space).
    fx_px = image_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    fy_px = fx_px  # square pixels assumed
    cx_px, cy_px = image_w / 2.0, image_h / 2.0

    cols = inlier_rc[:, 1].astype(np.float64)
    rows = inlier_rc[:, 0].astype(np.float64)

    # Back to original depth scale for the pixel coords.
    # (inlier_pts3d already carries the correct Z = depth.)
    z = inlier_pts3d[:, 2]

    pixel_u = cols  # direct column index = pixel x
    pixel_v = rows  # direct row index = pixel y
    _ = (z, fx_px, fy_px, cx_px, cy_px)  # intrinsics used for back-projection only

    src = np.stack([pixel_u, pixel_v], axis=1).astype(np.float64)
    dst = np.stack([floor_x, floor_y], axis=1).astype(np.float64)

    # Sub-sample to at most 256 correspondences so cv2.findHomography is fast.
    if len(src) > 256:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(src), 256, replace=False)
        src, dst = src[idx], dst[idx]

    h_raw, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=0.05)
    if h_raw is None:
        return None

    return h_raw.tolist()  # type: ignore[no-any-return]
