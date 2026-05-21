"""Depth-based posture detection for fully occluded persons.

Uses Depth Anything v2 to estimate a metric depth map, then analyses
the depth blob within the YOLO bounding box to classify posture.

This is a slow-path strategy — intended to run at low frequency (e.g.,
every 10-30 seconds per tracklet), not every frame.

Algorithm:
  1. Run DepthEstimator on the full frame to get a (H*W) float32 depth map.
  2. Crop the depth map to the detection's bounding box.
  3. Compute the principal axis of the depth blob via PCA on (y, x) coordinates
     weighted by inverse depth (closer points = more prominent).
  4. Map principal-axis angle to posture class:
     - Near-vertical (>60° from horizontal) → 'standing'
     - Near-horizontal (<30° from horizontal) → 'lying'
     - Intermediate → 'sitting' (conservative)
     - Insufficient depth contrast → 'unknown'
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from ..domain import Detection, PostureType
from ..inference.depth import DepthEstimator
from ..observability import metrics as _metrics

# Angle thresholds (degrees from horizontal)
_VERTICAL_DEG = 60.0
_HORIZONTAL_DEG = 30.0
_MIN_DEPTH_VARIANCE = 0.05  # metres; below this the blob is featureless


class DepthPostureStrategy:
    """Depth Anything v2 — based posture classification."""

    def __init__(self, depth_estimator: DepthEstimator) -> None:
        self._depth = depth_estimator

    @property
    def name(self) -> str:
        return "depth"

    async def infer(
        self,
        frame: npt.NDArray[np.uint8],
        detection: Detection,
        pose_result: object | None = None,
    ) -> PostureType:
        try:
            t0 = time.monotonic()
            depth_map = await self._depth.estimate(frame)
            _metrics.metrics.cts_posture_slow_path_runs_total.labels(
                camera_id=detection.camera_id,
            ).inc()
            elapsed = time.monotonic() - t0
            _metrics.metrics.cts_posture_slow_path_latency_seconds.observe(elapsed)
            return self._classify_from_depth(depth_map, detection)
        except Exception:
            return "unknown"

    def _classify_from_depth(
        self,
        depth_map: npt.NDArray[np.float32],
        detection: Detection,
    ) -> PostureType:
        bbox = detection.bbox
        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(depth_map.shape[1], int(bbox.x_max))
        y2 = min(depth_map.shape[0], int(bbox.y_max))

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        crop = depth_map[y1:y2, x1:x2]

        if crop.std() < _MIN_DEPTH_VARIANCE:
            return "unknown"

        # PCA on pixel coordinates weighted by inverse depth (foreground emphasis)
        ys, xs = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
        weights = 1.0 / (crop + 1e-6)
        weights /= weights.sum()

        mean_y = (ys * weights).sum()
        mean_x = (xs * weights).sum()

        dy = ys - mean_y
        dx = xs - mean_x

        cov_yy = (weights * dy * dy).sum()
        cov_xx = (weights * dx * dx).sum()
        cov_yx = (weights * dy * dx).sum()

        # Principal axis angle in degrees from horizontal.
        # Standard PCA: tan(2*theta) = 2*cov_xy / (cov_xx - cov_yy).
        angle_rad = 0.5 * np.arctan2(2 * cov_yx, cov_xx - cov_yy)
        angle_deg = abs(np.degrees(angle_rad))

        if angle_deg > _VERTICAL_DEG:
            return "standing"
        elif angle_deg < _HORIZONTAL_DEG:
            return "lying"
        else:
            return "sitting"

    def evict_tracklet(self, tracklet_id: str) -> None:
        """No-op — DepthPostureStrategy has no per-tracklet cache."""
        pass
