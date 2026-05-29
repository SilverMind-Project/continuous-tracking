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
from structlog import get_logger

from ..domain import Detection, PostureType
from ..inference.depth import DepthEstimator
from ..observability import metrics as _metrics
from .posture import _MIN_EVIDENCE, PostureScores

logger = get_logger(__name__)

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
        scores = await self.score(frame, detection, pose_result)
        best = max(scores.lying, scores.sitting, scores.standing_walking)
        if best < _MIN_EVIDENCE:
            return "unknown"
        if scores.lying >= scores.sitting and scores.lying >= scores.standing_walking:
            return "lying"
        if scores.sitting >= scores.standing_walking:
            return "sitting"
        return "standing"

    async def score(
        self,
        frame: npt.NDArray[np.uint8],
        detection: Detection,
        pose_result: object | None = None,
    ) -> PostureScores:
        """Return soft scores derived from the depth blob's principal axis angle.

        Angle from horizontal:
          >60° (vertical body) → standing_walking = 1.0
          <30° (horizontal body) → lying = 1.0
          30-60° → sitting score proportional to distance from each threshold
        Uses a confidence of 0.0 because depth provides no keypoint information.
        """
        try:
            t0 = time.monotonic()
            depth_map = await self._depth.estimate(frame)
            _metrics.metrics.cts_posture_slow_path_runs_total.labels(
                camera_id=detection.camera_id,
            ).inc()
            elapsed = time.monotonic() - t0
            _metrics.metrics.cts_posture_slow_path_latency_seconds.observe(elapsed)
            return self._score_from_depth(depth_map, detection)
        except Exception:  # noqa: BLE001
            logger.warning(
                "depth_posture_slow_path_failed",
                camera_id=detection.camera_id,
                exc_info=True,
            )
            return PostureScores(lying=0.0, sitting=0.0, standing_walking=0.0)

    def _score_from_depth(
        self,
        depth_map: npt.NDArray[np.float32],
        detection: Detection,
    ) -> PostureScores:
        bbox = detection.bbox
        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(depth_map.shape[1], int(bbox.x_max))
        y2 = min(depth_map.shape[0], int(bbox.y_max))

        if x2 <= x1 or y2 <= y1:
            return PostureScores(lying=0.0, sitting=0.0, standing_walking=0.0)

        crop = depth_map[y1:y2, x1:x2]
        if crop.std() < _MIN_DEPTH_VARIANCE:
            return PostureScores(lying=0.0, sitting=0.0, standing_walking=0.0)

        ys, xs = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
        weights = 1.0 / (crop + 1e-6)
        weights /= weights.sum()
        mean_y = float((ys * weights).sum())
        mean_x = float((xs * weights).sum())
        dy = ys - mean_y
        dx = xs - mean_x
        cov_yy = float((weights * dy * dy).sum())
        cov_xx = float((weights * dx * dx).sum())
        cov_yx = float((weights * dy * dx).sum())
        angle_rad = 0.5 * np.arctan2(2 * cov_yx, cov_xx - cov_yy)
        angle_deg = float(abs(np.degrees(angle_rad)))

        if angle_deg > _VERTICAL_DEG:
            # Strong vertical → standing
            standing = min(1.0, (angle_deg - _VERTICAL_DEG) / (90.0 - _VERTICAL_DEG) + 0.7)
            return PostureScores(
                lying=0.0, sitting=0.0, standing_walking=standing, keypoint_confidence=0.0
            )
        elif angle_deg < _HORIZONTAL_DEG:
            # Strong horizontal → lying
            lying = min(1.0, 1.0 - angle_deg / _HORIZONTAL_DEG * 0.3)
            return PostureScores(
                lying=lying, sitting=0.0, standing_walking=0.0, keypoint_confidence=0.0
            )
        else:
            # Intermediate → sitting (conservative)
            mid_range = _VERTICAL_DEG - _HORIZONTAL_DEG
            sitting = 0.55 + 0.2 * (angle_deg - _HORIZONTAL_DEG) / mid_range
            return PostureScores(
                lying=0.0,
                sitting=min(1.0, sitting),
                standing_walking=0.0,
                keypoint_confidence=0.0,
            )

    def _classify_from_depth(
        self,
        depth_map: npt.NDArray[np.float32],
        detection: Detection,
    ) -> PostureType:
        """Deprecated — use score() + resolve to label instead."""
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
