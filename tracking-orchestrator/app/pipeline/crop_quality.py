"""Crop quality scoring for person detection crops.

CropQuality is computed once per detection and used consistently for
gallery append, ReID likelihood, keyframe sampling, and face ID eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CropQuality:
    """Quality assessment of a person detection crop.

    All fields are in the range [0, 1] unless noted.
    """

    # Fraction of the full frame occupied by the bbox (0..1).
    area_fraction: float

    # Detection confidence from the person detector (0..1).
    detector_confidence: float

    # Whether the bbox touches any image edge (potentially truncated person).
    edge_truncated: bool

    # Crop pixel dimensions.
    crop_width_px: int
    crop_height_px: int

    # Number of visible keypoints (0..17) when pose is available, else 0.
    visible_keypoint_count: int = 0

    @property
    def quality(self) -> float:
        """Composite quality score in [0, 1].

        Higher = more reliable crop for ReID, gallery append, keyframe
        sampling, and face ID eligibility.
        """
        # Area: larger crops are more informative (cap at 0.1 = 10% of frame).
        area_score = min(self.area_fraction / 0.1, 1.0)

        # Edge truncation penalty.
        truncation_penalty = 0.4 if self.edge_truncated else 0.0

        # Confidence directly from the detector.
        conf_score = self.detector_confidence

        # Visible keypoints boost (0..17 → 0..1).
        kp_score = min(self.visible_keypoint_count / 10.0, 1.0)

        raw = 0.30 * area_score + 0.30 * conf_score + 0.20 * kp_score - truncation_penalty
        return max(0.0, min(raw, 1.0))
