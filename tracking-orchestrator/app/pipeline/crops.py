"""Crop extraction utilities for person detections.

Every crop used for ReID, pose, face ID, keyframes, and scene samples
must come from privacy-filtered imagery.  Detections dropped by the
privacy stage must never reach any model adapter.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..domain import BoundingBox
from ..inference.schemas import DetectionBox

# Minimum crop dimensions before a crop is considered degenerate.
# ReID requires 384x128; RTMPose requires 256x192.  Below these thresholds
# the model will produce meaningless outputs, so crops are skipped.
_MIN_CROP_WIDTH = 16
_MIN_CROP_HEIGHT = 32


def crop_detection(
    image: npt.NDArray[np.uint8],
    det: DetectionBox,
) -> npt.NDArray[np.uint8]:
    """Crop one normalised detector box from an RGB image.

    Returns a contiguous uint8 crop of shape (h, w, 3).  The crop is
    taken from the original image — the caller must ensure *image* has
    already been privacy-filtered.
    """
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(det.x1 * w)))
    y1 = max(0, min(h - 1, int(det.y1 * h)))
    x2 = max(x1 + 1, min(w, int(det.x2 * w)))
    y2 = max(y1 + 1, min(h, int(det.y2 * h)))
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def crop_detection_from_bbox(
    image: npt.NDArray[np.uint8],
    bbox: BoundingBox,
) -> npt.NDArray[np.uint8]:
    """Crop a pixel-space BoundingBox from an RGB image."""
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, bbox.x_min))
    y1 = max(0, min(h - 1, bbox.y_min))
    x2 = max(x1 + 1, min(w, bbox.x_max))
    y2 = max(y1 + 1, min(h, bbox.y_max))
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def is_degenerate(crop: npt.NDArray[np.uint8]) -> bool:
    """Return True when the crop is too small for meaningful model inference."""
    h, w = crop.shape[:2]
    return bool(w < _MIN_CROP_WIDTH or h < _MIN_CROP_HEIGHT)
