"""Crop extraction utilities for person detections.

Every crop used for ReID, pose, face ID, keyframes, and scene samples
must come from privacy-filtered imagery.  Detections dropped by the
privacy stage must never reach any model adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class FaceCropPadding:
    """How far the face crop extends beyond the person box, as fractions of the box size.

    SCRFD is trained on faces with surrounding context; a tight person box often cuts the head at
    its top edge. The bottom is never padded because a face is never below the person box.
    """

    top_frac: float = 0.15
    side_frac: float = 0.10

    def __post_init__(self) -> None:
        if self.top_frac < 0.0 or self.side_frac < 0.0:
            raise ValueError("face crop padding fractions must be non-negative")


def face_crop_box(
    person_box_px: BoundingBox,
    frame_width_px: int,
    frame_height_px: int,
    padding: FaceCropPadding,
) -> BoundingBox:
    """Expand a person box upward and sideways for face detection, clamped to the frame.

    Privacy zones need no handling here: ``PrivacyStage`` blurs or masks them in the frame image
    itself, so any crop of that image, padded or not, inherits the mask.
    """
    width_px = person_box_px.x_max - person_box_px.x_min
    height_px = person_box_px.y_max - person_box_px.y_min
    side_px = round(padding.side_frac * width_px)
    top_px = round(padding.top_frac * height_px)
    return BoundingBox(
        x_min=max(0, person_box_px.x_min - side_px),
        y_min=max(0, person_box_px.y_min - top_px),
        x_max=min(frame_width_px, person_box_px.x_max + side_px),
        y_max=min(frame_height_px, person_box_px.y_max),
    )


def is_degenerate(crop: npt.NDArray[np.uint8]) -> bool:
    """Return True when the crop is too small for meaningful model inference."""
    h, w = crop.shape[:2]
    return bool(w < _MIN_CROP_WIDTH or h < _MIN_CROP_HEIGHT)
