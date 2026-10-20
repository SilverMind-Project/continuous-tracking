"""face_crop_box: padded upward and sideways, never downward, clamped to the frame.

Privacy is inherited rather than clipped: PrivacyStage blurs or masks privacy zones in the frame
image itself, so a padded crop of that image cannot expose them. The last test proves that by
running the real blur and cropping a padded box that overlaps a mask zone.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.calibration.state import PrivacyZoneConfig
from app.domain import BoundingBox
from app.pipeline.crops import FaceCropPadding, crop_detection_from_bbox, face_crop_box
from app.pipeline.privacy import PrivacyZoneFilter

_PADDING = FaceCropPadding(top_frac=0.15, side_frac=0.10)


def test_pads_top_and_sides_by_fraction_of_box_size() -> None:
    box = BoundingBox(x_min=200, y_min=300, x_max=300, y_max=700)  # 100 x 400
    assert face_crop_box(box, 1920, 1080, _PADDING) == BoundingBox(
        x_min=190, y_min=240, x_max=310, y_max=700
    )


def test_bottom_is_never_padded() -> None:
    box = BoundingBox(x_min=200, y_min=300, x_max=300, y_max=700)
    assert face_crop_box(box, 1920, 1080, _PADDING).y_max == box.y_max


def test_box_at_frame_edges_is_clamped() -> None:
    box = BoundingBox(x_min=0, y_min=0, x_max=80, y_max=400)
    padded = face_crop_box(box, 100, 1080, _PADDING)
    assert (padded.x_min, padded.y_min) == (0, 0)
    assert padded.x_max == 88

    right = BoundingBox(x_min=60, y_min=10, x_max=100, y_max=200)
    assert face_crop_box(right, 100, 1080, _PADDING).x_max == 100


def test_zero_padding_returns_the_person_box() -> None:
    box = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220)
    assert face_crop_box(box, 640, 480, FaceCropPadding(top_frac=0.0, side_frac=0.0)) == box


def test_negative_padding_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FaceCropPadding(top_frac=-0.1, side_frac=0.1)


def test_padded_crop_inherits_privacy_mask() -> None:
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Mask the band above the person box; the padded crop reaches into it.
    zone = PrivacyZoneConfig(
        zone_id="z1",
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 0.25], [0.0, 0.25]],
        policy="mask_region",
    )
    privacy = PrivacyZoneFilter(
        "cam-1", [zone], homography_available=False, frame_width=200, frame_height=200
    )
    masked = privacy.apply_blur_mask(frame.copy())

    person = BoundingBox(x_min=80, y_min=60, x_max=120, y_max=190)
    padded = face_crop_box(person, 200, 200, _PADDING)
    assert padded.y_min < 50  # the padding overlaps the masked band

    crop = crop_detection_from_bbox(masked, padded)
    masked_rows = 50 - padded.y_min
    assert np.all(crop[:masked_rows] == 114)
    assert np.all(crop[masked_rows + 1 :] == 255)
