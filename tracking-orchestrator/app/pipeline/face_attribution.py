"""Attribute at most one face to a person detection.

A person crop can contain more than one face when people overlap: person A's crop may also hold
person B's face. A YOLO person box spans head to feet, so for standing, walking, and seated
people the face centre lies in the top part of the box and within its horizontal extent plus a
small margin. A face outside that head region belongs to someone else. When two faces fall
inside it, the crop's identity is ambiguous and no face is attributed: Unknown is preferable to a
confident swap.

All coordinates are frame pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..domain import BoundingBox


@dataclass(frozen=True)
class HeadRegionPolicy:
    """Where a person's face may appear relative to their detection box."""

    # Fraction of the person-box height, measured down from the box top, that can hold the face.
    head_height_frac: float = 0.35
    # Horizontal tolerance on each side, as a fraction of the person-box width.
    side_margin_frac: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.head_height_frac <= 1.0:
            raise ValueError("head_height_frac must be in (0, 1]")
        if self.side_margin_frac < 0.0:
            raise ValueError("side_margin_frac must be non-negative")


class FaceAttribution(StrEnum):
    ASSIGNED = "assigned"
    NO_FACE_IN_HEAD_REGION = "no_face_in_head_region"
    AMBIGUOUS = "ambiguous"


def assign_face_to_person(
    person_box_px: BoundingBox,
    face_boxes_px: Sequence[tuple[float, float, float, float]],
    policy: HeadRegionPolicy,
    pad_top_px: float = 0.0,
) -> tuple[FaceAttribution, int | None]:
    """Return the index of the single face whose centre lies in the person's head region.

    ``face_boxes_px`` are ``(x1, y1, x2, y2)`` in frame pixels. ``pad_top_px`` is how far above
    the person box the face crop extended: a head above the box top is still this person's head.
    """
    width_px = person_box_px.x_max - person_box_px.x_min
    height_px = person_box_px.y_max - person_box_px.y_min
    margin_px = policy.side_margin_frac * width_px
    left_px = person_box_px.x_min - margin_px
    right_px = person_box_px.x_max + margin_px
    top_px = person_box_px.y_min - max(0.0, pad_top_px)
    bottom_px = person_box_px.y_min + policy.head_height_frac * height_px

    inside = [
        index
        for index, (x1, y1, x2, y2) in enumerate(face_boxes_px)
        if left_px <= (x1 + x2) / 2.0 <= right_px and top_px <= (y1 + y2) / 2.0 <= bottom_px
    ]
    if not inside:
        return FaceAttribution.NO_FACE_IN_HEAD_REGION, None
    if len(inside) > 1:
        return FaceAttribution.AMBIGUOUS, None
    return FaceAttribution.ASSIGNED, inside[0]
