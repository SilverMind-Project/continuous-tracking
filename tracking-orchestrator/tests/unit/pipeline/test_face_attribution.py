"""assign_face_to_person: at most one face per person box, chosen by the head-region rule."""

from __future__ import annotations

import pytest

from app.domain import BoundingBox
from app.pipeline.face_attribution import (
    FaceAttribution,
    HeadRegionPolicy,
    assign_face_to_person,
)

_POLICY = HeadRegionPolicy(head_height_frac=0.35, side_margin_frac=0.10)
# Standing person: 100 px wide, 300 px tall. Head region: x in [90, 210], y in [100, 205].
_STANDING = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=400)


def _face(cx: float, cy: float, half: float = 10.0) -> tuple[float, float, float, float]:
    return (cx - half, cy - half, cx + half, cy + half)


@pytest.mark.parametrize(
    ("faces", "pad_top_px", "expected"),
    [
        ([_face(150, 140)], 0.0, (FaceAttribution.ASSIGNED, 0)),
        # Another person's face to the right, beyond the 10 px side margin, is ignored.
        ([_face(150, 140), _face(240, 140)], 0.0, (FaceAttribution.ASSIGNED, 0)),
        ([_face(240, 140), _face(150, 140)], 0.0, (FaceAttribution.ASSIGNED, 1)),
        # Two faces inside the head region: the crop's identity is ambiguous.
        ([_face(130, 140), _face(175, 150)], 0.0, (FaceAttribution.AMBIGUOUS, None)),
        # Head above the box top but inside the top padding still belongs to this person.
        ([_face(150, 80)], 45.0, (FaceAttribution.ASSIGNED, 0)),
        # Same face without padding is outside the region.
        ([_face(150, 80)], 0.0, (FaceAttribution.NO_FACE_IN_HEAD_REGION, None)),
        # Chest height (below the top 35%).
        ([_face(150, 260)], 0.0, (FaceAttribution.NO_FACE_IN_HEAD_REGION, None)),
        # Within the side margin counts as inside.
        ([_face(205, 140)], 0.0, (FaceAttribution.ASSIGNED, 0)),
        ([], 0.0, (FaceAttribution.NO_FACE_IN_HEAD_REGION, None)),
    ],
    ids=[
        "single_face",
        "other_face_beyond_margin",
        "assigned_index_is_position",
        "two_faces_ambiguous",
        "head_above_box_within_padding",
        "head_above_box_without_padding",
        "chest_height",
        "inside_side_margin",
        "no_faces",
    ],
)
def test_head_region_rule(
    faces: list[tuple[float, float, float, float]],
    pad_top_px: float,
    expected: tuple[FaceAttribution, int | None],
) -> None:
    assert assign_face_to_person(_STANDING, faces, _POLICY, pad_top_px=pad_top_px) == expected


def test_seated_person_box_selects_the_face() -> None:
    # Seated: box nearly square (160 x 170); head region is y in [300, 359.5].
    seated = BoundingBox(x_min=400, y_min=300, x_max=560, y_max=470)
    faces = [_face(470, 330), _face(480, 440)]  # face, and a hand-held photo at lap height
    assert assign_face_to_person(seated, faces, _POLICY) == (FaceAttribution.ASSIGNED, 0)


def test_negative_padding_is_treated_as_none() -> None:
    assert assign_face_to_person(_STANDING, [_face(150, 80)], _POLICY, pad_top_px=-30.0) == (
        FaceAttribution.NO_FACE_IN_HEAD_REGION,
        None,
    )


@pytest.mark.parametrize(
    ("height_frac", "margin_frac"),
    [(0.0, 0.1), (1.2, 0.1), (0.35, -0.1)],
)
def test_policy_rejects_invalid_fractions(height_frac: float, margin_frac: float) -> None:
    with pytest.raises(ValueError, match="frac"):
        HeadRegionPolicy(head_height_frac=height_frac, side_margin_frac=margin_frac)
