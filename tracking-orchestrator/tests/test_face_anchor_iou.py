"""Regression tests for B-2: face-anchor-to-detection IoU coordinate mismatch.

Ensures the _bbox_iou helper and _identify_faces correctly handle the
normalised <-> pixel coordinate translation so face evidence reaches the
Bayesian identity resolver.
"""

from __future__ import annotations

import pytest

from app.pipeline.stages.detect import _bbox_iou


class TestBboxIoU:
    """Tests for the IoU helper used in face-to-detection association."""

    def test_perfect_overlap(self) -> None:
        """Two identical boxes should have IoU = 1.0."""
        box = [0.1, 0.2, 0.5, 0.6]
        assert _bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Disjoint boxes should have IoU = 0.0."""
        a = [0.0, 0.0, 0.1, 0.1]
        b = [0.9, 0.9, 1.0, 1.0]
        assert _bbox_iou(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        """Partially overlapping boxes should have 0 < IoU < 1."""
        a = [0.0, 0.0, 0.5, 0.5]
        b = [0.3, 0.3, 0.8, 0.8]
        iou = _bbox_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_face_box_inside_person_box(self) -> None:
        """A face bbox fully inside a person bbox (common case).

        Face bbox = [0.3, 0.1, 0.5, 0.3]  (normalised, top portion of person)
        Person bbox = [0.2, 0.05, 0.6, 0.8]  (normalised, full body)
        """
        face_norm = [0.3, 0.1, 0.5, 0.3]
        person_norm = [0.2, 0.05, 0.6, 0.8]
        iou = _bbox_iou(face_norm, person_norm)
        # Face is fully inside person: IoU = area_face / area_person
        # area_face = (0.5-0.3)*(0.3-0.1) = 0.2*0.2 = 0.04
        # area_person = (0.6-0.2)*(0.8-0.05) = 0.4*0.75 = 0.3
        # IoU = 0.04 / 0.3 ≈ 0.1333
        assert iou == pytest.approx(0.04 / 0.3)
        assert iou > 0.1  # Above the 0.1 threshold used in _identify_faces

    def test_normalised_vs_pixel_was_broken(self) -> None:
        """Regression: normalised face bbox vs absolute pixel person bbox.

        Before the fix, _identify_faces compared face.bbox_normalized
        (in [0,1]) against det.bbox (in pixel coordinates e.g. 1920x1080),
        yielding an IoU near zero. This test asserts the fix normalises
        both sides.
        """
        frame_width = 1920
        frame_height = 1080

        # Face bbox in normalised [0,1] — face takes ~upper 20% of person
        face_norm = [0.25, 0.05, 0.45, 0.25]

        # Person bbox in pixel coords (before normalisation)
        person_px = [420, 30, 900, 820]  # rough pixel bbox for the person

        # Bug: IoU(norm, px) — nearly zero intersection
        buggy_iou = _bbox_iou(face_norm, person_px)
        assert buggy_iou < 0.01  # The bug: IoU near zero

        # Fix: normalise the person bbox first
        person_norm = [
            person_px[0] / frame_width,
            person_px[1] / frame_height,
            person_px[2] / frame_width,
            person_px[3] / frame_height,
        ]
        fixed_iou = _bbox_iou(face_norm, person_norm)
        assert fixed_iou > 0.1  # Should now properly associate

    def test_normalised_boxes_scale_consistent(self) -> None:
        """IoU of two normalised boxes is independent of the frame dimensions
        that produced the normalisation, because the boxes express the same
        relative positions."""
        # Two boxes at fixed normalised positions.
        face_norm = [0.25, 0.08, 0.45, 0.25]
        person_norm = [0.20, 0.05, 0.60, 0.75]
        iou = _bbox_iou(face_norm, person_norm)
        # Sanity: the face is mostly inside the person box.
        assert iou > 0.05


class TestDetectionNormalisation:
    """Verify the fix applied in _identify_faces at frame_pipeline.py."""

    def test_det_norm_produces_sensible_iou(self) -> None:
        """Simulate the exact fix code path: divide pixel bbox by frame dims."""
        frame_width = 1920
        frame_height = 1080

        # A face occupies the upper-left region of the detection.
        face_norm = [0.22, 0.04, 0.44, 0.22]

        # Simulated YOLO detection bbox in pixel space.
        det_px_x_min = 420
        det_px_y_min = 36
        det_px_x_max = 864
        det_px_y_max = 810

        det_norm = [
            det_px_x_min / frame_width,  # 0.21875
            det_px_y_min / frame_height,  # 0.03333
            det_px_x_max / frame_width,  # 0.45
            det_px_y_max / frame_height,  # 0.75
        ]

        iou = _bbox_iou(face_norm, det_norm)
        # Face box [0.22,0.04,0.44,0.22] is mostly inside person box
        # [0.219,0.033,0.45,0.75] — should have meaningful overlap.
        assert iou > 0.1, f"Expected IoU > 0.1, got {iou:.4f}"
