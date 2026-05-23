"""Tests for model evidence types, crop quality, and evidence-based degradation."""

from __future__ import annotations

import numpy as np
import pytest

from app.domain import BoundingBox, FloorPoint
from app.inference.evidence import (
    AppearanceEvidence,
    FaceEvidence,
    PersonDetectionEvidence,
    PoseEvidence,
)
from app.inference.schemas import DetectionBox
from app.pipeline.crop_quality import CropQuality
from app.pipeline.crops import crop_detection, is_degenerate


class TestCropDetection:
    """Crop extraction from normalised DetectionBox."""

    def test_crop_from_normalised_box(self) -> None:
        """Normalised [0, 1] box produces correct pixel crop."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[100:200, 150:300] = 255  # bright patch
        det = DetectionBox(x1=0.25, y1=0.2, x2=0.5, y2=0.4, confidence=0.9)
        crop = crop_detection(image, det)
        assert crop.shape == (96, 160, 3), f"expected (96, 160, 3), got {crop.shape}"

    def test_crop_clamped_to_image_bounds(self) -> None:
        """Out-of-bounds normalised coords are clamped."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        det = DetectionBox(x1=-0.5, y1=-0.5, x2=1.5, y2=1.5, confidence=0.5)
        crop = crop_detection(image, det)
        assert crop.shape[0] <= 100
        assert crop.shape[1] <= 100

    def test_degenerate_crop_detected(self) -> None:
        """Tiny crops are flagged as degenerate."""
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        assert is_degenerate(tiny)

    def test_valid_crop_not_degenerate(self) -> None:
        """Normal-sized crops are not degenerate."""
        valid = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not is_degenerate(valid)


class TestCropQuality:
    """CropQuality scores for gallery/ReID/keyframe eligibility."""

    def test_high_quality_detection(self) -> None:
        """Large, confident, non-truncated detection with visible keypoints."""
        cq = CropQuality(
            area_fraction=0.05,
            detector_confidence=0.95,
            edge_truncated=False,
            crop_width_px=200,
            crop_height_px=400,
            visible_keypoint_count=12,
        )
        # Quality formula: 0.3*area + 0.3*conf + 0.2*kp - truncation
        # With area_fraction=0.05 (area_score=0.5), conf=0.95, kp=1.0:
        #   0.3*0.5 + 0.3*0.95 + 0.2*1.0 = 0.15 + 0.285 + 0.20 = 0.635
        assert cq.quality > 0.6, f"expected > 0.6, got {cq.quality}"

    def test_edge_truncated_penalty(self) -> None:
        """Edge truncation reduces quality score."""
        cq_good = CropQuality(
            area_fraction=0.05,
            detector_confidence=0.9,
            edge_truncated=False,
            crop_width_px=200,
            crop_height_px=400,
        )
        cq_bad = CropQuality(
            area_fraction=0.05,
            detector_confidence=0.9,
            edge_truncated=True,
            crop_width_px=200,
            crop_height_px=400,
        )
        assert cq_bad.quality < cq_good.quality

    def test_tiny_crop_low_quality(self) -> None:
        """Very small crops get low quality scores."""
        cq = CropQuality(
            area_fraction=0.001,
            detector_confidence=0.5,
            edge_truncated=False,
            crop_width_px=20,
            crop_height_px=40,
        )
        assert cq.quality < 0.5

    def test_quality_bounded_0_to_1(self) -> None:
        """Quality score is always in [0, 1]."""
        cq = CropQuality(
            area_fraction=1.0,
            detector_confidence=1.0,
            edge_truncated=False,
            crop_width_px=1000,
            crop_height_px=1000,
            visible_keypoint_count=17,
        )
        assert 0.0 <= cq.quality <= 1.0


class TestEvidenceTypes:
    """Evidence dataclasses are frozen and carry correct metadata."""

    def test_person_detection_evidence_is_frozen(self) -> None:
        """PersonDetectionEvidence is immutable."""
        from datetime import UTC, datetime

        ev = PersonDetectionEvidence(
            detection_id="det-1",
            camera_id="cam-1",
            frame_index=0,
            bbox=BoundingBox(10, 20, 100, 200),
            confidence=0.95,
            floor_point=FloorPoint(0, 0),
            captured_at=datetime.now(UTC),
        )
        with pytest.raises(Exception):  # noqa: B017
            ev.confidence = 0.5  # type: ignore[misc]

    def test_appearance_evidence_does_not_contain_raw_bytes(self) -> None:
        """AppearanceEvidence.embedding is a tuple of floats, not raw bytes."""
        from datetime import UTC, datetime

        ev = AppearanceEvidence(
            detection_id="det-1",
            camera_id="cam-1",
            frame_index=0,
            embedding=tuple(float(i) for i in range(768)),
            crop_quality=0.8,
            captured_at=datetime.now(UTC),
        )
        assert isinstance(ev.embedding, tuple)
        assert all(isinstance(v, float) for v in ev.embedding[:10])
        # Evidence must not contain raw image data.
        assert not hasattr(ev, "crop_image")
        assert not hasattr(ev, "image_bytes")

    def test_pose_evidence_has_17_keypoints(self) -> None:
        """PoseEvidence carries exactly 17 keypoints."""
        from datetime import UTC, datetime

        kps = tuple((0.5, 0.5, 0.9) for _ in range(17))
        ev = PoseEvidence(
            detection_id="det-1",
            camera_id="cam-1",
            frame_index=0,
            keypoints=kps,
            visible_keypoint_count=15,
            quality=0.85,
            captured_at=datetime.now(UTC),
        )
        assert len(ev.keypoints) == 17
        assert ev.visible_keypoint_count == 15

    def test_face_evidence_sources_are_distinguishable(self) -> None:
        """Direct and propagated FaceEvidence are different objects."""
        from datetime import UTC, datetime

        direct = FaceEvidence(
            person_id="alice",
            confidence=0.95,
            tracklet_id="tl-1",
            camera_id="cam-1",
            frame_index=42,
            source="direct",
            captured_at=datetime.now(UTC),
        )
        propagated = FaceEvidence(
            person_id="alice",
            confidence=0.95,
            tracklet_id="tl-2",
            camera_id="cam-2",
            frame_index=0,
            source="propagated",
            captured_at=datetime.now(UTC),
        )
        assert direct.source == "direct"
        assert propagated.source == "propagated"
        assert direct != propagated  # different tracklet_id

    def test_face_evidence_no_raw_image(self) -> None:
        """FaceEvidence must not contain raw image bytes."""
        from datetime import UTC, datetime

        ev = FaceEvidence(
            person_id="alice",
            confidence=0.95,
            tracklet_id="tl-1",
            camera_id="cam-1",
            frame_index=0,
            source="direct",
            captured_at=datetime.now(UTC),
        )
        assert not hasattr(ev, "face_crop")
        assert not hasattr(ev, "image_data")
        assert not hasattr(ev, "jpeg_bytes")
