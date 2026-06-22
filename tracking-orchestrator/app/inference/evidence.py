"""Typed model evidence produced by inference adapters.

Each evidence type is a frozen dataclass carrying provenance metadata
(model name, version, quality) so downstream consumers can make decisions
without reaching back into raw model outputs or image data.

Evidence objects never contain raw images, embedding vectors in log
messages, or unredacted face data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from ..domain import BoundingBox, FloorPoint

# ---------------------------------------------------------------------------
# Person detection evidence (YOLO26L)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonDetectionEvidence:
    """One person detection from the YOLO person detector."""

    detection_id: str
    camera_id: str
    frame_index: int
    bbox: BoundingBox
    confidence: float
    floor_point: FloorPoint
    model_name: str = "yolo-v26l"
    model_version: str = ""
    preprocessing_width: int = 640
    preprocessing_height: int = 640
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Appearance evidence (SOLIDER-REID)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppearanceEvidence:
    """Appearance embedding from SOLIDER-REID for one person crop."""

    detection_id: str
    camera_id: str
    frame_index: int
    embedding: tuple[float, ...]
    crop_quality: float
    model_name: str = "solider-reid"
    model_version: str = ""
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Pose evidence (RTMPose)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseEvidence:
    """Pose keypoints from RTMPose for one person crop."""

    detection_id: str
    camera_id: str
    frame_index: int
    keypoints: tuple[tuple[float, float, float], ...]  # 17 x (x, y, score)
    visible_keypoint_count: int
    quality: float  # mean visible keypoint score
    model_name: str = "rtmpose-m"
    model_version: str = ""
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Face evidence (person-identification-service / ArcFace)
# ---------------------------------------------------------------------------

FaceEvidenceSource = Literal["direct", "propagated", "manual"]


@dataclass(frozen=True)
class FaceEvidence:
    """Face-based identity evidence from the person-identification-service.

    ``source`` distinguishes:
    - ``"direct"``  — ArcFace match from a real person crop sent to the service.
    - ``"propagated"`` — synthetic anchor created by cross-GT face propagation.
    - ``"manual"`` — operator-applied identity correction.

    ``recognition_state`` is a separate axis from ``source``:
    - ``"recognized"`` — similarity >= threshold (strong positive).
    - ``"candidate"`` — grey zone between thresholds (weak positive).
    - ``"unrecognized"`` — face present but similarity < unknown_threshold.
    """

    person_id: str
    confidence: float
    tracklet_id: str = ""
    detection_id: str = ""
    camera_id: str = ""
    frame_index: int = 0
    source: FaceEvidenceSource = "direct"
    quality: float = 1.0
    model_name: str = "arcface"
    model_version: str = ""
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Three-valued recognition state.
    recognition_state: str = "recognized"
    # Raw cosine similarity to best candidate.
    similarity: float = 0.0
    # Head pose yaw in degrees.
    yaw_deg: float = 0.0
    # Calibrated ArcFace probability from the person-id service (None when degraded).
    calibrated_confidence: float | None = None
