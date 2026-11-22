"""Pipeline stage implementations."""

from __future__ import annotations

from .appearance_samples import AppearanceSampleRecorderStage
from .base import FrameStage, StageRunner, timed_phase
from .detect import DetectStage
from .detection_backfill import DetectionBackfillStage
from .face_identity import FaceIdentityStage
from .fall_detection import FallDetectionConfig, FallDetectionStage
from .fetch import FetchStage
from .height_estimation import HeightEstimationConfig, HeightEstimationStage
from .image_only_evaluation import ImageOnlyEvaluationStage
from .inference import InferenceStage
from .keyframes import KeyframeStage
from .posture_stage import PostureStage
from .posture_trails import TrailsStage
from .privacy import PrivacyStage
from .provenance import ProvenancePersistStage
from .publish import PublishStage
from .reid_candidates import ReIDCandidateStage
from .revisions import RevisionsStage
from .spatial_projection import SpatialProjectionStage
from .trajectory import ClosePHStage, TrajectoryStage
from .wearable_associations import WearableAssociationPersistStage
from .wearable_face_attestation import WearableFaceAttestationStage
from .world_tracking import WorldTrackingStage

__all__ = [
    "AppearanceSampleRecorderStage",
    "ClosePHStage",
    "DetectStage",
    "DetectionBackfillStage",
    "FaceIdentityStage",
    "FallDetectionConfig",
    "FallDetectionStage",
    "FetchStage",
    "FrameStage",
    "HeightEstimationConfig",
    "HeightEstimationStage",
    "ImageOnlyEvaluationStage",
    "InferenceStage",
    "KeyframeStage",
    "PostureStage",
    "PrivacyStage",
    "ProvenancePersistStage",
    "PublishStage",
    "ReIDCandidateStage",
    "RevisionsStage",
    "SpatialProjectionStage",
    "StageRunner",
    "TrailsStage",
    "TrajectoryStage",
    "WearableAssociationPersistStage",
    "WearableFaceAttestationStage",
    "WorldTrackingStage",
    "timed_phase",
]
