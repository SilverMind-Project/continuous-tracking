"""Pipeline stage implementations."""

from .base import FrameStage, StageRunner
from .detect import DetectStage
from .detection_backfill import DetectionBackfillStage
from .face_identity import FaceIdentityStage
from .fall_detection import FallDetectionConfig, FallDetectionStage
from .fetch import FetchStage
from .inference import InferenceStage
from .keyframes import KeyframeStage
from .posture_stage import PostureStage
from .posture_trails import TrailsStage
from .privacy import PrivacyStage
from .provenance import ProvenancePersistStage
from .publish import PublishStage
from .revisions import RevisionsStage
from .spatial_projection import SpatialProjectionStage
from .trajectory import ClosePHStage, TrajectoryStage
from .world_tracking import WorldTrackingStage

__all__ = [
    "ClosePHStage",
    "DetectStage",
    "DetectionBackfillStage",
    "FaceIdentityStage",
    "FallDetectionConfig",
    "FallDetectionStage",
    "FetchStage",
    "FrameStage",
    "InferenceStage",
    "KeyframeStage",
    "PostureStage",
    "PrivacyStage",
    "ProvenancePersistStage",
    "PublishStage",
    "RevisionsStage",
    "SpatialProjectionStage",
    "StageRunner",
    "TrailsStage",
    "TrajectoryStage",
    "WorldTrackingStage",
]
