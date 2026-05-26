"""Pipeline stage implementations."""

from .base import FrameStage, StageRunner
from .detect import DetectStage
from .detection_backfill import DetectionBackfillStage
from .face_identity import FaceIdentityStage
from .fetch import FetchStage
from .inference import InferenceStage
from .keyframes import KeyframeStage
from .posture_stage import PostureStage
from .posture_trails import PostureAndTrailsStage, TrailsStage
from .privacy import PrivacyStage
from .publish import PublishStage
from .revisions import RevisionsStage
from .spatial_projection import SpatialProjectionStage
from .trajectory import CloseTerminatedStage, TrajectoryStage
from .world_tracking import WorldTrackingStage

__all__ = [
    "CloseTerminatedStage",
    "DetectStage",
    "DetectionBackfillStage",
    "FaceIdentityStage",
    "FetchStage",
    "FrameStage",
    "InferenceStage",
    "KeyframeStage",
    "PostureAndTrailsStage",
    "PostureStage",
    "PrivacyStage",
    "PublishStage",
    "RevisionsStage",
    "SpatialProjectionStage",
    "StageRunner",
    "TrailsStage",
    "TrajectoryStage",
    "WorldTrackingStage",
]
