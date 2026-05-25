"""Pipeline stage implementations."""

from .base import FrameStage, StageRunner
from .detect import DetectStage
from .detection_backfill import DetectionBackfillStage
from .face_identity import FaceIdentityStage
from .fetch import FetchStage
from .global_tracking import GlobalTrackingStage
from .inference import InferenceStage
from .keyframes import KeyframeStage
from .local_tracking import LocalTrackingStage
from .posture_stage import PostureStage
from .posture_trails import PostureAndTrailsStage, TrailsStage
from .privacy import PrivacyStage
from .publish import PublishStage
from .revisions import RevisionsStage
from .spatial_projection import SpatialProjectionStage
from .trajectory import CloseTerminatedStage, TrajectoryStage

__all__ = [
    "CloseTerminatedStage",
    "DetectStage",
    "DetectionBackfillStage",
    "FaceIdentityStage",
    "FetchStage",
    "FrameStage",
    "GlobalTrackingStage",
    "InferenceStage",
    "KeyframeStage",
    "LocalTrackingStage",
    "PostureAndTrailsStage",
    "PostureStage",
    "PrivacyStage",
    "PublishStage",
    "RevisionsStage",
    "SpatialProjectionStage",
    "StageRunner",
    "TrailsStage",
    "TrajectoryStage",
]
