"""Storage protocols — re-exported for backward compatibility.

Import directly from the sub-module for new code:
    from app.storage.tracking import TrackingRepository, InMemoryTrackingRepository
"""

from __future__ import annotations

from .annotations import BboxAnnotationRepository, InMemoryBboxAnnotationRepository
from .gallery import GalleryRepository, InMemoryGalleryRepository
from .global_track import GlobalTrackRepository, InMemoryGlobalTrackRepository
from .hints import DoNotFuseRepository, InMemoryDoNotFuseRepository
from .misc import (
    ActivityRepository,
    AssignmentRepository,
    CorrectionRepository,
    InMemoryActivityRepository,
    InMemoryAssignmentRepository,
    InMemoryCorrectionRepository,
    InMemoryPrivacyRepository,
    InMemorySettingsRepository,
    PrivacyRepository,
    SettingsRepository,
)
from .signals import (
    BehaviorBaselineRepository,
    DementiaSignalRepository,
    HourlyActivitySummary,
    InMemoryBehaviorBaselineRepository,
    InMemoryDementiaSignalRepository,
    StillnessEpisode,
)
from .tracking import InMemoryTrackingRepository, TrackingRepository
from .trajectory import (
    InMemoryKeyframeRepository,
    InMemoryTrajectoryRepository,
    KeyframeRepository,
    TrajectoryRepository,
)

__all__ = [
    "ActivityRepository",
    "AssignmentRepository",
    "BboxAnnotationRepository",
    "BehaviorBaselineRepository",
    "CorrectionRepository",
    "DementiaSignalRepository",
    "DoNotFuseRepository",
    "GalleryRepository",
    "GlobalTrackRepository",
    "HourlyActivitySummary",
    "InMemoryActivityRepository",
    "InMemoryAssignmentRepository",
    "InMemoryBboxAnnotationRepository",
    "InMemoryBehaviorBaselineRepository",
    "InMemoryCorrectionRepository",
    "InMemoryDementiaSignalRepository",
    "InMemoryDoNotFuseRepository",
    "InMemoryGalleryRepository",
    "InMemoryGlobalTrackRepository",
    "InMemoryKeyframeRepository",
    "InMemoryPrivacyRepository",
    "InMemorySettingsRepository",
    "InMemoryTrackingRepository",
    "InMemoryTrajectoryRepository",
    "KeyframeRepository",
    "PrivacyRepository",
    "SettingsRepository",
    "StillnessEpisode",
    "TrackingRepository",
    "TrajectoryRepository",
]
