"""Storage package — repository protocols and implementations."""

from .base import (
    ActivityRepository,
    AssignmentRepository,
    CorrectionRepository,
    GalleryRepository,
    InMemoryActivityRepository,
    InMemoryAssignmentRepository,
    InMemoryCorrectionRepository,
    InMemoryGalleryRepository,
    InMemoryPrivacyRepository,
    InMemorySettingsRepository,
    InMemoryTrackingRepository,
    PrivacyRepository,
    SettingsRepository,
    TrackingRepository,
)

__all__ = [
    "ActivityRepository",
    "AssignmentRepository",
    "CorrectionRepository",
    "GalleryRepository",
    "InMemoryActivityRepository",
    "InMemoryAssignmentRepository",
    "InMemoryCorrectionRepository",
    "InMemoryGalleryRepository",
    "InMemoryPrivacyRepository",
    "InMemorySettingsRepository",
    "InMemoryTrackingRepository",
    "PrivacyRepository",
    "SettingsRepository",
    "TrackingRepository",
]
