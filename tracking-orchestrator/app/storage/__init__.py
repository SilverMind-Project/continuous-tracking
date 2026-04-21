"""Storage package — repository protocols and implementations.

Exports:
- Base protocols: TrackingRepository, GalleryRepository, GlobalTrackRepository, etc.
- In-memory implementations for testing
- Postgres implementations for production
"""

from .base import (
    ActivityRepository,
    AssignmentRepository,
    CorrectionRepository,
    GalleryRepository,
    GlobalTrackRepository,
    InMemoryActivityRepository,
    InMemoryAssignmentRepository,
    InMemoryCorrectionRepository,
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
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
    "GlobalTrackRepository",
    "InMemoryActivityRepository",
    "InMemoryAssignmentRepository",
    "InMemoryCorrectionRepository",
    "InMemoryGalleryRepository",
    "InMemoryGlobalTrackRepository",
    "InMemoryPrivacyRepository",
    "InMemorySettingsRepository",
    "InMemoryTrackingRepository",
    "PrivacyRepository",
    "SettingsRepository",
    "TrackingRepository",
]
