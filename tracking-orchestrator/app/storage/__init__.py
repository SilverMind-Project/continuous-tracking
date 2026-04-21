"""Storage package — repository protocols and implementations.

Exports:
- Base protocols: TrackingRepository, GalleryRepository, etc.
- In-memory implementations for testing
- Postgres implementations for production
"""

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
