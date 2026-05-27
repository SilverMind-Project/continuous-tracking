"""Storage package — repository protocols and implementations.

Exports:
- Base protocols: TrackingRepository, GalleryRepository, PHRepositoryProtocol, etc.
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
    InMemoryPHRepository,
    InMemoryPrivacyRepository,
    InMemorySettingsRepository,
    InMemoryTrackingRepository,
    InMemoryWorldObservationRepository,
    PHRepositoryProtocol,
    PrivacyRepository,
    SettingsRepository,
    TrackingRepository,
    WorldObservationRepositoryProtocol,
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
    "InMemoryPHRepository",
    "InMemoryPrivacyRepository",
    "InMemorySettingsRepository",
    "InMemoryTrackingRepository",
    "InMemoryWorldObservationRepository",
    "PHRepositoryProtocol",
    "PrivacyRepository",
    "SettingsRepository",
    "TrackingRepository",
    "WorldObservationRepositoryProtocol",
]
