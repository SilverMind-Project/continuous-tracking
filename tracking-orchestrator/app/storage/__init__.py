"""Storage package — repository protocols and implementations.

Exports:
- Base protocols: PHRepositoryProtocol, WorldObservationRepositoryProtocol, etc.
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
    InMemoryKeyframeRepository,
    InMemoryPHRepository,
    InMemoryPrivacyRepository,
    InMemorySettingsRepository,
    InMemoryWorldObservationRepository,
    PHRepositoryProtocol,
    PrivacyRepository,
    SettingsRepository,
    WorldObservationRepositoryProtocol,
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
    "InMemoryKeyframeRepository",
    "InMemoryPHRepository",
    "InMemoryPrivacyRepository",
    "InMemorySettingsRepository",
    "InMemoryWorldObservationRepository",
    "PHRepositoryProtocol",
    "PrivacyRepository",
    "SettingsRepository",
    "WorldObservationRepositoryProtocol",
]
