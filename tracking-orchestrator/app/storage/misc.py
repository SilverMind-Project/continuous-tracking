"""Settings, activity, assignment, correction, and privacy zone storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..domain import (
    CameraConfig,
    IdentityCorrection,
    PersonActivity,
    PrivacyZone,
    StreamAssignment,
    StreamConfig,
)


class SettingsRepository(ABC):
    """Persist camera and stream configuration."""

    @abstractmethod
    async def get_camera_config(self, camera_id: str) -> CameraConfig | None:
        """Retrieve camera configuration."""

    @abstractmethod
    async def save_camera_config(self, config: CameraConfig) -> None:
        """Store camera configuration."""

    @abstractmethod
    async def list_camera_configs(self) -> list[CameraConfig]:
        """List all camera configurations."""

    @abstractmethod
    async def get_stream_config(self, stream_id: str) -> StreamConfig | None:
        """Retrieve stream configuration."""

    @abstractmethod
    async def save_stream_config(self, config: StreamConfig) -> None:
        """Store stream configuration."""

    @abstractmethod
    async def list_stream_configs(self) -> list[StreamConfig]:
        """List all stream configurations."""


class ActivityRepository(ABC):
    """Persist dementia activity layer records."""

    @abstractmethod
    async def save_activity(self, activity: PersonActivity) -> str:
        """Store a person activity record. Returns its ID."""

    @abstractmethod
    async def get_activity(self, activity_id: str) -> PersonActivity | None:
        """Retrieve a person activity record by ID."""

    @abstractmethod
    async def list_activities(
        self,
        identity_id: str | None = None,
        activity_type: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
    ) -> list[PersonActivity]:
        """List activity records with optional filters."""


class AssignmentRepository(ABC):
    """Persist stream-to-camera and room assignments."""

    @abstractmethod
    async def save_assignment(self, assignment: StreamAssignment) -> None:
        """Store a stream assignment."""

    @abstractmethod
    async def get_assignment(self, stream_id: str) -> StreamAssignment | None:
        """Retrieve a stream assignment by stream ID."""

    @abstractmethod
    async def list_assignments(self) -> list[StreamAssignment]:
        """List all stream assignments."""


class CorrectionRepository(ABC):
    """Persist manual identity corrections."""

    @abstractmethod
    async def save_correction(self, correction: IdentityCorrection) -> None:
        """Store a correction."""

    @abstractmethod
    async def list_corrections(
        self, global_track_id: str | None = None
    ) -> list[IdentityCorrection]:
        """List corrections."""


class PrivacyRepository(ABC):
    """Persist privacy zones."""

    @abstractmethod
    async def save_privacy_zone(self, zone: PrivacyZone) -> None:
        """Store a privacy zone."""

    @abstractmethod
    async def list_privacy_zones(self, camera_id: str | None = None) -> list[PrivacyZone]:
        """List privacy zones."""


class InMemorySettingsRepository(SettingsRepository):
    def __init__(self) -> None:
        self._cameras: dict[str, CameraConfig] = {}
        self._streams: dict[str, StreamConfig] = {}

    async def get_camera_config(self, camera_id: str) -> CameraConfig | None:
        return self._cameras.get(camera_id)

    async def save_camera_config(self, config: CameraConfig) -> None:
        self._cameras[config.camera_id] = config

    async def list_camera_configs(self) -> list[CameraConfig]:
        return list(self._cameras.values())

    async def get_stream_config(self, stream_id: str) -> StreamConfig | None:
        return self._streams.get(stream_id)

    async def save_stream_config(self, config: StreamConfig) -> None:
        self._streams[config.stream_id] = config

    async def list_stream_configs(self) -> list[StreamConfig]:
        return list(self._streams.values())


class InMemoryActivityRepository(ActivityRepository):
    def __init__(self) -> None:
        self._activities: dict[str, PersonActivity] = {}

    async def save_activity(self, activity: PersonActivity) -> str:
        self._activities[activity.activity_id] = activity
        return activity.activity_id

    async def get_activity(self, activity_id: str) -> PersonActivity | None:
        return self._activities.get(activity_id)

    async def list_activities(
        self,
        identity_id: str | None = None,
        activity_type: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
    ) -> list[PersonActivity]:
        results = list(self._activities.values())
        if identity_id:
            results = [activity for activity in results if activity.identity_id == identity_id]
        if activity_type:
            results = [activity for activity in results if activity.activity_type == activity_type]
        if after:
            results = [activity for activity in results if activity.occurred_at >= after]
        if before:
            results = [activity for activity in results if activity.occurred_at <= before]
        results.sort(key=lambda activity: activity.occurred_at, reverse=True)
        return results[:limit]


class InMemoryAssignmentRepository(AssignmentRepository):
    def __init__(self) -> None:
        self._assignments: dict[str, StreamAssignment] = {}

    async def save_assignment(self, assignment: StreamAssignment) -> None:
        self._assignments[assignment.stream_id] = assignment

    async def get_assignment(self, stream_id: str) -> StreamAssignment | None:
        return self._assignments.get(stream_id)

    async def list_assignments(self) -> list[StreamAssignment]:
        return list(self._assignments.values())


class InMemoryCorrectionRepository(CorrectionRepository):
    def __init__(self) -> None:
        self._corrections: dict[str, IdentityCorrection] = {}

    async def save_correction(self, correction: IdentityCorrection) -> None:
        self._corrections[correction.correction_id] = correction

    async def list_corrections(
        self, global_track_id: str | None = None
    ) -> list[IdentityCorrection]:
        corrections = list(self._corrections.values())
        if global_track_id is not None:
            corrections = [
                correction
                for correction in corrections
                if correction.global_track_id == global_track_id
            ]
        return corrections


class InMemoryPrivacyRepository(PrivacyRepository):
    def __init__(self) -> None:
        self._zones: dict[str, PrivacyZone] = {}

    async def save_privacy_zone(self, zone: PrivacyZone) -> None:
        self._zones[zone.zone_id] = zone

    async def list_privacy_zones(self, camera_id: str | None = None) -> list[PrivacyZone]:
        zones = list(self._zones.values())
        if camera_id is not None:
            zones = [zone for zone in zones if zone.camera_id == camera_id]
        return zones
