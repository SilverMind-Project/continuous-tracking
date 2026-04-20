"""Repository protocols and in-memory implementations.

This module defines the storage abstraction layer (repository pattern)
that decouples domain logic from persistence. The protocols define
what the domain layer needs; the implementations handle the details.

Layering rule: domain → storage (never the reverse).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..domain import (
    CameraConfig,
    Detection,
    GalleryEntry,
    GlobalTrack,
    IdentityRevision,
    PersonActivity,
    StreamAssignment,
    StreamConfig,
    Tracklet,
    TrackingEvent,
)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class TrackingRepository(ABC):
    """Persist tracking events, detections, tracklets, and identity data."""

    @abstractmethod
    async def save_tracking_event(self, event: TrackingEvent) -> str:
        """Store a tracking event and return its ID."""

    @abstractmethod
    async def get_tracking_event(self, event_id: str) -> TrackingEvent | None:
        """Retrieve a tracking event by ID."""

    @abstractmethod
    async def save_detections(self, detections: list[Detection]) -> None:
        """Bulk store detections for a single frame."""

    @abstractmethod
    async def save_tracklet(self, tracklet: Tracklet) -> None:
        """Store or update a tracklet."""

    @abstractmethod
    async def get_tracklet(self, tracklet_id: str) -> Tracklet | None:
        """Retrieve a tracklet by ID."""

    @abstractmethod
    async def save_global_track(self, track: GlobalTrack) -> None:
        """Store or update a global track."""

    @abstractmethod
    async def get_global_track(self, global_track_id: str) -> GlobalTrack | None:
        """Retrieve a global track by ID."""

    @abstractmethod
    async def save_identity_revision(self, revision: IdentityRevision) -> None:
        """Store an identity posterior update."""

    @abstractmethod
    async def list_identity_revisions(
        self, global_track_id: str, after: datetime | None = None
    ) -> list[IdentityRevision]:
        """List identity revisions for a track, optionally filtered by time."""


class GalleryRepository(ABC):
    """Persist gallery entries (known persons) and their embeddings."""

    @abstractmethod
    async def upsert_gallery_entry(self, entry: GalleryEntry) -> str:
        """Store or update a gallery entry. Returns the identity ID."""

    @abstractmethod
    async def get_gallery_entry(self, identity_id: str) -> GalleryEntry | None:
        """Retrieve a gallery entry by ID."""

    @abstractmethod
    async def list_gallery_entries(self, active_only: bool = True) -> list[GalleryEntry]:
        """List all gallery entries."""

    @abstractmethod
    async def search_similar(self, embedding: list[float], limit: int = 10) -> list[GalleryEntry]:
        """Nearest-neighbor search over gallery embeddings."""


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


# ---------------------------------------------------------------------------
# In-memory implementations (for testing / dev)
# ---------------------------------------------------------------------------

class InMemoryTrackingRepository(TrackingRepository):
    """In-memory store for tracking data. Used in tests and dev mode."""

    def __init__(self) -> None:
        self._events: dict[str, TrackingEvent] = {}
        self._detections: dict[str, list[Detection]] = {}
        self._tracklets: dict[str, Tracklet] = {}
        self._global_tracks: dict[str, GlobalTrack] = {}
        self._revisions: dict[str, list[IdentityRevision]] = {}

    async def save_tracking_event(self, event: TrackingEvent) -> str:
        self._events[event.event_id] = event
        return event.event_id

    async def get_tracking_event(self, event_id: str) -> TrackingEvent | None:
        return self._events.get(event_id)

    async def save_detections(self, detections: list[Detection]) -> None:
        if not detections:
            return
        key = detections[0].detection_id  # group by frame-level key
        self._detections[key] = detections

    async def save_tracklet(self, tracklet: Tracklet) -> None:
        self._tracklets[tracklet.tracklet_id] = tracklet

    async def get_tracklet(self, tracklet_id: str) -> Tracklet | None:
        return self._tracklets.get(tracklet_id)

    async def save_global_track(self, track: GlobalTrack) -> None:
        existing = self._global_tracks.get(track.global_track_id)
        if existing is None:
            self._global_tracks[track.global_track_id] = track
        else:
            # Merge: append new camera/tracklet IDs
            merged = GlobalTrack(
                global_track_id=track.global_track_id,
                camera_ids=list(dict.fromkeys(existing.camera_ids + track.camera_ids)),
                tracklet_ids=list(dict.fromkeys(existing.tracklet_ids + track.tracklet_ids)),
                started_at=existing.started_at,
                last_seen_at=max(existing.last_seen_at, track.last_seen_at),
                state=track.state,
            )
            self._global_tracks[track.global_track_id] = merged

    async def get_global_track(self, global_track_id: str) -> GlobalTrack | None:
        return self._global_tracks.get(global_track_id)

    async def save_identity_revision(self, revision: IdentityRevision) -> None:
        self._revisions.setdefault(revision.global_track_id, []).append(revision)

    async def list_identity_revisions(
        self, global_track_id: str, after: datetime | None = None
    ) -> list[IdentityRevision]:
        revisions = self._revisions.get(global_track_id, [])
        if after is not None:
            revisions = [r for r in revisions if r.revision_time >= after]
        return revisions


class InMemoryGalleryRepository(GalleryRepository):

    def __init__(self) -> None:
        self._entries: dict[str, GalleryEntry] = {}

    async def upsert_gallery_entry(self, entry: GalleryEntry) -> str:
        self._entries[entry.identity_id] = entry
        return entry.identity_id

    async def get_gallery_entry(self, identity_id: str) -> GalleryEntry | None:
        return self._entries.get(identity_id)

    async def list_gallery_entries(self, active_only: bool = True) -> list[GalleryEntry]:
        entries = list(self._entries.values())
        if active_only:
            entries = [e for e in entries if e.is_active]
        return entries

    async def search_similar(self, embedding: list[float], limit: int = 10) -> list[GalleryEntry]:
        # Naive linear scan — replace with pgvector ANN in Postgres impl.
        import math

        def _cosine_sim(a: list[float], b: list[float]) -> float:
            if len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        entries = await self.list_gallery_entries()
        scored = [(e, _cosine_sim(embedding, e.embedding)) for e in entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]


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
            results = [a for a in results if a.identity_id == identity_id]
        if activity_type:
            results = [a for a in results if a.activity_type == activity_type]
        if after:
            results = [a for a in results if a.timestamp >= after]
        if before:
            results = [a for a in results if a.timestamp <= before]
        results.sort(key=lambda a: a.timestamp, reverse=True)
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
