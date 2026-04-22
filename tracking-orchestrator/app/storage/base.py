"""Repository protocols and in-memory implementations."""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

from ..domain import (
    CameraConfig,
    Detection,
    GalleryEmbedding,
    GlobalTrack,
    Identity,
    IdentityCandidate,
    IdentityCorrection,
    IdentityRevision,
    PersonActivity,
    PrivacyZone,
    StreamAssignment,
    StreamConfig,
    TrackingEvent,
    Tracklet,
)


class TrackingRepository(ABC):
    """Persist tracking events, detections, tracklets, and identity data."""

    @abstractmethod
    async def save_tracking_event(self, event: TrackingEvent) -> str:
        """Store a tracking event and return its ID."""

    @abstractmethod
    async def get_tracking_event(self, event_id: str) -> TrackingEvent | None:
        """Retrieve a tracking event by ID."""

    @abstractmethod
    async def save_detections(self, event_id: str, detections: list[Detection]) -> None:
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
        """List identity revisions for a track."""


class GalleryRepository(ABC):
    """Persist identities and their gallery embeddings."""

    @abstractmethod
    async def upsert_identity(self, identity: Identity) -> str:
        """Store or update an identity. Returns the identity ID."""

    @abstractmethod
    async def get_identity(self, identity_id: str) -> Identity | None:
        """Retrieve an identity by ID."""

    @abstractmethod
    async def list_identities(self, active_only: bool = True) -> list[Identity]:
        """List all identities."""

    @abstractmethod
    async def upsert_gallery_entry(self, entry: GalleryEmbedding) -> str:
        """Store or update a gallery embedding. Returns the identity ID."""

    @abstractmethod
    async def get_gallery_entry(self, gallery_entry_id: str) -> GalleryEmbedding | None:
        """Retrieve a gallery embedding row by ID."""

    @abstractmethod
    async def list_gallery_entries(
        self, identity_id: str | None = None, active_only: bool = True
    ) -> list[GalleryEmbedding]:
        """List gallery embeddings."""

    @abstractmethod
    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
    ) -> list[GalleryEmbedding]:
        """Nearest-neighbor search over gallery embeddings.

        Args:
            embedding: query embedding vector.
            limit: maximum number of results.
            camera_id: if provided, filter to gallery entries from this camera.
            max_age_seconds: if provided, filter to entries newer than now - max_age_seconds.
        """


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


class GlobalTrackRepository(ABC):
    """Persist global tracks and their identity assignments."""

    @abstractmethod
    async def save(self, track: GlobalTrack) -> None:
        """Store or update a global track."""

    @abstractmethod
    async def get(self, global_track_id: str) -> GlobalTrack | None:
        """Retrieve a global track by ID."""

    @abstractmethod
    async def list_active(self) -> list[GlobalTrack]:
        """List all active global tracks."""

    @abstractmethod
    async def merge_tracklets(
        self,
        tracklet_ids: list[str],
        camera_ids: list[str],
        existing: GlobalTrack | None = None,
    ) -> GlobalTrack:
        """Create or extend a global track from tracklet IDs.

        If existing is provided, the new tracklet IDs and camera IDs are
        merged into it. Otherwise a new GlobalTrack is created.
        """

    @abstractmethod
    async def assign_identity(
        self,
        global_track_id: str,
        identity_id: str | None,
        candidates: list[IdentityCandidate] | None = None,
    ) -> None:
        """Assign an identity to a global track."""

    @abstractmethod
    async def get_by_tracklet_id(self, tracklet_id: str) -> GlobalTrack | None:
        """Find the global track that contains a given tracklet ID."""


class InMemoryTrackingRepository(TrackingRepository):
    """In-memory store for tracking data."""

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

    async def save_detections(self, event_id: str, detections: list[Detection]) -> None:
        self._detections[event_id] = detections

    async def save_tracklet(self, tracklet: Tracklet) -> None:
        existing = self._tracklets.get(tracklet.tracklet_id)
        if existing is None:
            self._tracklets[tracklet.tracklet_id] = tracklet
            return

        ended_at = existing.ended_at
        if tracklet.ended_at is not None:
            ended_at = tracklet.ended_at if ended_at is None else max(ended_at, tracklet.ended_at)

        state = existing.state
        if existing.state == "active" and tracklet.state == "terminated":
            state = "terminated"

        self._tracklets[tracklet.tracklet_id] = Tracklet(
            tracklet_id=tracklet.tracklet_id,
            camera_id=tracklet.camera_id,
            detection_ids=list(dict.fromkeys(existing.detection_ids + tracklet.detection_ids)),
            started_at=min(existing.started_at, tracklet.started_at),
            ended_at=ended_at,
            state=state,
        )

    async def get_tracklet(self, tracklet_id: str) -> Tracklet | None:
        return self._tracklets.get(tracklet_id)

    async def save_global_track(self, track: GlobalTrack) -> None:
        existing = self._global_tracks.get(track.global_track_id)
        if existing is None:
            self._global_tracks[track.global_track_id] = track
            return

        self._global_tracks[track.global_track_id] = GlobalTrack(
            global_track_id=track.global_track_id,
            camera_ids=list(dict.fromkeys(existing.camera_ids + track.camera_ids)),
            tracklet_ids=list(dict.fromkeys(existing.tracklet_ids + track.tracklet_ids)),
            started_at=min(existing.started_at, track.started_at),
            last_seen_at=max(existing.last_seen_at, track.last_seen_at),
            state=track.state,
        )

    async def get_global_track(self, global_track_id: str) -> GlobalTrack | None:
        return self._global_tracks.get(global_track_id)

    async def save_identity_revision(self, revision: IdentityRevision) -> None:
        self._revisions.setdefault(revision.global_track_id, []).append(revision)

    async def list_identity_revisions(
        self, global_track_id: str, after: datetime | None = None
    ) -> list[IdentityRevision]:
        revisions = self._revisions.get(global_track_id, [])
        if after is not None:
            revisions = [revision for revision in revisions if revision.revision_time >= after]
        return revisions


class InMemoryGalleryRepository(GalleryRepository):
    def __init__(self) -> None:
        self._identities: dict[str, Identity] = {}
        self._entries: dict[str, GalleryEmbedding] = {}

    async def upsert_identity(self, identity: Identity) -> str:
        self._identities[identity.identity_id] = identity
        return identity.identity_id

    async def get_identity(self, identity_id: str) -> Identity | None:
        return self._identities.get(identity_id)

    async def list_identities(self, active_only: bool = True) -> list[Identity]:
        identities = list(self._identities.values())
        if active_only:
            identities = [identity for identity in identities if identity.is_active]
        return identities

    async def upsert_gallery_entry(self, entry: GalleryEmbedding) -> str:
        self._entries[entry.gallery_entry_id] = entry
        return entry.identity_id

    async def get_gallery_entry(self, gallery_entry_id: str) -> GalleryEmbedding | None:
        return self._entries.get(gallery_entry_id)

    async def list_gallery_entries(
        self, identity_id: str | None = None, active_only: bool = True
    ) -> list[GalleryEmbedding]:
        entries = list(self._entries.values())
        if identity_id is not None:
            entries = [entry for entry in entries if entry.identity_id == identity_id]
        if active_only:
            active_ids = {
                identity.identity_id for identity in await self.list_identities(active_only=True)
            }
            entries = [entry for entry in entries if entry.identity_id in active_ids]
        return entries

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
    ) -> list[GalleryEmbedding]:
        # Intentional O(n): this in-memory implementation favors clarity over ANN performance.
        entries = await self.list_gallery_entries()
        if camera_id is not None:
            entries = [e for e in entries if e.camera_id == camera_id]
        if max_age_seconds is not None:
            cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
            entries = [e for e in entries if e.seen_at >= cutoff]
        scored = [(entry, _cosine_sim(embedding, entry.embedding)) for entry in entries]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [entry for entry, _ in scored[:limit]]


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


class InMemoryGlobalTrackRepository(GlobalTrackRepository):
    """In-memory store for global tracks."""

    def __init__(self) -> None:
        self._tracks: dict[str, GlobalTrack] = {}
        # Reverse index: tracklet_id -> global_track_id
        self._by_tracklet: dict[str, str] = {}

    async def save(self, track: GlobalTrack) -> None:
        old = self._tracks.get(track.global_track_id)
        if old is None:
            # Sort tracklet_ids for deterministic ordering.
            sorted_track = GlobalTrack(
                global_track_id=track.global_track_id,
                camera_ids=track.camera_ids,
                tracklet_ids=sorted(set(track.tracklet_ids)),
                started_at=track.started_at,
                last_seen_at=track.last_seen_at,
                current_identity_id=track.current_identity_id,
                state=track.state,
            )
            self._tracks[track.global_track_id] = sorted_track
            for tid in sorted_track.tracklet_ids:
                self._by_tracklet[tid] = track.global_track_id
            return

        # Merge: extend camera_ids and tracklet_ids (sorted for determinism).
        merged = GlobalTrack(
            global_track_id=track.global_track_id,
            camera_ids=list(dict.fromkeys(old.camera_ids + track.camera_ids)),
            tracklet_ids=sorted(set(old.tracklet_ids + track.tracklet_ids)),
            started_at=min(old.started_at, track.started_at),
            last_seen_at=max(old.last_seen_at, track.last_seen_at),
            current_identity_id=track.current_identity_id or old.current_identity_id,
            state=track.state,
        )
        self._tracks[track.global_track_id] = merged
        for tid in track.tracklet_ids:
            self._by_tracklet[tid] = track.global_track_id

    async def get(self, global_track_id: str) -> GlobalTrack | None:
        return self._tracks.get(global_track_id)

    async def list_active(self) -> list[GlobalTrack]:
        return [t for t in self._tracks.values() if t.state == "active"]

    async def merge_tracklets(
        self,
        tracklet_ids: list[str],
        camera_ids: list[str],
        existing: GlobalTrack | None = None,
    ) -> GlobalTrack:
        if existing is not None:
            merged = GlobalTrack(
                global_track_id=existing.global_track_id,
                camera_ids=list(dict.fromkeys(existing.camera_ids + camera_ids)),
                tracklet_ids=list(dict.fromkeys(existing.tracklet_ids + tracklet_ids)),
                started_at=existing.started_at,
                last_seen_at=datetime.now(UTC),
                current_identity_id=existing.current_identity_id,
                state="active",
            )
            await self.save(merged)
            return merged

        new_gt = GlobalTrack(
            global_track_id=str(uuid.uuid4()),
            camera_ids=list(dict.fromkeys(camera_ids)),
            tracklet_ids=list(dict.fromkeys(tracklet_ids)),
            started_at=datetime.now(UTC),
            last_seen_at=datetime.now(UTC),
            current_identity_id=None,
            state="active",
        )
        await self.save(new_gt)
        return new_gt

    async def assign_identity(
        self,
        global_track_id: str,
        identity_id: str | None,
        candidates: list[IdentityCandidate] | None = None,
    ) -> None:
        track = self._tracks.get(global_track_id)
        if track is not None:
            self._tracks[global_track_id] = GlobalTrack(
                global_track_id=track.global_track_id,
                camera_ids=track.camera_ids,
                tracklet_ids=track.tracklet_ids,
                started_at=track.started_at,
                last_seen_at=track.last_seen_at,
                current_identity_id=identity_id,
                state=track.state,
            )

    async def get_by_tracklet_id(self, tracklet_id: str) -> GlobalTrack | None:
        gt_id = self._by_tracklet.get(tracklet_id)
        if gt_id is None:
            return None
        return self._tracks.get(gt_id)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
