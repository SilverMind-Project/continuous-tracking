"""Repository protocols and in-memory implementations."""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ..domain import (
    CameraConfig,
    DementiaSignal,
    Detection,
    GalleryEmbedding,
    GlobalTrack,
    Identity,
    IdentityCandidate,
    IdentityCorrection,
    IdentityRevision,
    PersonActivity,
    PersonTrajectoryPoint,
    PrivacyZone,
    RoomDwell,
    StreamAssignment,
    StreamConfig,
    TaggedKeyframe,
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
    ) -> list[tuple[GalleryEmbedding, float]]:
        """Nearest-neighbor search over gallery embeddings.

        Returns a list of (GalleryEmbedding, similarity_score) tuples,
        sorted by similarity descending. The similarity score is cosine
        similarity in [0, 1].

        Args:
            embedding: query embedding vector.
            limit: maximum number of results.
            camera_id: if provided, filter to gallery entries from this camera.
            max_age_seconds: if provided, filter to entries newer than now - max_age_seconds.
        """

    @abstractmethod
    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
    ) -> list[GalleryEmbedding]:
        """List gallery entries whose origin_tracklet_id is in *tracklet_ids*.

        Used by the identity resolver to build a real query embedding from
        a GlobalTrack's existing gallery entries.
        """

    @abstractmethod
    async def update_identity_for_tracklets(
        self,
        tracklet_ids: set[str],
        identity_id: str,
    ) -> int:
        """Backfill identity_id on all gallery entries for the given tracklets.

        Called after the identity resolver commits an identity so that
        future ReID gallery searches can use these entries as identity
        evidence.  Returns the number of rows updated.
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
    async def list_since(
        self,
        since: datetime,
        open_only: bool = False,
        limit: int = 500,
    ) -> list[GlobalTrack]:
        """List tracks last seen at or after *since*.

        Includes closed tracks when ``open_only=False``.
        """

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
    async def merge_global_tracks(self, into_id: str, from_id: str) -> GlobalTrack | None:
        """Merge one active GlobalTrack into another and close the source track."""

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

    @abstractmethod
    async def close_global_track(self, global_track_id: str) -> None:
        """Mark a global track as closed (state='closed').

        Called when the tracker loses the track and it disappears from the
        active set.  Prevents stale 'active' rows from accumulating and
        bloating list_active() queries over time.
        """

    @abstractmethod
    async def update_last_posterior(
        self,
        global_track_id: str,
        posterior_json: dict[str, float],
        at: datetime,
    ) -> None:
        """Persist the latest Bayesian posterior distribution for the inspector drawer.

        *posterior_json* is a mapping of identity_id -> probability (including
        the ``"UNKNOWN"`` key). Written every frame; does not create a revision.
        """

    @abstractmethod
    async def batch_update_last_seen_at(
        self,
        global_track_ids: list[str],
        at: datetime,
    ) -> None:
        """Refresh last_seen_at for all given global tracks.

        Called every frame by CrossCameraAssociator so the active-track window
        (``last_seen_at > now() - 5 minutes``) never evicts a track whose
        tracklet is still alive.  Uses a single UPDATE rather than per-row saves
        to keep per-frame DB overhead low.
        """


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
            last_bbox=tracklet.last_bbox,
            last_floor_point=tracklet.last_floor_point,
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
            # Keep entries that belong to active identities OR have no identity yet
            # (unowned gallery entries written before identity resolution).
            entries = [
                entry
                for entry in entries
                if entry.identity_id in active_ids or entry.identity_id == ""
            ]
        return entries

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
    ) -> list[tuple[GalleryEmbedding, float]]:
        # Intentional O(n): this in-memory implementation favors clarity over ANN performance.
        entries = await self.list_gallery_entries()
        if camera_id is not None:
            entries = [e for e in entries if e.camera_id == camera_id]
        if max_age_seconds is not None:
            cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
            entries = [e for e in entries if e.seen_at >= cutoff]
        scored = [(entry, _cosine_sim(embedding, entry.embedding)) for entry in entries]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
    ) -> list[GalleryEmbedding]:
        if not tracklet_ids:
            return []
        entries = [
            entry for entry in self._entries.values() if entry.origin_tracklet_id in tracklet_ids
        ]
        entries.sort(key=lambda e: e.seen_at, reverse=True)
        return entries[:limit]

    async def update_identity_for_tracklets(
        self,
        tracklet_ids: set[str],
        identity_id: str,
    ) -> int:
        updated = 0
        for entry_id, entry in self._entries.items():
            if entry.origin_tracklet_id in tracklet_ids and not entry.identity_id:
                # Replace with an updated copy (GalleryEmbedding is frozen).
                self._entries[entry_id] = GalleryEmbedding(
                    gallery_entry_id=entry.gallery_entry_id,
                    identity_id=identity_id,
                    embedding=entry.embedding,
                    seen_at=entry.seen_at,
                    quality=entry.quality,
                    origin_tracklet_id=entry.origin_tracklet_id,
                    face_confirmed=entry.face_confirmed,
                )
                updated += 1
        return updated


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

    async def list_since(
        self,
        since: datetime,
        open_only: bool = False,
        limit: int = 500,
    ) -> list[GlobalTrack]:
        tracks = [
            t
            for t in self._tracks.values()
            if t.last_seen_at >= since and (not open_only or t.state == "active")
        ]
        tracks.sort(key=lambda t: t.last_seen_at, reverse=True)
        return tracks[:limit]

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

    async def merge_global_tracks(self, into_id: str, from_id: str) -> GlobalTrack | None:
        into = self._tracks.get(into_id)
        from_track = self._tracks.get(from_id)
        if into is None or from_track is None or into_id == from_id:
            return into

        merged = GlobalTrack(
            global_track_id=into.global_track_id,
            camera_ids=list(dict.fromkeys(into.camera_ids + from_track.camera_ids)),
            tracklet_ids=list(dict.fromkeys(into.tracklet_ids + from_track.tracklet_ids)),
            started_at=min(into.started_at, from_track.started_at),
            last_seen_at=max(into.last_seen_at, from_track.last_seen_at),
            current_identity_id=into.current_identity_id or from_track.current_identity_id,
            state="active",
        )
        closed_source = GlobalTrack(
            global_track_id=from_track.global_track_id,
            camera_ids=from_track.camera_ids,
            tracklet_ids=from_track.tracklet_ids,
            started_at=from_track.started_at,
            last_seen_at=from_track.last_seen_at,
            current_identity_id=from_track.current_identity_id,
            state="closed",
        )
        self._tracks[into_id] = merged
        self._tracks[from_id] = closed_source
        for tid in merged.tracklet_ids:
            self._by_tracklet[tid] = into_id
        return merged

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

    async def close_global_track(self, global_track_id: str) -> None:
        track = self._tracks.get(global_track_id)
        if track is not None and track.state == "active":
            self._tracks[global_track_id] = GlobalTrack(
                global_track_id=track.global_track_id,
                camera_ids=track.camera_ids,
                tracklet_ids=track.tracklet_ids,
                started_at=track.started_at,
                last_seen_at=track.last_seen_at,
                current_identity_id=track.current_identity_id,
                state="closed",
            )

    async def update_last_posterior(
        self,
        global_track_id: str,
        posterior_json: dict[str, float],
        at: datetime,
    ) -> None:
        # In-memory: no-op. The posterior is only useful for the CC inspector
        # drawer which reads the Postgres column directly.
        pass

    async def batch_update_last_seen_at(
        self,
        global_track_ids: list[str],
        at: datetime,
    ) -> None:
        for gt_id in global_track_ids:
            track = self._tracks.get(gt_id)
            if track is not None and track.last_seen_at < at:
                self._tracks[gt_id] = GlobalTrack(
                    global_track_id=track.global_track_id,
                    camera_ids=track.camera_ids,
                    tracklet_ids=track.tracklet_ids,
                    started_at=track.started_at,
                    last_seen_at=at,
                    current_identity_id=track.current_identity_id,
                    state=track.state,
                )


class TrajectoryRepository(ABC):
    """Persist person trajectory points and room dwell intervals."""

    @abstractmethod
    async def save_trajectory_point(self, point: PersonTrajectoryPoint) -> None:
        """Append a confirmed trajectory point."""

    @abstractmethod
    async def save_room_dwell(self, dwell: RoomDwell) -> None:
        """Open a new room dwell interval."""

    @abstractmethod
    async def update_room_dwell(self, dwell: RoomDwell) -> None:
        """Close (or update) an existing room dwell interval."""

    @abstractmethod
    async def get_open_dwell(self, identity_id: str, global_track_id: str) -> RoomDwell | None:
        """Return the open (not-yet-exited) dwell for a track, if any."""

    @abstractmethod
    async def list_trajectory_points(
        self,
        identity_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[PersonTrajectoryPoint]:
        """List trajectory points with optional filters."""

    @abstractmethod
    async def list_room_dwells(
        self,
        identity_id: str | None = None,
        room_name: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[RoomDwell]:
        """List room dwell intervals with optional filters."""


class KeyframeRepository(ABC):
    """Persist tagged keyframes."""

    @abstractmethod
    async def save_keyframe(self, keyframe: TaggedKeyframe) -> None:
        """Store a tagged keyframe."""

    @abstractmethod
    async def get_keyframe(self, keyframe_id: str) -> TaggedKeyframe | None:
        """Retrieve a tagged keyframe by ID."""

    @abstractmethod
    async def update_retention(self, keyframe_id: str, expires_at: datetime) -> bool:
        """Update keyframe retention expiry. Returns True if the row existed."""

    @abstractmethod
    async def list_keyframes(
        self,
        tracklet_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[TaggedKeyframe]:
        """List keyframes with optional filters."""


class InMemoryTrajectoryRepository(TrajectoryRepository):
    """In-memory store for trajectory points and room dwells."""

    def __init__(self) -> None:
        self._points: list[PersonTrajectoryPoint] = []
        # Keyed by (identity_id, global_track_id) -> open RoomDwell
        self._open_dwells: dict[tuple[str | None, str], RoomDwell] = {}
        self._closed_dwells: list[RoomDwell] = []

    async def save_trajectory_point(self, point: PersonTrajectoryPoint) -> None:
        self._points.append(point)

    async def save_room_dwell(self, dwell: RoomDwell) -> None:
        key = (dwell.identity_id, dwell.global_track_id)
        self._open_dwells[key] = dwell

    async def update_room_dwell(self, dwell: RoomDwell) -> None:
        key = (dwell.identity_id, dwell.global_track_id)
        self._open_dwells.pop(key, None)
        self._closed_dwells.append(dwell)

    async def get_open_dwell(self, identity_id: str, global_track_id: str) -> RoomDwell | None:
        return self._open_dwells.get((identity_id, global_track_id))

    async def list_trajectory_points(
        self,
        identity_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[PersonTrajectoryPoint]:
        results = list(self._points)
        if identity_id is not None:
            results = [p for p in results if p.identity_id == identity_id]
        if global_track_id is not None:
            results = [p for p in results if p.global_track_id == global_track_id]
        if after is not None:
            results = [p for p in results if p.observed_at >= after]
        results.sort(key=lambda p: p.observed_at, reverse=True)
        return results[:limit]

    async def list_room_dwells(
        self,
        identity_id: str | None = None,
        room_name: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[RoomDwell]:
        all_dwells = list(self._open_dwells.values()) + self._closed_dwells
        if identity_id is not None:
            all_dwells = [d for d in all_dwells if d.identity_id == identity_id]
        if room_name is not None:
            all_dwells = [d for d in all_dwells if d.room_name == room_name]
        if after is not None:
            all_dwells = [d for d in all_dwells if d.entered_at >= after]
        all_dwells.sort(key=lambda d: d.entered_at, reverse=True)
        return all_dwells[:limit]


class InMemoryKeyframeRepository(KeyframeRepository):
    """In-memory store for tagged keyframes."""

    def __init__(self) -> None:
        self._keyframes: dict[str, TaggedKeyframe] = {}

    async def save_keyframe(self, keyframe: TaggedKeyframe) -> None:
        self._keyframes[keyframe.keyframe_id] = keyframe

    async def get_keyframe(self, keyframe_id: str) -> TaggedKeyframe | None:
        return self._keyframes.get(keyframe_id)

    async def update_retention(self, keyframe_id: str, expires_at: datetime) -> bool:
        keyframe = self._keyframes.get(keyframe_id)
        if keyframe is None:
            return False
        self._keyframes[keyframe_id] = TaggedKeyframe(
            keyframe_id=keyframe.keyframe_id,
            tracklet_id=keyframe.tracklet_id,
            global_track_id=keyframe.global_track_id,
            camera_id=keyframe.camera_id,
            minio_key=keyframe.minio_key,
            captured_at=keyframe.captured_at,
            annotations=keyframe.annotations,
            tag_reason=keyframe.tag_reason,
            expires_at=expires_at,
        )
        return True

    async def list_keyframes(
        self,
        tracklet_id: str | None = None,
        global_track_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[TaggedKeyframe]:
        results = list(self._keyframes.values())
        if tracklet_id is not None:
            results = [k for k in results if k.tracklet_id == tracklet_id]
        if global_track_id is not None:
            results = [k for k in results if k.global_track_id == global_track_id]
        if after is not None:
            results = [k for k in results if k.captured_at >= after]
        results.sort(key=lambda k: k.captured_at, reverse=True)
        return results[:limit]


class DementiaSignalRepository(ABC):
    """Persist dementia signals."""

    @abstractmethod
    async def upsert_signal(self, signal: DementiaSignal) -> None:
        """Store or update a dementia signal."""

    @abstractmethod
    async def list_signals(
        self,
        identity_id: str | None = None,
        signal_kind: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 200,
    ) -> list[DementiaSignal]:
        """List dementia signals with optional filters."""


@dataclass(frozen=True)
class HourlyActivitySummary:
    """Per-hour activity summary for a resident."""

    transition_count: int
    observed_minutes: int


@dataclass(frozen=True)
class StillnessEpisode:
    """A historical contiguous low-motion episode."""

    room_name: str
    posture: str
    duration_seconds: int
    min_motion_energy: float
    occurred_at: datetime


class BehaviorBaselineRepository(ABC):
    """Summarise raw trajectory/dwell history for robust signal baselines.

    All methods are independent of the signal repository — baselines are
    derived from raw behaviour data, never from previously emitted signals.
    """

    @abstractmethod
    async def dwell_durations(
        self,
        identity_id: str,
        room_predicate: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[float]:
        """Return closed-dwell durations (seconds) for rooms matching *room_predicate*."""

    @abstractmethod
    async def hourly_activity(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[int, HourlyActivitySummary]:
        """Return per-hour-of-day room-transition counts and observed-minutes."""

    @abstractmethod
    async def stillness_episodes(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[StillnessEpisode]:
        """Return historical low-motion episodes for baseline comparison."""


class InMemoryBehaviorBaselineRepository(BehaviorBaselineRepository):
    """In-memory baseline repository backed by trajectory/dwell lists."""

    def __init__(
        self,
        points: list[PersonTrajectoryPoint] | None = None,
        dwells: list[RoomDwell] | None = None,
    ) -> None:
        self.points: list[PersonTrajectoryPoint] = points or []
        self.dwells: list[RoomDwell] = dwells or []

    async def dwell_durations(
        self,
        identity_id: str,
        room_predicate: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[float]:
        results: list[float] = []
        for d in self.dwells:
            if d.identity_id != identity_id:
                continue
            if d.exited_at is None or d.duration_seconds is None:
                continue
            if room_predicate and room_predicate not in d.room_name.lower():
                continue
            if since is not None and d.entered_at < since:
                continue
            if until is not None and d.entered_at > until:
                continue
            results.append(float(d.duration_seconds))
        return results

    async def hourly_activity(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[int, HourlyActivitySummary]:
        buckets: dict[int, dict[str, int]] = {}
        sorted_pts = sorted(
            [p for p in self.points if p.identity_id == identity_id],
            key=lambda p: p.observed_at,
        )
        prev_room: str | None = None
        for p in sorted_pts:
            if since and p.observed_at < since:
                continue
            if until and p.observed_at > until:
                continue
            hour = p.observed_at.hour
            b = buckets.setdefault(hour, {"transitions": 0, "minutes": 0})
            b["minutes"] += 1
            if prev_room is not None and p.room_name != prev_room:
                b["transitions"] += 1
            prev_room = p.room_name
        return {
            h: HourlyActivitySummary(
                transition_count=v["transitions"],
                observed_minutes=v["minutes"],
            )
            for h, v in buckets.items()
        }

    async def stillness_episodes(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[StillnessEpisode]:
        episodes: list[StillnessEpisode] = []
        for d in self.dwells:
            if d.identity_id != identity_id:
                continue
            if d.exited_at is None:
                continue
            if since and d.entered_at < since:
                continue
            if until and d.entered_at > until:
                continue
            if d.min_motion_energy is not None or d.still_seconds > 0:
                episodes.append(
                    StillnessEpisode(
                        room_name=d.room_name,
                        posture=d.primary_posture,
                        duration_seconds=d.duration_seconds or 0,
                        min_motion_energy=d.min_motion_energy or 0.0,
                        occurred_at=d.entered_at,
                    )
                )
        return episodes


class InMemoryDementiaSignalRepository(DementiaSignalRepository):
    """In-memory store for dementia signals."""

    def __init__(self) -> None:
        self._signals: dict[str, DementiaSignal] = {}

    async def upsert_signal(self, signal: DementiaSignal) -> None:
        self._signals[signal.signal_id] = signal

    async def list_signals(
        self,
        identity_id: str | None = None,
        signal_kind: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 200,
    ) -> list[DementiaSignal]:
        results = list(self._signals.values())
        if identity_id is not None:
            results = [s for s in results if s.identity_id == identity_id]
        if signal_kind is not None:
            results = [s for s in results if s.signal_kind == signal_kind]
        if after is not None:
            results = [s for s in results if s.emitted_at >= after]
        if before is not None:
            results = [s for s in results if s.emitted_at <= before]
        results.sort(key=lambda s: s.emitted_at, reverse=True)
        return results[:limit]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
