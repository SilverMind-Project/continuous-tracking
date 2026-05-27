"""Tracking event, tracklet, and detection storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..domain import (
    Detection,
    GlobalTrack,
    IdentityRevision,
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
        self._revisions.setdefault(revision.ph_id, []).append(revision)

    async def list_identity_revisions(
        self, global_track_id: str, after: datetime | None = None
    ) -> list[IdentityRevision]:
        revisions = self._revisions.get(global_track_id, [])
        if after is not None:
            revisions = [revision for revision in revisions if revision.applied_at >= after]
        return revisions
