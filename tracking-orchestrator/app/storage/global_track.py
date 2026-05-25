"""Global track persistence."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from ..domain import GlobalTrack, IdentityCandidate


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
    async def remove_tracklet(self, global_track_id: str, tracklet_id: str) -> GlobalTrack | None:
        """Remove a tracklet from a global track.

        Returns the updated GlobalTrack, or None if the global track
        does not exist.  If the global track has only this tracklet
        remaining it is closed rather than left empty.
        """
        ...

    @abstractmethod
    async def close_global_track(self, global_track_id: str) -> None:
        """Mark a global track as closed (state='closed').

        Called when the tracker loses the track and it disappears from the
        active set.  Prevents stale 'active' rows from accumulating and
        bloating list_active() queries over time.
        """

    @abstractmethod
    async def set_identity_committed_at(
        self,
        global_track_id: str,
        committed_at: datetime,
    ) -> None:
        """Record the time when the current identity was first committed.

        Only called when the identity changes (new commit or reassignment),
        not on every frame.  Cleared by ``clear_identity_committed_at`` on
        demotion.
        """

    @abstractmethod
    async def clear_identity_committed_at(
        self,
        global_track_id: str,
    ) -> None:
        """Clear the committed_at timestamp on demotion to UNKNOWN."""

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


class InMemoryGlobalTrackRepository(GlobalTrackRepository):
    """In-memory store for global tracks."""

    def __init__(self) -> None:
        self._tracks: dict[str, GlobalTrack] = {}
        self._by_tracklet: dict[str, str] = {}

    async def save(self, track: GlobalTrack) -> None:
        old = self._tracks.get(track.global_track_id)
        if old is None:
            sorted_track = GlobalTrack(
                global_track_id=track.global_track_id,
                camera_ids=track.camera_ids,
                tracklet_ids=sorted(set(track.tracklet_ids)),
                started_at=track.started_at,
                last_seen_at=track.last_seen_at,
                current_identity_id=track.current_identity_id,
                current_identity_committed_at=track.current_identity_committed_at,
                state=track.state,
            )
            self._tracks[track.global_track_id] = sorted_track
            for tid in sorted_track.tracklet_ids:
                self._by_tracklet[tid] = track.global_track_id
            return

        merged = GlobalTrack(
            global_track_id=track.global_track_id,
            camera_ids=list(dict.fromkeys(old.camera_ids + track.camera_ids)),
            tracklet_ids=sorted(set(old.tracklet_ids + track.tracklet_ids)),
            started_at=min(old.started_at, track.started_at),
            last_seen_at=max(old.last_seen_at, track.last_seen_at),
            current_identity_id=track.current_identity_id or old.current_identity_id,
            current_identity_committed_at=(
                track.current_identity_committed_at or old.current_identity_committed_at
            ),
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
                current_identity_committed_at=existing.current_identity_committed_at,
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
            current_identity_committed_at=(
                into.current_identity_committed_at or from_track.current_identity_committed_at
            ),
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
                current_identity_committed_at=track.current_identity_committed_at,
                state=track.state,
            )

    async def set_identity_committed_at(
        self,
        global_track_id: str,
        committed_at: datetime,
    ) -> None:
        track = self._tracks.get(global_track_id)
        if track is not None:
            self._tracks[global_track_id] = GlobalTrack(
                global_track_id=track.global_track_id,
                camera_ids=track.camera_ids,
                tracklet_ids=track.tracklet_ids,
                started_at=track.started_at,
                last_seen_at=track.last_seen_at,
                current_identity_id=track.current_identity_id,
                current_identity_committed_at=committed_at,
                state=track.state,
            )

    async def clear_identity_committed_at(self, global_track_id: str) -> None:
        track = self._tracks.get(global_track_id)
        if track is not None:
            self._tracks[global_track_id] = GlobalTrack(
                global_track_id=track.global_track_id,
                camera_ids=track.camera_ids,
                tracklet_ids=track.tracklet_ids,
                started_at=track.started_at,
                last_seen_at=track.last_seen_at,
                current_identity_id=track.current_identity_id,
                current_identity_committed_at=None,
                state=track.state,
            )

    async def get_by_tracklet_id(self, tracklet_id: str) -> GlobalTrack | None:
        gt_id = self._by_tracklet.get(tracklet_id)
        if gt_id is None:
            return None
        return self._tracks.get(gt_id)

    async def remove_tracklet(self, global_track_id: str, tracklet_id: str) -> GlobalTrack | None:
        track = self._tracks.get(global_track_id)
        if track is None:
            return None
        new_tids = [tid for tid in track.tracklet_ids if tid != tracklet_id]
        if not new_tids:
            self._tracks[global_track_id] = GlobalTrack(
                global_track_id=track.global_track_id,
                camera_ids=track.camera_ids,
                tracklet_ids=[],
                started_at=track.started_at,
                last_seen_at=track.last_seen_at,
                current_identity_id=track.current_identity_id,
                state="closed",
            )
            self._by_tracklet.pop(tracklet_id, None)
            return self._tracks[global_track_id]
        self._tracks[global_track_id] = GlobalTrack(
            global_track_id=track.global_track_id,
            camera_ids=track.camera_ids,
            tracklet_ids=new_tids,
            started_at=track.started_at,
            last_seen_at=track.last_seen_at,
            current_identity_id=track.current_identity_id,
            state=track.state,
        )
        self._by_tracklet.pop(tracklet_id, None)
        return self._tracks[global_track_id]

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
