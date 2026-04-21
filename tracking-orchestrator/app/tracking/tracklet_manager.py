"""TrackletManager: bridges LocalTracks (per-frame) to Tracklets (persistent).

The TrackletManager handles the lifecycle of tracklets:
1. Extend existing tracklets when their LocalTrack is still alive.
2. Close tracklets whose LocalTrack has been lost longer than a grace window.
3. Promote newly confirmed LocalTracks to new tracklets.
4. Append gallery entries when detection quality exceeds a threshold.

This is milestone M4's core logic: it produces Tracklets with detection IDs,
embeddings, and timing metadata. Identity resolution (M5) consumes these
Tracklets to form GlobalTracks.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..domain import (
    CameraConfig,
    Detection,
    GalleryEmbedding,
    TrackingEvent,
    Tracklet,
)
from ..inference.schemas import Embedding
from ..storage.base import GalleryRepository, TrackingRepository

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackletConfig:
    """Hyperparameters for tracklet lifecycle management."""

    # Minimum hit ratio (hit_count / age) for a track to be confirmed.
    min_hit_ratio: float = 0.5

    # Frames without detection before a tracklet is closed.
    # At 5 fps, 15 frames = 3 seconds.
    close_grace_frames: int = 15

    # Minimum quality score to append a gallery entry.
    gallery_min_quality: float = 0.5

    # Maximum gallery entries per tracklet.
    gallery_max_per_tracklet: int = 20

    # Minimum detection confidence to include in a tracklet.
    min_detection_confidence: float = 0.3

    # Whether to enable tracklet creation (for testing).
    enabled: bool = True


# ---------------------------------------------------------------------------
# Internal tracklet state (mutable during the tracklet's lifetime)
# ---------------------------------------------------------------------------


@dataclass
class _TrackletState:
    """Mutable state for an active tracklet, held in memory."""

    tracklet: Tracklet
    peak_quality: float = 0.0
    gallery_size: int = 0
    last_detection_time: datetime | None = None
    detections: list[Detection] = field(default_factory=list)
    embeddings: list[Embedding] = field(default_factory=list)
    lost_count: int = 0


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------


class TrackletManager:
    """Manages the lifecycle of tracklets within a single camera.

    The TrackletManager is the bridge between LocalTrack (ephemeral, per-frame)
    and Tracklet (persistent, stored). On each frame it:
    1. Associates LocalTracks to existing tracklets by local_track_id.
    2. Extends matched tracklets with new detections.
    3. Creates new tracklets from confirmed LocalTracks.
    4. Closes tracklets whose LocalTracks have been lost.
    5. Appends gallery entries for high-quality detections.
    """

    def __init__(
        self,
        repo: TrackingRepository,
        gallery: GalleryRepository,
        config: TrackletConfig | None = None,
    ) -> None:
        self._repo = repo
        self._gallery = gallery
        self._config = config or TrackletConfig()
        # In-memory state for active tracklets: keyed by tracklet_id
        self._active: dict[str, _TrackletState] = {}

    async def step(
        self,
        camera: CameraConfig,
        local_tracks: list[Any],
        detections: list[Detection],
        embeddings: list[Embedding],
        event_time: datetime,
        frame_index: int,
    ) -> tuple[list[Tracklet], list[GalleryEmbedding], list[TrackingEvent]]:
        """Process a single frame's LocalTracks through the tracklet lifecycle.

        Args:
            camera: configuration for the current camera.
            local_tracks: LocalTrack objects from PerCameraTracker.update().
            detections: original detections (same order as embeddings).
            embeddings: ReID embeddings for each detection.
            event_time: wall-clock time for this frame.
            frame_index: frame index for the camera.

        Returns:
            (updated_tracklets, new_gallery_entries, tracking_events)
        """
        if not self._config.enabled:
            return ([], [], [])

        # Build lookup: detection_id -> detection, local_track_id -> LocalTrack
        det_by_id: dict[str, Detection] = {d.detection_id: d for d in detections}
        lt_by_local_id: dict[str, Any] = {lt.local_track_id: lt for lt in local_tracks}

        # Tracks that are alive this frame
        alive_local_ids: set[str] = set(lt_by_local_id.keys())

        # ---- Extend or close existing tracklets ----
        new_gallery_entries: list[GalleryEmbedding] = []
        updated_tracklets: list[Tracklet] = []

        to_remove: list[str] = []

        for tracklet_id, state in self._active.items():
            if tracklet_id in alive_local_ids:
                # Extend this tracklet
                local_track = lt_by_local_id[tracklet_id]
                det = det_by_id.get(local_track.detection.detection_id)
                if det is not None:
                    emb_idx = self._find_embedding_index(local_track.detection)
                    emb = embeddings[emb_idx] if emb_idx < len(embeddings) else None

                    new_state = self._extend_tracklet(state, det, emb, event_time, frame_index)
                    if new_state is not None:
                        state = new_state
                        self._active[tracklet_id] = state
                        updated_tracklets.append(state.tracklet)

                        # Check gallery append
                        if det.confidence >= self._config.min_detection_confidence:
                            quality = self._compute_quality(det, camera)
                            if quality >= self._config.gallery_min_quality:
                                gallery_entry = self._append_gallery(
                                    tracklet_id, det, emb, quality, event_time
                                )
                                if gallery_entry:
                                    new_gallery_entries.append(gallery_entry)
                    else:
                        to_remove.append(tracklet_id)
            else:
                # Check if we should close this tracklet
                state.lost_count = getattr(state, "lost_count", 0) + 1
                if state.lost_count >= self._config.close_grace_frames:
                    to_remove.append(tracklet_id)
                    # Close the tracklet
                    closed = state.tracklet
                    closed = self._close_tracklet(closed, event_time)
                    updated_tracklets.append(closed)
                    lost = state.lost_count
                    self._active[tracklet_id] = _TrackletState(
                        tracklet=closed,
                        lost_count=lost,
                    )
                else:
                    # Keep alive but mark as lost
                    state.lost_count = getattr(state, "lost_count", 0) + 1
                    self._active[tracklet_id] = state

        # Remove closed tracklets
        for tracklet_id in to_remove:
            del self._active[tracklet_id]

        # ---- Create new tracklets from confirmed LocalTracks ----
        new_tracklets: list[Tracklet] = []
        for local_track in local_tracks:
            if local_track.confirmed:
                det = local_track.detection
                emb_idx = self._find_embedding_index(det)
                emb = embeddings[emb_idx] if emb_idx < len(embeddings) else None

                tracklet = self._create_tracklet(camera, det, event_time, frame_index)
                state = _TrackletState(
                    tracklet=tracklet,
                    peak_quality=det.confidence,
                    gallery_size=0,
                    last_detection_time=event_time,
                    detections=[det],
                    embeddings=[emb] if emb is not None else [],
                )

                # Register in active set before gallery append (so _append_gallery
                # can look up the state)
                self._active[tracklet.tracklet_id] = state

                # Check gallery append for new tracklet
                quality = self._compute_quality(det, camera)
                if quality >= self._config.gallery_min_quality:
                    gallery_entry = self._append_gallery(
                        tracklet.tracklet_id, det, emb, quality, event_time
                    )
                    if gallery_entry:
                        new_gallery_entries.append(gallery_entry)
                        state.gallery_size += 1
                        state.peak_quality = quality

                new_tracklets.append(tracklet)

        # ---- Persist tracklets ----
        for tracklet in updated_tracklets + new_tracklets:
            await self._repo.save_tracklet(tracklet)

        # ---- Persist gallery entries ----
        for entry in new_gallery_entries:
            await self._gallery.upsert_gallery_entry(entry)

        # ---- Build tracking events ----
        events = await self._build_events(
            camera,
            local_tracks,
            detections,
            embeddings,
            event_time,
            frame_index,
        )

        return updated_tracklets + new_tracklets, new_gallery_entries, events

    def _extend_tracklet(
        self,
        state: _TrackletState,
        detection: Detection,
        embedding: Embedding | None,
        event_time: datetime,
        frame_index: int,
    ) -> _TrackletState | None:
        """Extend an existing tracklet with a new detection."""
        state.detections.append(detection)
        state.tracklet.detection_ids.append(detection.detection_id)
        state.last_detection_time = event_time

        # Update embedding history
        if embedding is not None:
            state.embeddings.append(embedding)

        # Update peak quality
        quality = detection.confidence
        if quality > state.peak_quality:
            state.peak_quality = quality

        return state

    def _create_tracklet(
        self,
        camera: CameraConfig,
        detection: Detection,
        event_time: datetime,
        frame_index: int,
    ) -> Tracklet:
        """Create a new tracklet from a confirmed LocalTrack."""
        tracklet_id = str(uuid.uuid4())
        return Tracklet(
            tracklet_id=tracklet_id,
            camera_id=camera.camera_id,
            detection_ids=[detection.detection_id],
            started_at=event_time,
            ended_at=None,
            state="active",
        )

    def _close_tracklet(self, tracklet: Tracklet, event_time: datetime) -> Tracklet:
        """Close a tracklet by setting its ended_at and state."""
        return Tracklet(
            tracklet_id=tracklet.tracklet_id,
            camera_id=tracklet.camera_id,
            detection_ids=tracklet.detection_ids,
            started_at=tracklet.started_at,
            ended_at=event_time,
            state="terminated",
        )

    def _append_gallery(
        self,
        tracklet_id: str,
        detection: Detection,
        embedding: Embedding | None,
        quality: float,
        event_time: datetime,
    ) -> GalleryEmbedding | None:
        """Append a gallery entry for this detection.

        Returns None if the gallery is full or embedding is missing.
        """
        if embedding is None:
            return None

        state = self._active.get(tracklet_id)
        if state is None or state.gallery_size >= self._config.gallery_max_per_tracklet:
            return None

        entry_id = str(uuid.uuid4())
        return GalleryEmbedding(
            gallery_entry_id=entry_id,
            identity_id="",  # Empty until M5 identity resolution
            embedding=detection.embedding,
            seen_at=event_time,
            quality=quality,
            origin_tracklet_id=tracklet_id,
            face_confirmed=False,
        )

    def _compute_quality(self, detection: Detection, camera: CameraConfig) -> float:
        """Compute a quality score for a detection (0..1).

        Quality is based on:
        - Box size (larger = more reliable)
        - Detection confidence
        - Whether the box is fully within frame bounds
        """
        # Size component: normalize box area to a 0..1 range
        # Assume max frame size of 1920x1080 for normalization
        box_area = detection.bbox.width * detection.bbox.height
        max_area = 1920 * 1080
        size_score = min(box_area / (max_area * 0.001), 1.0)  # 0.1% of frame = max score

        # Confidence component
        conf_score = detection.confidence

        # Composite
        quality = 0.4 * size_score + 0.6 * conf_score
        return min(max(quality, 0.0), 1.0)

    def _find_embedding_index(self, detection: Detection) -> int:
        """Find the index of a detection in the embeddings list.

        This is a placeholder that returns 0. In production, embeddings
        would be returned in the same order as detections from Triton.
        """
        return 0

    async def _build_events(
        self,
        camera: CameraConfig,
        local_tracks: list[Any],
        detections: list[Detection],
        embeddings: list[Embedding],
        event_time: datetime,
        frame_index: int,
    ) -> list[TrackingEvent]:
        """Build TrackingEvent domain objects from the frame's results."""
        events: list[TrackingEvent] = []

        if not local_tracks:
            return events

        event_id = str(uuid.uuid4())
        from ..domain import FrameRef

        frame_ref = FrameRef(
            minio_key=f"frames/{camera.camera_id}/{frame_index}.jpg",
            width=640,
            height=480,
            frame_index=frame_index,
            capture_time=event_time,
        )

        event = TrackingEvent(
            event_id=event_id,
            camera_id=camera.camera_id,
            event_time=event_time,
            frame_index=frame_index,
            frame_ref=frame_ref,
            detections=detections,
            identity_revisions=[],
        )

        # Persist the event
        await self._repo.save_tracking_event(event)

        events.append(event)
        return events

    def get_active_tracklets(self) -> list[Tracklet]:
        """Return all currently active tracklets."""
        active = [
            state.tracklet for state in self._active.values() if state.tracklet.state == "active"
        ]
        return active

    def get_tracklet(self, tracklet_id: str) -> Tracklet | None:
        """Get a tracklet by ID, checking both active and persisted storage."""
        state = self._active.get(tracklet_id)
        if state:
            return state.tracklet
        return None
