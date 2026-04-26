"""Frame processing pipeline for M6.

This is the core orchestrator that wires together:
1. Transport (Redis Streams consumer for FrameReady)
2. Inference (Triton person detector + ReID)
3. Tracking (BoT-SORT per-camera tracker)
4. Tracklet management (lifecycle, gallery append)
5. Cross-camera association (GlobalTrack formation)
6. Identity resolution (Bayesian posterior + retroactive revision)
7. Trajectory writer (person_trajectories + room_dwells)
8. Keyframe sampler (tagged_keyframes + scene.samples publisher)
9. Persistence (repository layer)
10. Event emission (Redis Streams producer + revision publisher)

The pipeline runs as a background task in the FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ..calibration.state import calibration_state
from ..domain import (
    CameraConfig,
    Detection,
    FaceAnchor,
    FloorPoint,
    FrameRef,
    GlobalTrack,
    Identity,
    IdentityRevision,
    TaggedKeyframe,
    TrackingEvent,
)
from ..inference.detector import PersonDetector
from ..inference.schemas import DetectionBox, Embedding
from ..sampling.keyframe_sampler import KeyframeSampler, SamplerConfig
from ..storage.base import (
    GalleryRepository,
    GlobalTrackRepository,
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
    InMemoryKeyframeRepository,
    InMemoryTrackingRepository,
    InMemoryTrajectoryRepository,
    KeyframeRepository,
    TrackingRepository,
    TrajectoryRepository,
)
from ..tracking.camera_adjacency import AdjacencyEdge as GraphAdjacencyEdge
from ..tracking.camera_adjacency import CameraAdjacency
from ..tracking.cross_camera import CrossCamConfig, CrossCameraAssociator
from ..tracking.floor_projector import FloorProjector
from ..tracking.identity_resolver import IdentityResolver, ResolverConfig
from ..tracking.tracker import PerCameraTrackers, TrackerConfig
from ..tracking.tracklet_manager import TrackletConfig, TrackletManager
from ..trajectory.trajectory_writer import TrajectoryWriter
from ..transport.redis_streams import (
    FrameReady,
    RedisStreamsTransport,
    TransportConfig,
)
from ..transport.revision_publisher import RevisionPublisher
from ..transport.scene_publisher import SceneSamplesPublisher

logger = get_logger(__name__)


class FrameImageFetcher(Protocol):
    """Loads an RGB frame image from object storage."""

    async def fetch_rgb(self, minio_key: str) -> npt.NDArray[np.uint8]:
        """Return an RGB uint8 image for the object key."""


class ReidEmbedderProtocol(Protocol):
    """Appearance embedding boundary used by the pipeline."""

    async def embed_batch(
        self,
        crops: list[npt.NDArray[np.uint8]],
    ) -> list[Embedding]:
        """Return one ReID embedding per crop."""


def _crop_detection(
    image: npt.NDArray[np.uint8],
    det: DetectionBox,
) -> npt.NDArray[np.uint8]:
    """Crop one normalized detector box from an RGB image."""
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(det.x1 * w)))
    y1 = max(0, min(h - 1, int(det.y1 * h)))
    x2 = max(x1 + 1, min(w, int(det.x2 * w)))
    y2 = max(y1 + 1, min(h, int(det.y2 * h)))
    return np.ascontiguousarray(image[y1:y2, x1:x2])


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the frame processing pipeline."""

    transport: TransportConfig = field(default_factory=TransportConfig)
    tracklet: TrackletConfig = field(default_factory=TrackletConfig)
    detector_confidence: float = 0.25
    max_concurrent_frames: int = 4
    shutdown_timeout: float = 5.0
    # Cross-camera association
    cross_cam: CrossCamConfig = field(default_factory=CrossCamConfig)
    # Identity resolution
    resolver: ResolverConfig = field(default_factory=ResolverConfig)
    # Known identities (from the persons table)
    known_identities: list[Identity] = field(default_factory=list)
    # Keyframe sampling
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    # Camera-to-room mapping (camera_id -> room_name); resolved from stream assignments.
    camera_room_map: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class FrameProcessingPipeline:
    """End-to-end frame processing pipeline for M4.

    Usage::

        config = PipelineConfig()
        pipeline = FrameProcessingPipeline(config)
        await pipeline.initialize(detector)
        await pipeline.start()
        # ... run until shutdown ...
        await pipeline.stop()
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._transport: RedisStreamsTransport | None = None
        self._detector: PersonDetector | None = None
        self._tracklet_manager: TrackletManager | None = None
        self._tracker: PerCameraTrackers | None = None
        self._repo: TrackingRepository | None = None
        self._gallery_repo: GalleryRepository | None = None
        self._global_track_repo: GlobalTrackRepository | None = None
        self._cross_camera: CrossCameraAssociator | None = None
        self._identity_resolver: IdentityResolver | None = None
        self._revision_publisher: RevisionPublisher | None = None
        self._adjacency: CameraAdjacency | None = None
        self._adjacency_version: int = -1
        self._frame_fetcher: FrameImageFetcher | None = None
        self._reid_embedder: ReidEmbedderProtocol | None = None
        # M6
        self._trajectory_writer: TrajectoryWriter | None = None
        self._keyframe_sampler: KeyframeSampler | None = None
        self._scene_publisher: SceneSamplesPublisher | None = None
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._frame_tasks: set[asyncio.Task[None]] = set()
        self._frame_semaphore: asyncio.Semaphore | None = None
        self._camera_locks: dict[str, asyncio.Lock] = {}
        # Track previously active global track IDs for close_track wiring (Issue #23).
        self._prev_active_gt_ids: set[str] = set()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tracking_repo(self) -> TrackingRepository | None:
        """Public accessor for the tracking repository."""
        return self._repo

    @property
    def global_track_repo(self) -> GlobalTrackRepository | None:
        """Public accessor for the global track repository."""
        return self._global_track_repo

    @property
    def revision_publisher(self) -> RevisionPublisher | None:
        """Public accessor for the revision publisher."""
        return self._revision_publisher

    async def initialize(
        self,
        detector: PersonDetector | None = None,
        # Tracking repository (required for tracklet manager)
        tracking_repo: TrackingRepository | None = None,
        # Gallery repository (required for tracklet manager + identity resolver)
        gallery_repo: GalleryRepository | None = None,
        # Global track repository (required for cross-camera associator)
        global_track_repo: GlobalTrackRepository | None = None,
        # Trajectory repository (required for trajectory writer)
        trajectory_repo: TrajectoryRepository | None = None,
        # Keyframe repository (required for keyframe sampler)
        keyframe_repo: KeyframeRepository | None = None,
        # Frame image source + ReID embedder for the real inference path.
        frame_fetcher: FrameImageFetcher | None = None,
        reid_embedder: ReidEmbedderProtocol | None = None,
    ) -> None:
        """Initialize all pipeline components.

        Args:
            detector: Triton-backed person detector. If None, skeleton mode.
            tracking_repo: Tracking repository. Defaults to InMemoryTrackingRepository.
            gallery_repo: Gallery repository. Defaults to InMemoryGalleryRepository.
            global_track_repo: Global track repository. Defaults to InMemoryGlobalTrackRepository.
            trajectory_repo: Trajectory repository. Defaults to InMemoryTrajectoryRepository.
            keyframe_repo: Keyframe repository. Defaults to InMemoryKeyframeRepository.
            frame_fetcher: Object-storage backed RGB frame loader.
            reid_embedder: Triton-backed ReID embedder.
        """
        # Transport
        self._transport = RedisStreamsTransport(self._config.transport)
        await self._transport.connect()

        # Repositories — use in-memory defaults for skeleton/test mode.
        # In production, concrete Postgres implementations are injected.
        self._repo = tracking_repo or InMemoryTrackingRepository()
        gallery = gallery_repo or InMemoryGalleryRepository()
        self._gallery_repo = gallery
        self._global_track_repo = global_track_repo or InMemoryGlobalTrackRepository()

        # Detector
        self._detector = detector
        self._frame_fetcher = frame_fetcher
        self._reid_embedder = reid_embedder

        # Tracklet manager
        tracker = PerCameraTrackers(TrackerConfig())

        self._tracklet_manager = TrackletManager(
            repo=self._repo,
            gallery=gallery,
            config=self._config.tracklet,
        )

        # Store tracker reference for pipeline step
        self._tracker = tracker

        # ---- M5: Cross-camera + identity resolution ----
        self._adjacency = CameraAdjacency()

        self._cross_camera = CrossCameraAssociator(
            gallery=gallery,
            adjacency=self._adjacency,
            global_track_repo=self._global_track_repo,
            config=self._config.cross_cam,
            floor_projector=FloorProjector(calibration_state),
        )
        self._sync_adjacency()

        self._identity_resolver = IdentityResolver(
            tracking_repo=self._repo,
            gallery_repo=gallery,
            global_track_repo=self._global_track_repo,
            identities=self._config.known_identities,
            config=self._config.resolver,
        )

        self._revision_publisher = RevisionPublisher(
            redis_url=self._config.transport.redis_url,
        )
        await self._revision_publisher.connect()

        # ---- M6: Trajectory writer + keyframe sampler ----
        self._trajectory_writer = TrajectoryWriter(
            repo=trajectory_repo or InMemoryTrajectoryRepository(),
        )

        self._keyframe_sampler = KeyframeSampler(
            repo=keyframe_repo or InMemoryKeyframeRepository(),
            config=self._config.sampler,
        )

        self._scene_publisher = SceneSamplesPublisher(
            redis_url=self._config.transport.redis_url,
        )
        await self._scene_publisher.connect()

        logger.info(
            "Pipeline initialized",
            detector=bool(detector),
            m5_components=True,
            m6_components=True,
        )

    async def start(self) -> None:
        """Start the pipeline background tasks."""
        if self._running:
            return

        self._running = True
        self._tasks = [
            asyncio.create_task(self._consume_loop()),
        ]
        self._frame_semaphore = asyncio.Semaphore(max(1, self._config.max_concurrent_frames))
        logger.info("Pipeline started")

    async def stop(self) -> None:
        """Stop the pipeline and wait for tasks to complete."""
        self._running = False

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._frame_tasks:
            await asyncio.gather(*self._frame_tasks, return_exceptions=True)
            self._frame_tasks.clear()

        # Close all open dwells in the trajectory writer to prevent unbounded
        # per-track state from accumulating across restarts.
        if self._trajectory_writer:
            await self._trajectory_writer.close_all(closed_at=datetime.now(UTC))

        if self._transport:
            await self._transport.disconnect()

        if self._scene_publisher:
            await self._scene_publisher.disconnect()

        self._tasks.clear()
        logger.info("Pipeline stopped")

    async def _consume_loop(self) -> None:
        """Main consumption loop: read FrameReady, process, publish."""
        if self._transport is None:
            logger.error("Cannot start consume loop: transport not initialized")
            return

        logger.info("Consume loop started")

        while self._running:
            try:
                # Read a batch and process frames concurrently. Per-camera
                # locks preserve tracker ordering for each camera.
                async for frame in self._transport.consume_frames(
                    count=max(1, self._config.max_concurrent_frames)
                ):
                    if not self._running:
                        break
                    task = asyncio.create_task(self._handle_frame(frame))
                    self._frame_tasks.add(task)
                    task.add_done_callback(self._frame_tasks.discard)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Consume loop error, retrying in 1s")
                await asyncio.sleep(1)

        logger.info("Consume loop stopped")

    async def _handle_frame(self, frame: FrameReady) -> None:
        """Process and ACK one frame under global and per-camera concurrency gates."""
        assert self._transport is not None
        if self._frame_semaphore is None:
            self._frame_semaphore = asyncio.Semaphore(max(1, self._config.max_concurrent_frames))

        camera_lock = self._camera_locks.setdefault(frame.camera_id, asyncio.Lock())
        async with self._frame_semaphore, camera_lock:
            start = time.monotonic()
            try:
                await self._process_frame(frame)
                await self._transport.ack_frame(frame)
                latency_us = int((time.monotonic() - start) * 1e6)
                logger.debug(
                    "Frame processed",
                    camera_id=frame.camera_id,
                    frame_index=frame.frame_index,
                    latency_us=latency_us,
                )
            except Exception:
                logger.exception(
                    "Frame processing failed",
                    camera_id=frame.camera_id,
                    frame_index=frame.frame_index,
                )
                await self._transport.publish_response(
                    frame,
                    success=False,
                    error_code="processing_error",
                )

    def _sync_adjacency(self) -> None:
        """Hot-reload operator-pushed calibration adjacency into the associator."""
        if self._adjacency_version == calibration_state.version:
            return
        new_adjacency = CameraAdjacency()
        for edge in calibration_state.adjacency_edges:
            new_adjacency.add_edge(
                GraphAdjacencyEdge(
                    from_camera=edge.from_camera,
                    to_camera=edge.to_camera,
                    max_transition_seconds=edge.max_transit_s,
                    overlap=edge.overlap,
                )
            )
        self._adjacency = new_adjacency
        self._adjacency_version = calibration_state.version
        if self._gallery_repo is not None and self._global_track_repo is not None:
            self._cross_camera = CrossCameraAssociator(
                gallery=self._gallery_repo,
                adjacency=new_adjacency,
                global_track_repo=self._global_track_repo,
                config=self._config.cross_cam,
            )

    async def _process_frame(self, frame: FrameReady) -> None:
        """Process a single FrameReady message through the full pipeline.

        Steps (M4):
        1. (Skeleton mode) Skip MinIO fetch — in production, fetch JPEG.
        2. Run person detection via Triton (if detector available).
        3. Run per-camera tracking (BoT-SORT).
        4. Run TrackletManager step.

        Steps (M5):
        5. Cross-camera association (GlobalTrack formation).
        6. Identity resolution (Bayesian posterior + retroactive revision).

        Steps (M6):
        7. Trajectory writer (person_trajectories + room_dwells).
        8. Keyframe sampler (tagged_keyframes + scene.samples).
        9. Persist identity revisions.
        10. Publish tracking event.
        """
        if self._detector is None or self._tracklet_manager is None or self._tracker is None:
            # Skeleton mode: produce a zero-detection event
            await self._skeleton_frame(frame)
            return

        self._sync_adjacency()

        event_time = datetime.now(UTC)
        capture_time = datetime.fromtimestamp(frame.capture_time_unix_ns / 1e9, tz=UTC)

        # Step 1: Fetch frame from object storage. Tests may omit the fetcher
        # and use a detector mock; in that case a blank image keeps the real
        # detector call path active without external services.
        if self._frame_fetcher is not None:
            image = await self._frame_fetcher.fetch_rgb(frame.minio_key)
        else:
            image = np.zeros((max(frame.height, 1), max(frame.width, 1), 3), dtype=np.uint8)

        # Step 2: Run detection
        detections = await self._detector.detect(image)
        domain_detections: list[Detection] = []

        if detections:
            crops = [_crop_detection(image, det) for det in detections]
            embeddings: list[Embedding] = (
                await self._reid_embedder.embed_batch(crops)
                if self._reid_embedder is not None
                else []
            )

            for det in detections:
                from ..domain import BoundingBox

                bbox = BoundingBox(
                    x_min=int(det.x1 * frame.width),
                    y_min=int(det.y1 * frame.height),
                    x_max=int(det.x2 * frame.width),
                    y_max=int(det.y2 * frame.height),
                )
                det_idx = len(domain_detections)
                emb = embeddings[det_idx] if det_idx < len(embeddings) else None

                domain_det = Detection(
                    detection_id=str(uuid.uuid4()),
                    camera_id=frame.camera_id,
                    bbox=bbox,
                    embedding=emb.tolist() if emb is not None else [],
                    capture_time=capture_time,
                    event_time=event_time,
                    confidence=det.confidence,
                )
                domain_detections.append(domain_det)

            # Step 3: Per-camera tracking
            local_tracks = self._tracker.update(
                camera_id=frame.camera_id,
                detections=domain_detections,
                embeddings=embeddings or None,
                frame_index=frame.frame_index,
            )

            # Step 4: Tracklet management
            camera_config = CameraConfig(camera_id=frame.camera_id)
            await self._tracklet_manager.step(
                camera=camera_config,
                local_tracks=local_tracks,
                detections=domain_detections,
                embeddings=embeddings,
                event_time=event_time,
                frame_index=frame.frame_index,
            )

            detection_count = len(local_tracks)
        else:
            detection_count = 0

        # ---- M5: Cross-camera association ----
        active_tracklets = (
            self._tracklet_manager.get_active_tracklets() if self._tracklet_manager else []
        )
        active_global_tracks: list[GlobalTrack] = []
        face_anchors: list[FaceAnchor] = []
        new_revisions: list[IdentityRevision] = []
        from ..domain import ResolveOutcome

        outcome: ResolveOutcome = ResolveOutcome()

        if active_tracklets:
            assert self._cross_camera is not None
            assert self._identity_resolver is not None

            # Step 5: Cross-camera association
            active_global_tracks = await self._cross_camera.associate(
                active_tracklets,
                captured_at=datetime.now(UTC),
            )

            # Step 5b: Close trajectory dwells for terminated global tracks.
            # A global track is considered terminated when it was active in a
            # previous frame but no longer appears in the current active list
            # (all its tracklets have been closed by the tracklet manager).
            current_gt_ids = {gt.global_track_id for gt in active_global_tracks}
            terminated_gt_ids = self._prev_active_gt_ids - current_gt_ids
            if terminated_gt_ids and self._trajectory_writer:
                traj_close_time = datetime.now(UTC)
                for gt_id in terminated_gt_ids:
                    logger.debug(
                        "Closing trajectory dwell for terminated global track",
                        global_track_id=gt_id,
                    )
                    await self._trajectory_writer.close_track(gt_id, closed_at=traj_close_time)
            self._prev_active_gt_ids = current_gt_ids

            # Step 6: Identity resolution
            outcome = await self._identity_resolver.resolve(
                global_tracks=active_global_tracks,
                new_face_anchors=face_anchors,
                captured_at=datetime.now(UTC),
            )

            # Apply decisions: update GlobalTrack identity assignments.
            if self._global_track_repo:
                for decision in outcome.decisions:
                    if decision.identity_id is not None or decision.revises_previous:
                        await self._global_track_repo.assign_identity(
                            global_track_id=decision.global_track_id,
                            identity_id=decision.identity_id,
                        )

            # Collect revisions to emit.
            new_revisions = list(outcome.revisions)

        # Step 7: Write trajectory points and manage room dwells.
        if outcome.decisions and self._trajectory_writer:
            traj_time = datetime.now(UTC)
            room_name = self._config.camera_room_map.get(frame.camera_id, "")
            for decision in outcome.decisions:
                if decision.identity_id is None:
                    continue
                _top_id, top_prob = decision.posterior.top_identity()
                await self._trajectory_writer.write(
                    identity_id=decision.identity_id,
                    global_track_id=decision.global_track_id,
                    room_name=room_name,
                    floor_point=FloorPoint(0, 0),  # homography-based projection added in M9
                    captured_at=traj_time,
                    identity_confidence=top_prob,
                )

        # Step 8: Keyframe sampling (periodic per tracklet + triggered on identity change).
        if self._keyframe_sampler and active_tracklets:
            sample_time = datetime.now(UTC)
            revised_gt_ids = {rev.global_track_id for rev in new_revisions}
            for tracklet in active_tracklets:
                gt_id = next(
                    (
                        gt.global_track_id
                        for gt in active_global_tracks
                        if tracklet.tracklet_id in gt.tracklet_ids
                    ),
                    tracklet.tracklet_id,
                )
                annotations: dict[str, object] = {
                    "tracklet_id": tracklet.tracklet_id,
                    "camera_id": tracklet.camera_id,
                }

                sampled: TaggedKeyframe | None
                # Trigger on identity revision.
                if gt_id in revised_gt_ids:
                    sampled = await self._keyframe_sampler.trigger_sample(
                        tracklet_id=tracklet.tracklet_id,
                        global_track_id=gt_id,
                        camera_id=tracklet.camera_id,
                        minio_key=frame.minio_key,
                        captured_at=sample_time,
                        annotations=annotations,
                        tag_reason="identity_changed",
                    )
                else:
                    sampled = await self._keyframe_sampler.maybe_sample(
                        tracklet_id=tracklet.tracklet_id,
                        global_track_id=gt_id,
                        camera_id=tracklet.camera_id,
                        minio_key=frame.minio_key,
                        captured_at=sample_time,
                        annotations=annotations,
                    )

                if sampled is not None and self._scene_publisher:
                    await self._scene_publisher.publish(sampled)

        # Step 9: Persist identity revisions.
        if new_revisions and self._revision_publisher:
            await self._revision_publisher.publish_many(new_revisions)
            if self._repo:
                for rev in new_revisions:
                    await self._repo.save_identity_revision(revision=rev)

        # Step 10: Publish tracking event with identity + room context so the
        # CC-side TrackingEventSubscriber can write PersonLocationState directly.
        identities: dict[str, tuple[str, float]] = {}
        if active_tracklets and outcome.decisions:
            for decision in outcome.decisions:
                if decision.identity_id is None:
                    continue
                _top_id, top_prob = decision.posterior.top_identity()
                identities[decision.global_track_id] = (decision.identity_id, top_prob)

        assert self._transport is not None
        await self._transport.publish_event(
            camera_id=frame.camera_id,
            event_time=datetime.now(UTC),
            frame_index=frame.frame_index,
            detection_count=detection_count,
            detections=domain_detections if detections else None,
            minio_key=frame.minio_key,
            room_name=self._config.camera_room_map.get(frame.camera_id, ""),
            identities=identities or None,
        )

        if new_revisions:
            logger.info(
                "Identity revisions emitted",
                camera_id=frame.camera_id,
                frame_index=frame.frame_index,
                revision_count=len(new_revisions),
            )

    async def _skeleton_frame(self, frame: FrameReady) -> None:
        """Process a frame in skeleton mode (no detector, no tracking).

        Produces a zero-detection tracking event and publishes it.
        This allows the pipeline to run end-to-end without Triton.
        """
        event_time = datetime.now(UTC)

        # Build a minimal tracking event
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            camera_id=frame.camera_id,
            event_time=event_time,
            frame_index=frame.frame_index,
            frame_ref=FrameRef(
                minio_key=frame.minio_key,
                width=frame.width,
                height=frame.height,
                frame_index=frame.frame_index,
                capture_time=datetime.fromtimestamp(frame.capture_time_unix_ns / 1e9, tz=UTC),
            ),
            detections=[],
        )

        if self._repo:
            await self._repo.save_tracking_event(event)

        assert self._transport is not None
        await self._transport.publish_event(
            camera_id=frame.camera_id,
            event_time=event_time,
            frame_index=frame.frame_index,
            detection_count=0,
            minio_key=frame.minio_key,
            room_name=self._config.camera_room_map.get(frame.camera_id, ""),
        )

        logger.debug(
            "Skeleton frame processed",
            camera_id=frame.camera_id,
            frame_index=frame.frame_index,
        )
