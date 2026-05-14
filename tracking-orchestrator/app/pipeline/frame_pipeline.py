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
from dataclasses import dataclass, field, replace
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
from ..inference.face_id_client import FaceIdentificationClient
from ..inference.schemas import DetectionBox, Embedding
from ..observability import metrics as _metrics
from ..sampling.keyframe_sampler import KeyframeSampler, SamplerConfig
from ..storage.base import (
    DementiaSignalRepository,
    GalleryRepository,
    GlobalTrackRepository,
    InMemoryDementiaSignalRepository,
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
    InMemoryKeyframeRepository,
    InMemorySettingsRepository,
    InMemoryTrackingRepository,
    InMemoryTrajectoryRepository,
    KeyframeRepository,
    SettingsRepository,
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
from ..trajectory.dementia_signals import DementiaSignalWorker, SignalConfig
from ..trajectory.trajectory_writer import TrajectoryWriter
from ..transport.redis_streams import (
    FrameReady,
    RedisStreamsTransport,
    TransportConfig,
)
from ..transport.revision_publisher import RevisionPublisher
from ..transport.scene_publisher import SceneSamplesPublisher
from ..transport.signal_publisher import SignalPublisher

# Frames older than this are replayed backlog, not live feeds.  They are
# XACK'd normally so the pending-entry list stays clean, but pipeline work
# (inference, tracking, DB writes) is skipped entirely.
_MAX_FRAME_AGE_S: float = 30.0

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


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-Union of two normalised [x1,y1,x2,y2] boxes."""
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
    # Dementia signal computation interval (seconds).
    signal_interval_s: int = 60
    signal_enabled: bool = True
    # IANA timezone name used for time-of-day signal computations.
    timezone: str = "UTC"
    # Face identification via person-identification-service (ArcFace).
    face_id_url: str = ""
    face_id_cooldown_s: float = 5.0
    face_id_timeout_s: float = 2.0
    face_id_min_confidence: float = 0.4
    face_id_enabled: bool = True
    # Per-camera overrides: camera_id -> enabled flag and optional higher threshold.
    # Top-down cameras should set enabled=false; face-level cameras with
    # difficult angles can raise min_confidence above the global default.
    face_id_camera_configs: dict[str, FaceIdCameraConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class FaceIdCameraConfig:
    """Per-camera face identification configuration.

    If *enabled* is False, face identification is skipped entirely for
    this camera (e.g. top-down surveillance cameras where faces are
    never visible).  If *min_confidence* is not None it overrides the
    global ``face_id_min_confidence`` for this camera.
    """

    enabled: bool = True
    min_confidence: float | None = None


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
        self._settings_repo: SettingsRepository | None = None
        # Cameras whose FK row we've already ensured this process lifetime.
        self._seen_cameras: set[str] = set()
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
        # Dementia signal worker
        self._signal_publisher: SignalPublisher | None = None
        self._signal_worker: DementiaSignalWorker | None = None
        self._signal_repo: DementiaSignalRepository | None = None
        self._stop_event = asyncio.Event()
        # Face identification (via person-identification-service)
        self._face_id_client: FaceIdentificationClient | None = None
        self._last_face_id_call: dict[str, datetime] = {}
        # Floor projection (homography-based, hot-reloaded from calibration state).
        self._floor_projector: FloorProjector | None = None

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
        # Signal repository (required for dementia signal worker)
        signal_repo: DementiaSignalRepository | None = None,
        # Settings repository (used to upsert FK-anchoring camera rows).
        settings_repo: SettingsRepository | None = None,
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
        self._settings_repo = settings_repo or InMemorySettingsRepository()
        self._seen_cameras = set()

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

        self._floor_projector = FloorProjector(calibration_state)

        self._cross_camera = CrossCameraAssociator(
            gallery=gallery,
            adjacency=self._adjacency,
            global_track_repo=self._global_track_repo,
            config=self._config.cross_cam,
            floor_projector=self._floor_projector,
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

        # Signal worker — computes dementia signals from trajectory/dwell data.
        self._signal_repo = signal_repo or InMemoryDementiaSignalRepository()
        if self._config.signal_enabled:
            self._signal_publisher = SignalPublisher(
                redis_url=self._config.transport.redis_url,
            )
            await self._signal_publisher.connect()
            self._signal_worker = DementiaSignalWorker(
                trajectory_repo=trajectory_repo or InMemoryTrajectoryRepository(),
                signal_repo=self._signal_repo,
                cfg=SignalConfig(tz_name=self._config.timezone),
            )

        # Face identification client — calls person-identification-service.
        if self._config.face_id_enabled and self._config.face_id_url:
            self._face_id_client = FaceIdentificationClient(
                base_url=self._config.face_id_url,
                timeout_s=self._config.face_id_timeout_s,
                min_confidence=self._config.face_id_min_confidence,
            )
            await self._face_id_client.connect()

        logger.info(
            "Pipeline initialized",
            detector=bool(detector),
            m5_components=True,
            m6_components=True,
            signal_worker=bool(self._signal_worker),
            face_id=bool(self._face_id_client),
        )

        if self._detector is None:
            logger.warning(
                "Detector not configured; running in SKELETON mode — no bboxes will be produced. "
                "Set TRITON_GRPC_URL to enable person detection."
            )

    async def start(self) -> None:
        """Start the pipeline background tasks."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._consume_loop()),
        ]
        if self._signal_worker is not None:
            self._tasks.append(asyncio.create_task(self._signal_loop()))
        self._frame_semaphore = asyncio.Semaphore(max(1, self._config.max_concurrent_frames))
        logger.info("Pipeline started")

    async def stop(self) -> None:
        """Stop the pipeline and wait for tasks to complete."""
        self._running = False
        self._stop_event.set()

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

        if self._signal_publisher:
            await self._signal_publisher.disconnect()

        if self._face_id_client:
            await self._face_id_client.disconnect()

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

    async def _signal_loop(self) -> None:
        """Periodic dementia signal computation loop."""
        assert self._signal_worker is not None
        assert self._signal_publisher is not None

        logger.info(
            "Signal loop started",
            interval_s=self._config.signal_interval_s,
        )

        first_run = True
        while not self._stop_event.is_set():
            try:
                if not first_run:
                    await asyncio.sleep(self._config.signal_interval_s)
                else:
                    first_run = False
                if self._stop_event.is_set():
                    break
                signals = await self._signal_worker.run_once(now=datetime.now(UTC))
                if signals:
                    await self._signal_publisher.publish_batch(signals)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Signal loop error, will retry on next cycle")

        logger.info("Signal loop stopped")

    async def _handle_frame(self, frame: FrameReady) -> None:
        """Process and ACK one frame under global and per-camera concurrency gates."""
        assert self._transport is not None
        if self._frame_semaphore is None:
            self._frame_semaphore = asyncio.Semaphore(max(1, self._config.max_concurrent_frames))

        camera_lock = self._camera_locks.setdefault(frame.camera_id, asyncio.Lock())
        async with self._frame_semaphore, camera_lock:
            start = time.monotonic()
            try:
                await self._ensure_camera_row(frame.camera_id)
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

    async def _ensure_camera_row(self, camera_id: str) -> None:
        """Lazily seed an FK-anchor row in ``cameras`` on first sight per process.

        The orchestrator's ``cameras`` table backs foreign keys from
        tracking_events, tracklets, detections, keyframes, and trajectories.
        Camera metadata (RTSP URL, name, location) is owned by
        cognitive-companion; the orchestrator only needs a row to exist.

        We ``get`` first and only ``save`` when absent — a blind upsert with a
        bare ``CameraConfig(camera_id=...)`` would stomp any real metadata
        written by a future CC sync. The per-camera lock in ``_handle_frame``
        serialises this check, so ``_seen_cameras`` is race-free per camera.
        """
        if camera_id in self._seen_cameras or self._settings_repo is None:
            return
        existing = await self._settings_repo.get_camera_config(camera_id)
        if existing is None:
            await self._settings_repo.save_camera_config(CameraConfig(camera_id=camera_id))
        self._seen_cameras.add(camera_id)

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
                floor_projector=self._floor_projector,
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
        if frame.capture_time_unix_ns > 0:
            age_s = datetime.now(UTC).timestamp() - frame.capture_time_unix_ns / 1e9
            if age_s > _MAX_FRAME_AGE_S:
                logger.warning(
                    "Stale frame dropped",
                    camera_id=frame.camera_id,
                    frame_index=frame.frame_index,
                    age_s=round(age_s, 1),
                )
                _metrics.metrics.frames_dropped_stale_total.labels(
                    camera_id=frame.camera_id
                ).inc()
                return

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

        # Step 4b: Face identification (rate-limited call to person-identification-service)
        face_anchors: list[FaceAnchor] = []
        if self._face_id_client is not None and domain_detections:
            now = datetime.now(UTC)
            if self._should_call_face_id(frame.camera_id, now):
                face_anchors = await self._identify_faces(
                    image=image,
                    frame_width=frame.width,
                    frame_height=frame.height,
                    camera_id=frame.camera_id,
                    domain_detections=domain_detections,
                )

        # ---- M5: Cross-camera association ----
        active_tracklets = (
            self._tracklet_manager.get_active_tracklets() if self._tracklet_manager else []
        )
        active_global_tracks: list[GlobalTrack] = []
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

        # Step 5b: Close trajectory dwells for terminated global tracks.
        # Runs regardless of whether active_tracklets is empty — when it is
        # empty, every previously-active track is considered terminated.
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

        # Step 7: Write trajectory points and manage room dwells.
        if outcome.decisions and self._trajectory_writer:
            traj_time = datetime.now(UTC)
            room_name = self._config.camera_room_map.get(frame.camera_id, "")

            # Build a lookup from global_track_id → best bbox for the
            # current camera, using the tracklet's last_bbox (updated by
            # the tracklet manager in step 4).
            gt_bbox: dict[str, BoundingBox] = {}
            if active_tracklets and self._floor_projector:
                for tracklet in active_tracklets:
                    if tracklet.camera_id != frame.camera_id:
                        continue
                    last_bbox: BoundingBox | None = getattr(tracklet, "last_bbox", None)
                    if last_bbox is None:
                        continue
                    for gt in active_global_tracks:
                        if tracklet.tracklet_id in gt.tracklet_ids:
                            gt_bbox[gt.global_track_id] = last_bbox
                            break

            for decision in outcome.decisions:
                if decision.identity_id is None:
                    continue
                # Project the tracklet's footpoint through the per-camera
                # homography, falling back to uncalibrated (0,0) when no
                # homography is configured for this camera.
                gt_bbox_entry = gt_bbox.get(decision.global_track_id)
                floor_point = (
                    self._floor_projector.project(frame.camera_id, gt_bbox_entry)
                    if gt_bbox_entry is not None and self._floor_projector is not None
                    else FloorPoint(0, 0)
                )
                _top_id, top_prob = decision.posterior.top_identity()
                await self._trajectory_writer.write(
                    identity_id=decision.identity_id,
                    global_track_id=decision.global_track_id,
                    room_name=room_name,
                    floor_point=floor_point,
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

        # Step 9b: Back-fill tracklet_id and global_track_id onto each
        # Detection so the serialised proto carries identity context for the
        # CC subscriber and the live-view overlay.
        if domain_detections and self._tracklet_manager is not None and active_global_tracks:
            tracklet_to_gt: dict[str, str] = {}
            for gt in active_global_tracks:
                for tid in gt.tracklet_ids:
                    tracklet_to_gt[tid] = gt.global_track_id

            updated: list[Detection] = []
            for domain_det in domain_detections:
                tid = self._tracklet_manager.get_tracklet_id_for_detection(domain_det.detection_id)
                gt_id = tracklet_to_gt.get(tid, "")
                updated.append(replace(domain_det, tracklet_id=tid, global_track_id=gt_id))
            domain_detections = updated

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
            detections=domain_detections if detections else None,
            minio_key=frame.minio_key,
            room_name=self._config.camera_room_map.get(frame.camera_id, ""),
            identities=identities or None,
            frame_width=frame.width,
            frame_height=frame.height,
            capture_time_unix_ns=frame.capture_time_unix_ns,
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
            minio_key=frame.minio_key,
            room_name=self._config.camera_room_map.get(frame.camera_id, ""),
            frame_width=frame.width,
            frame_height=frame.height,
            capture_time_unix_ns=frame.capture_time_unix_ns,
        )

        logger.debug(
            "Skeleton frame processed",
            camera_id=frame.camera_id,
            frame_index=frame.frame_index,
        )

    # ------------------------------------------------------------------
    # Face identification helpers
    # ------------------------------------------------------------------

    def _get_face_id_config(self, camera_id: str) -> FaceIdCameraConfig:
        """Return the effective face-id config for *camera_id*.

        Per-camera overrides take precedence; falls back to a default-enabled
        config when no override is defined.
        """
        return self._config.face_id_camera_configs.get(camera_id, FaceIdCameraConfig())

    def _should_call_face_id(self, camera_id: str, now: datetime) -> bool:
        """Return True if enough time has passed since the last call for this camera.

        Also returns False when face-id is disabled for this specific camera
        (e.g. top-down surveillance views).
        """
        cam_cfg = self._get_face_id_config(camera_id)
        if not cam_cfg.enabled:
            return False
        last = self._last_face_id_call.get(camera_id)
        if last is None:
            return True
        return (now - last).total_seconds() >= self._config.face_id_cooldown_s

    async def _identify_faces(
        self,
        image: npt.NDArray[np.uint8],
        frame_width: int,
        frame_height: int,
        camera_id: str,
        domain_detections: list[Detection],
    ) -> list[FaceAnchor]:
        """Call person-identification-service and build FaceAnchors.

        Associates face detections with YOLO person detections via bbox IoU,
        maps to tracklets via the local_track lookup, and returns
        FaceAnchors for the identity resolver.

        If no face_id_client is configured, or the call fails, an empty
        list is returned (graceful degradation).
        """
        if self._face_id_client is None:
            return []

        cam_cfg = self._get_face_id_config(camera_id)
        if not cam_cfg.enabled:
            return []

        face_results = await self._face_id_client.identify(
            image,
            orig_width=frame_width,
            orig_height=frame_height,
        )

        if not face_results:
            return []

        # Effective per-camera confidence threshold.
        min_conf = (
            cam_cfg.min_confidence
            if cam_cfg.min_confidence is not None
            else self._config.face_id_min_confidence
        )

        # Associate each face with the best-matching YOLO detection via IoU.
        face_anchors: list[FaceAnchor] = []
        assigned_detections: set[int] = set()

        for face in face_results:
            if face.person_id == "unknown":
                continue
            if face.confidence < min_conf:
                continue

            best_det_idx = -1
            best_iou = 0.0
            for i, det in enumerate(domain_detections):
                if i in assigned_detections:
                    continue
                det_norm = [
                    det.bbox.x_min / frame_width,
                    det.bbox.y_min / frame_height,
                    det.bbox.x_max / frame_width,
                    det.bbox.y_max / frame_height,
                ]
                iou = _bbox_iou(face.bbox_normalized, det_norm)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            # Minimum IoU threshold for face-to-person association.
            if best_det_idx < 0 or best_iou < 0.1:
                continue

            assigned_detections.add(best_det_idx)

            # Map detection → local_track → tracklet_id
            det = domain_detections[best_det_idx]
            tracklet_id = ""
            if self._tracklet_manager is not None:
                tracklet_id = self._tracklet_manager.get_tracklet_id_for_detection(det.detection_id)

            if not tracklet_id:
                continue

            face_anchors.append(
                FaceAnchor(
                    person_id=face.person_id,
                    confidence=face.confidence,
                    tracklet_id=tracklet_id,
                    camera_id=camera_id,
                    captured_at=datetime.now(UTC),
                )
            )

        self._last_face_id_call[camera_id] = datetime.now(UTC)
        return face_anchors
