"""Frame processing pipeline orchestrator.

Wires together transport, inference, tracking, identity resolution,
trajectory writer, keyframe sampler, persistence, and event emission.
Per-frame business logic lives in ``app/pipeline/stages/``; this module
owns lifecycle, concurrency, and stage runner invocation.

The pipeline runs as a background task in the FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from structlog import get_logger

from ..calibration.state import calibration_state
from ..domain import (
    CameraConfig,
    FrameRef,
    Identity,
    OverlapGroup,
    TrackingEvent,
)
from ..inference.detector import PersonDetector
from ..inference.face_id_client import FaceIdentificationClient
from ..inference.pose import PoseEstimator
from ..observability import metrics as _metrics
from ..pipeline.batcher import FrameBatcher
from ..pipeline.frame_context import FrameContext
from ..pipeline.gallery_cache import GalleryCache
from ..pipeline.stages import (
    ClosePHStage,
    CloseTerminatedStage,
    DetectionBackfillStage,
    DetectStage,
    FaceIdentityStage,
    FetchStage,
    InferenceStage,
    KeyframeStage,
    PostureStage,
    PrivacyStage,
    PublishStage,
    RevisionsStage,
    SpatialProjectionStage,
    StageRunner,
    TrailsStage,
    TrajectoryStage,
    WorldTrackingStage,
)
from ..pipeline.types import FaceIdCameraConfig, FrameImageFetcher, ReidEmbedderProtocol
from ..sampling.keyframe_sampler import KeyframeSampler, SamplerConfig
from ..services.identity_rewriter import (
    IdentityRewriter,
    InMemoryIdentityRewriter,
)
from ..storage.base import (
    BboxAnnotationRepository,
    BehaviorBaselineRepository,
    DementiaSignalRepository,
    DoNotFuseRepository,
    GalleryRepository,
    GlobalTrackRepository,
    InMemoryBboxAnnotationRepository,
    InMemoryBehaviorBaselineRepository,
    InMemoryDementiaSignalRepository,
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
    InMemoryKeyframeRepository,
    InMemoryPHRepository,
    InMemorySettingsRepository,
    InMemoryTrackingRepository,
    InMemoryTrajectoryRepository,
    InMemoryWorldObservationRepository,
    KeyframeRepository,
    PHRepositoryProtocol,
    SettingsRepository,
    TrackingRepository,
    TrajectoryRepository,
    WorldObservationRepositoryProtocol,
)
from ..tracking.floor_projector import FloorProjector
from ..tracking.identity_resolver import IdentityResolver, ResolverConfig
from ..tracking.spatial_projection import SpatialProjectionService
from ..tracking.world.config import WorldTrackerConfig
from ..tracking.world.tracker import WorldTracker
from ..trajectory.dementia_signals import DementiaSignalWorker
from ..trajectory.dementia_signals import SignalConfig as DementiaSignalConfig
from ..trajectory.motion_energy import MotionEnergyTracker
from ..trajectory.posture import GlobalPostureTracker
from ..trajectory.posture_strategy import PostureStrategy
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


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalConfig:
    """Dementia signal thresholds and scheduling."""

    interval_s: int = 60
    enabled: bool = True
    timezone: str = "UTC"
    stillness_threshold_minutes: int = 60
    stillness_emergency_minutes: int = 120
    stillness_motion_floor: float = 0.02
    pacing_room_threshold: int = 8
    pacing_window_minutes: int = 30
    nighttime_transition_threshold: int = 3
    absence_threshold_minutes: int = 60
    bathroom_absolute_threshold_seconds: int = 2700


@dataclass(frozen=True)
class FaceIdConfig:
    """Face identification client configuration."""

    url: str = ""
    cooldown_s: float = 5.0
    timeout_s: float = 2.0
    min_confidence: float = 0.5
    enabled: bool = True
    camera_configs: dict[str, FaceIdCameraConfig] = field(default_factory=dict)


@dataclass
class PipelineDependencies:
    """All external dependencies for FrameProcessingPipeline.

    Every field is Optional. None means "use the InMemory default".
    In production, main.py passes Postgres implementations.
    In unit tests, fields are left None (InMemory) or replaced with controlled fakes.
    """

    detector: PersonDetector | None = None
    tracking_repo: TrackingRepository | None = None
    gallery_repo: GalleryRepository | None = None
    global_track_repo: GlobalTrackRepository | None = None
    trajectory_repo: TrajectoryRepository | None = None
    keyframe_repo: KeyframeRepository | None = None
    signal_repo: DementiaSignalRepository | None = None
    settings_repo: SettingsRepository | None = None
    frame_fetcher: FrameImageFetcher | None = None
    reid_embedder: ReidEmbedderProtocol | None = None
    pose_estimator: PoseEstimator | None = None
    posture_strategy: PostureStrategy | None = None
    identity_rewriter: IdentityRewriter | None = None
    bbox_repo: BboxAnnotationRepository | None = None
    dnf_repo: DoNotFuseRepository | None = None
    # M1 world tracker repositories
    ph_repo: PHRepositoryProtocol | None = None
    obs_repo: WorldObservationRepositoryProtocol | None = None


# NOTE: Every field in PipelineConfig has a default value. These defaults are
# ONLY used in unit tests. In production, main.py explicitly sets every field
# from settings.yaml. The YAML file is the single source of truth for
# production values; dataclass defaults are a testing convenience.
@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the frame processing pipeline."""

    # --- Infrastructure ---
    transport: TransportConfig = field(default_factory=TransportConfig)
    max_concurrent_frames: int = 4
    shutdown_timeout: float = 5.0
    batch_window_s: float = 0.0
    max_batch_size: int = 4
    allow_skeleton: bool = False
    pose_enabled: bool = True

    # --- Tracking ---
    world_tracker: WorldTrackerConfig = field(default_factory=WorldTrackerConfig)

    # --- Identity resolution ---
    resolver: ResolverConfig = field(default_factory=ResolverConfig)
    known_identities: list[Identity] = field(default_factory=list)
    identity_commit_window_s: float = 3.0
    identity_high_confidence_face_threshold: float = 0.80
    # Delay (seconds) after an identity is first committed before backfilling
    # gallery entries with that identity.  Prevents a false identity commit from
    # contaminating the gallery before it has a chance to be revised.  Set to 0
    # to restore the previous immediate-backfill behaviour.
    gallery_identity_backfill_delay_s: float = 10.0
    # Retroactive cross-table rewrite on identity revisions.
    # When True, the IdentityRewriter updates trajectory, dwell, and signal
    # rows so that history reflects the newly committed identity.
    identity_rewrite_on_face_commit: bool = True

    # --- Detection ---
    # Post-decode IoU suppression threshold. Detections whose bboxes overlap
    # an already-kept detection by more than this IoU are dropped before
    # the tracker sees them. Belt-and-braces since the ONNX model bakes NMS.
    detection_iou_dedup_threshold: float = 0.55
    # Per-camera tracker dedup IoU threshold (see TrackerConfig.dedup_iou_threshold).
    tracker_dedup_iou_threshold: float = 0.6
    # Stability gate: tracklets must survive this many frames before being
    # exposed to downstream pipeline stages and publication.
    tracker_min_frames_to_publish: int = 3

    # --- Face ID ---
    face_id: FaceIdConfig = field(default_factory=FaceIdConfig)

    # --- Keyframe sampling ---
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    min_keyframe_detection_confidence: float = 0.5
    camera_room_map: dict[str, str] = field(default_factory=dict)

    # --- Signals ---
    signals: SignalConfig = field(default_factory=SignalConfig)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class FrameProcessingPipeline:
    """End-to-end frame processing pipeline.

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
        self._repo: TrackingRepository | None = None
        self._gallery_repo: GalleryRepository | None = None
        self._gallery_cache: GalleryCache | None = None
        self._global_track_repo: GlobalTrackRepository | None = None
        self._settings_repo: SettingsRepository | None = None
        # Cameras whose FK row we've already ensured this process lifetime.
        self._seen_cameras: set[str] = set()
        self._identity_resolver: IdentityResolver | None = None
        self._revision_publisher: RevisionPublisher | None = None
        # Overlap groups fetched from CC at startup.
        self._overlap_groups: list[OverlapGroup] = []
        # M1 world tracker
        self._world_tracker: WorldTracker | None = None
        self._ph_repo: PHRepositoryProtocol | None = None
        self._obs_repo: WorldObservationRepositoryProtocol | None = None
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
        # Track previously active PH ids for ClosePHStage (WTR3).
        self._prev_active_ph_ids: set[str] = set()
        # Legacy: previously active GT ids for CloseTerminatedStage (deprecated).
        self._prev_active_gt_ids: set[str] = set()
        # Dementia signal worker
        self._signal_publisher: SignalPublisher | None = None
        self._signal_worker: DementiaSignalWorker | None = None
        self._signal_repo: DementiaSignalRepository | None = None
        self._stop_event = asyncio.Event()
        # Face identification (via person-identification-service)
        self._face_id_client: FaceIdentificationClient | None = None
        # Per-tracklet face ID cooldown.  Each tracklet independently throttled
        # so a newly-appeared person gets an immediate call even when another
        # person in the same camera was recently identified.
        self._last_face_id_by_tracklet: dict[str, datetime] = {}
        # Floor projection (homography-based, hot-reloaded from calibration state).
        self._floor_projector: FloorProjector | None = None
        # Pose estimation (RTMPose) + motion energy tracking.
        self._pose_estimator: PoseEstimator | None = None
        self._motion_energy_tracker: MotionEnergyTracker | None = None
        self._posture_tracker: GlobalPostureTracker | None = None
        self._posture_strategy: PostureStrategy | None = None
        self._spatial_projection: SpatialProjectionService | None = None
        # Phase 1: cross-table identity rewriter (orchestrator side).
        self._identity_rewriter: IdentityRewriter | None = None
        # M3: Bbox annotation repository for per-keyframe bbox persistence.
        self._bbox_repo: BboxAnnotationRepository | None = None
        # Phase 7: per-PH trail deque (last 12 normalised foot-points).
        _trail_maxlen = 12
        self._trail_by_ph: dict[str, deque[tuple[float, float]]] = {}
        self._TRAIL_MAXLEN = _trail_maxlen
        # Frame batcher (optional; None when batch_window_s=0).
        self._batcher: FrameBatcher | None = None
        self._stage_runner: StageRunner | None = None

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

    async def initialize(self, deps: PipelineDependencies | None = None) -> None:
        """Initialize all pipeline components.

        Args:
            deps: External dependencies. None means use all InMemory defaults.
        """
        deps = deps or PipelineDependencies()

        # Transport
        self._transport = RedisStreamsTransport(self._config.transport)
        await self._transport.connect()

        # Repositories — use in-memory defaults for skeleton/test mode.
        # In production, concrete Postgres implementations are injected.
        self._repo = deps.tracking_repo or InMemoryTrackingRepository()
        gallery = deps.gallery_repo or InMemoryGalleryRepository()
        self._gallery_repo = gallery
        self._gallery_cache = GalleryCache(gallery)
        self._global_track_repo = deps.global_track_repo or InMemoryGlobalTrackRepository()
        self._settings_repo = deps.settings_repo or InMemorySettingsRepository()
        self._seen_cameras = set()

        # Detector
        self._detector = deps.detector
        self._frame_fetcher = deps.frame_fetcher
        self._reid_embedder = deps.reid_embedder

        # Pose estimation
        self._pose_estimator = deps.pose_estimator
        self._posture_strategy = deps.posture_strategy
        self._motion_energy_tracker = (
            MotionEnergyTracker() if deps.pose_estimator is not None else None
        )
        self._posture_tracker = GlobalPostureTracker(required_consecutive=2)

        # Identity rewriter (orchestrator side)
        self._identity_rewriter = deps.identity_rewriter or InMemoryIdentityRewriter()

        # Bbox annotation repository
        self._bbox_repo = deps.bbox_repo or InMemoryBboxAnnotationRepository()  # type: ignore[assignment]

        # M1: World-coordinate person tracker replaces per-camera + cross-camera.
        self._ph_repo = deps.ph_repo or InMemoryPHRepository()
        self._obs_repo = deps.obs_repo or InMemoryWorldObservationRepository()

        # ---- Identity resolution ----
        self._floor_projector = FloorProjector(calibration_state)
        self._spatial_projection = SpatialProjectionService(calibration_state)

        self._identity_resolver = IdentityResolver(
            tracking_repo=self._repo,
            gallery_repo=gallery,
            global_track_repo=self._global_track_repo,
            identities=self._config.known_identities,
            config=self._config.resolver,
            gallery_cache=self._gallery_cache,
        )

        self._revision_publisher = RevisionPublisher(
            redis_url=self._config.transport.redis_url,
            stream=self._config.transport.revisions_stream,
        )
        await self._revision_publisher.connect()

        self._world_tracker = WorldTracker(
            ph_repo=self._ph_repo,
            obs_repo=self._obs_repo,
            config=self._config.world_tracker,
            identity_resolver=self._identity_resolver,
        )

        # ---- Trajectory writer + keyframe sampler ----
        # Use a single in-memory fallback so the signal worker can read
        # trajectories written by the writer.
        _traj_repo = deps.trajectory_repo or InMemoryTrajectoryRepository()
        self._trajectory_writer = TrajectoryWriter(repo=_traj_repo)

        self._keyframe_sampler = KeyframeSampler(
            repo=deps.keyframe_repo or InMemoryKeyframeRepository(),
            config=self._config.sampler,
            bbox_repo=self._bbox_repo,
        )

        self._scene_publisher = SceneSamplesPublisher(
            redis_url=self._config.transport.redis_url,
            stream=self._config.transport.scene_samples_stream,
        )
        await self._scene_publisher.connect()

        # Signal worker — computes dementia signals from trajectory/dwell data.
        self._signal_repo = deps.signal_repo or InMemoryDementiaSignalRepository()
        if self._config.signals.enabled:
            self._signal_publisher = SignalPublisher(
                redis_url=self._config.transport.redis_url,
                stream=self._config.transport.signals_stream,
            )
            await self._signal_publisher.connect()
            # Build a baseline repository from the same trajectory source.
            baseline_repo: BehaviorBaselineRepository = InMemoryBehaviorBaselineRepository()
            self._signal_worker = DementiaSignalWorker(
                trajectory_repo=_traj_repo,
                signal_repo=self._signal_repo,
                cfg=DementiaSignalConfig(
                    tz_name=self._config.signals.timezone,
                    stillness_threshold_minutes=self._config.signals.stillness_threshold_minutes,
                    stillness_emergency_minutes=self._config.signals.stillness_emergency_minutes,
                    stillness_motion_floor=self._config.signals.stillness_motion_floor,
                    pacing_room_threshold=self._config.signals.pacing_room_threshold,
                    pacing_window_minutes=self._config.signals.pacing_window_minutes,
                    nighttime_transition_threshold=self._config.signals.nighttime_transition_threshold,
                    absence_threshold_minutes=self._config.signals.absence_threshold_minutes,
                    bathroom_absolute_threshold_seconds=self._config.signals.bathroom_absolute_threshold_seconds,
                ),
                baseline_repo=baseline_repo,
            )

        # Frame batcher — buffer frames for parallel cross-camera processing.
        if self._config.batch_window_s > 0:
            self._batcher = FrameBatcher(
                batch_window_s=self._config.batch_window_s,
                max_batch_size=self._config.max_batch_size,
                handler=self._handle_batch,
            )
            logger.info(
                "Frame batcher enabled",
                batch_window_s=self._config.batch_window_s,
                max_batch_size=self._config.max_batch_size,
            )

        # Face identification client — calls person-identification-service.
        if self._config.face_id.enabled and self._config.face_id.url:
            self._face_id_client = FaceIdentificationClient(
                base_url=self._config.face_id.url,
                timeout_s=self._config.face_id.timeout_s,
                min_confidence=self._config.face_id.min_confidence,
            )
            await self._face_id_client.connect()

        # ---- Build the stage runner (order matters) ----
        self._stage_runner = StageRunner(
            [
                FetchStage(frame_fetcher=self._frame_fetcher),
                DetectStage(
                    detector=self._detector,  # type: ignore[arg-type]
                    iou_dedup_threshold=self._config.detection_iou_dedup_threshold,
                ),
                PrivacyStage(),
                SpatialProjectionStage(projection_service=self._spatial_projection),
                InferenceStage(
                    reid_embedder=self._reid_embedder,
                    pose_estimator=self._pose_estimator,
                    pose_enabled=self._config.pose_enabled,
                ),
                FaceIdentityStage(
                    face_id_client=self._face_id_client,
                    tracklet_manager=None,  # tracklets removed in M1
                    gallery_repo=self._gallery_repo,
                    face_id_cooldown_s=self._config.face_id.cooldown_s,
                    face_id_min_confidence=self._config.face_id.min_confidence,
                    face_id_camera_configs=self._config.face_id.camera_configs,
                    last_face_id_by_tracklet=self._last_face_id_by_tracklet,
                ),
                WorldTrackingStage(
                    tracker=self._world_tracker,
                    config=self._config.world_tracker,
                    camera_room_map=self._config.camera_room_map,
                ),
                DetectionBackfillStage(),
                ClosePHStage(
                    trajectory_writer=self._trajectory_writer,
                    motion_energy_tracker=self._motion_energy_tracker,
                    posture_tracker=self._posture_tracker,
                    prev_active_ph_ids=self._prev_active_ph_ids,
                ),
                CloseTerminatedStage(
                    global_track_repo=self._global_track_repo,
                    trajectory_writer=self._trajectory_writer,
                    motion_energy_tracker=self._motion_energy_tracker,
                    posture_tracker=self._posture_tracker,
                    prev_active_gt_ids=self._prev_active_gt_ids,
                ),
                PostureStage(
                    posture_strategy=self._posture_strategy,
                    camera_room_map=self._config.camera_room_map,
                ),
                TrajectoryStage(
                    trajectory_writer=self._trajectory_writer,
                    floor_projector=self._floor_projector,
                    motion_energy_tracker=self._motion_energy_tracker,
                    posture_tracker=self._posture_tracker,
                    tracklet_manager=None,  # tracklets removed in M1
                    camera_room_map=self._config.camera_room_map,
                ),
                KeyframeStage(
                    keyframe_sampler=self._keyframe_sampler,
                    scene_publisher=self._scene_publisher,
                    min_keyframe_detection_confidence=self._config.min_keyframe_detection_confidence,
                ),
                RevisionsStage(
                    revision_publisher=self._revision_publisher,
                    repo=self._repo,
                    identity_rewriter=self._identity_rewriter,
                    bbox_repo=self._bbox_repo,
                    identity_rewrite_on_face_commit=self._config.identity_rewrite_on_face_commit,
                ),
                TrailsStage(
                    trail_by_tracklet=self._trail_by_ph,
                    trail_maxlen=self._TRAIL_MAXLEN,
                ),
                PublishStage(
                    transport=self._transport,
                    camera_room_map=self._config.camera_room_map,
                ),
            ]
        )

        logger.info(
            "Pipeline initialized",
            detector=bool(deps.detector),
            signal_worker=bool(self._signal_worker),
            face_id=bool(self._face_id_client),
            pose=bool(self._pose_estimator),
        )

        if self._detector is None:
            if not self._config.allow_skeleton:
                raise RuntimeError(
                    "Detector not configured and pipeline.allow_skeleton is False. "
                    "Set PIPELINE_ALLOW_SKELETON=true to run without a detector (tests only)."
                )
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

        if self._revision_publisher:
            await self._revision_publisher.disconnect()

        if self._face_id_client:
            await self._face_id_client.disconnect()

        if self._batcher:
            await self._batcher.close()

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
            interval_s=self._config.signals.interval_s,
        )

        first_run = True
        while not self._stop_event.is_set():
            try:
                if not first_run:
                    await asyncio.sleep(self._config.signals.interval_s)
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
        """Push a frame into the batcher (or process directly when batching is off)."""
        if self._batcher is not None:
            try:
                await self._batcher.push(frame)
            except Exception:
                logger.exception("Batcher push failed, processing frame directly")
                await self._process_frame_direct(frame)
        else:
            await self._process_frame_direct(frame)

    async def _process_frame_direct(self, frame: FrameReady) -> None:
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

    async def _handle_batch(self, camera_id: str, frames: list[FrameReady]) -> None:
        """Process a batch of frames for one camera (called by FrameBatcher).

        Frames are already sorted by frame_index. Each camera's frames
        are processed sequentially; different cameras run concurrently.
        """
        assert self._transport is not None
        if self._frame_semaphore is None:
            self._frame_semaphore = asyncio.Semaphore(max(1, self._config.max_concurrent_frames))

        camera_lock = self._camera_locks.setdefault(camera_id, asyncio.Lock())
        async with self._frame_semaphore, camera_lock:
            for frame in frames:
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

    def set_camera_room_map(self, camera_room_map: object) -> None:
        """M2: Inject the live CameraRoomMap into pipeline stages.

        Called at startup after the CCConfigSyncService is created.
        Stages that need room attribution read from this map instead of
        the static ``PipelineConfig.camera_room_map`` dict.
        """
        self._camera_room_map = camera_room_map

    def set_transit_config(
        self,
        transit_detector: object,
        transit_zones: list[object],
        room_transition_publisher: object,
    ) -> None:
        """WTR5: Inject transit zone dependencies into WorldTrackingStage.

        Called at startup after transit zones are loaded from CC sync.
        """
        if self._stage_runner is not None:
            for stage in self._stage_runner._stages:
                if stage.name == "world_tracking":
                    stage._transit_detector = transit_detector
                    stage._transit_zones = transit_zones
                    stage._room_transition_publisher = room_transition_publisher
                    break

    def set_overlap_groups(self, groups: list[OverlapGroup]) -> None:
        """Apply overlap group data from CC.

        Stored for the WorldTracker which handles overlap directly.
        """
        self._overlap_groups = groups

    async def _process_frame(self, frame: FrameReady) -> None:
        """Process a single FrameReady through the full pipeline."""
        if self._gallery_cache is not None:
            self._gallery_cache.invalidate()

        if self._is_stale(frame):
            return
        if self._detector is None:
            await self._skeleton_frame(frame)
            return

        ctx = self._init_context(frame)
        assert self._stage_runner is not None
        await self._stage_runner.run(ctx)

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_stale(frame: FrameReady) -> bool:
        if frame.capture_time_unix_ns <= 0:
            return False
        age_s = datetime.now(UTC).timestamp() - frame.capture_time_unix_ns / 1e9
        if age_s > _MAX_FRAME_AGE_S:
            logger.warning(
                "Stale frame dropped",
                camera_id=frame.camera_id,
                frame_index=frame.frame_index,
                age_s=round(age_s, 1),
            )
            _metrics.metrics.frames_dropped_stale_total.labels(camera_id=frame.camera_id).inc()
            return True
        return False

    def _init_context(self, frame: FrameReady) -> FrameContext:
        return FrameContext(
            frame=frame,
            event_time=datetime.now(UTC),
            capture_time=datetime.fromtimestamp(frame.capture_time_unix_ns / 1e9, tz=UTC),
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
