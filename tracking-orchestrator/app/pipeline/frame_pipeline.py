"""Frame processing pipeline orchestrator.

Wires together transport, inference, tracking, identity resolution,
trajectory writer, keyframe sampler, persistence, and event emission.
Per-frame business logic lives in ``app/pipeline/stages/``; this module
owns lifecycle, concurrency, and stage runner invocation.

The pipeline runs as a background task in the FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from structlog import get_logger

from ..calibration.state import calibration_state
from ..domain import (
    CameraConfig,
    Identity,
    OverlapGroup,
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
from ..pipeline.stages._room_maps import camera_room_name
from ..pipeline.types import FaceIdCameraConfig, FrameImageFetcher, ReidEmbedderProtocol
from ..sampling.keyframe_sampler import KeyframeSampler, SamplerConfig
from ..services.camera_room_map import CameraRoomMap, RoomPolygonMap
from ..services.identity_rewriter import (
    IdentityRewriter,
    InMemoryIdentityRewriter,
)
from ..services.transit_zone_map import TransitZoneMap
from ..storage.base import (
    BboxAnnotationRepository,
    BehaviorBaselineRepository,
    CameraTopologyRepository,
    CoPresenceRepository,
    DementiaSignalRepository,
    GalleryRepository,
    InMemoryBboxAnnotationRepository,
    InMemoryBehaviorBaselineRepository,
    InMemoryDementiaSignalRepository,
    InMemoryGalleryRepository,
    InMemoryKeyframeRepository,
    InMemoryPHRepository,
    InMemorySettingsRepository,
    InMemoryTrajectoryRepository,
    InMemoryWorldObservationRepository,
    KeyframeRepository,
    PHRepositoryProtocol,
    SettingsRepository,
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
from ..transport.dwell_publisher import DwellPublisher
from ..transport.ph_continuation_publisher import PHContinuationPublisher
from ..transport.presence_publisher import PresencePublisher
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


def _group_frames_by_camera(frames: list[FrameReady]) -> dict[str, list[FrameReady]]:
    """Group frames by camera and sort each camera stream by frame_index."""
    grouped: dict[str, list[FrameReady]] = {}
    for frame in frames:
        grouped.setdefault(frame.camera_id, []).append(frame)
    for camera_frames in grouped.values():
        camera_frames.sort(key=lambda f: f.frame_index)
    return grouped


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
    absence_threshold_minutes: int = 90
    bathroom_absolute_threshold_seconds: int = 2700


@dataclass(frozen=True)
class FaceIdConfig:
    """Face identification client configuration."""

    url: str = ""
    cooldown_s: float = 5.0
    timeout_s: float = 2.0
    min_confidence: float = 0.6
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
    gallery_repo: GalleryRepository | None = None
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
    # World tracker repositories
    ph_repo: PHRepositoryProtocol | None = None
    obs_repo: WorldObservationRepositoryProtocol | None = None
    topology_repo: CameraTopologyRepository | None = None
    copresence_repo: CoPresenceRepository | None = None
    overlap_groups: list[OverlapGroup] | None = None


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
    cross_camera_detector_batching: bool = True
    allow_skeleton: bool = False
    pose_enabled: bool = True

    # --- Tracking ---
    world_tracker: WorldTrackerConfig = field(default_factory=WorldTrackerConfig)

    # --- Identity resolution ---
    resolver: ResolverConfig = field(default_factory=ResolverConfig)
    known_identities: list[Identity] = field(default_factory=list)
    # Retroactive cross-table rewrite on identity revisions.
    # When True, the IdentityRewriter updates trajectory, dwell, and signal
    # rows so that history reflects the newly committed identity.
    identity_rewrite_on_face_commit: bool = True

    # --- Detection ---
    # Post-decode IoU suppression threshold. Detections whose bboxes overlap
    # an already-kept detection by more than this IoU are dropped before
    # the tracker sees them. Belt-and-braces since the ONNX model bakes NMS.
    detection_iou_dedup_threshold: float = 0.55

    # --- Face ID ---
    face_id: FaceIdConfig = field(default_factory=FaceIdConfig)

    # --- Keyframe sampling ---
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    min_keyframe_detection_confidence: float = 0.5
    # Maximum per-camera publish rate for tracking.events (UI live feed).
    # Inference runs every frame regardless.
    live_publish_max_hz: float = 3.0

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
        self._gallery_repo: GalleryRepository | None = None
        self._gallery_cache: GalleryCache | None = None
        self._settings_repo: SettingsRepository | None = None
        # Cameras whose FK row we've already ensured this process lifetime.
        self._seen_cameras: set[str] = set()
        self._identity_resolver: IdentityResolver | None = None
        self._revision_publisher: RevisionPublisher | None = None
        self._ph_continuation_publisher: PHContinuationPublisher | None = None
        self._presence_publisher: PresencePublisher | None = None
        self._dwell_publisher: DwellPublisher | None = None
        # World tracker
        self._world_tracker: WorldTracker | None = None
        self._ph_repo: PHRepositoryProtocol | None = None
        self._obs_repo: WorldObservationRepositoryProtocol | None = None
        self._frame_fetcher: FrameImageFetcher | None = None
        self._reid_embedder: ReidEmbedderProtocol | None = None
        self._trajectory_writer: TrajectoryWriter | None = None
        self._keyframe_sampler: KeyframeSampler | None = None
        self._scene_publisher: SceneSamplesPublisher | None = None
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._frame_tasks: set[asyncio.Task[None]] = set()
        self._frame_semaphore: asyncio.Semaphore | None = None
        self._camera_locks: dict[str, asyncio.Lock] = {}
        # Track previously active PH ids for ClosePHStage.
        self._prev_active_ph_ids: set[str] = set()
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
        # Bbox annotation repository for per-keyframe bbox persistence.
        self._bbox_repo: BboxAnnotationRepository | None = None
        # Phase 7: per-PH trail deque (last 12 normalised foot-points).
        _trail_maxlen = 12
        self._trail_by_ph: dict[str, deque[tuple[float, float]]] = {}
        self._TRAIL_MAXLEN = _trail_maxlen
        # Frame batcher (optional; None when batch_window_s=0).
        self._batcher: FrameBatcher | None = None
        self._stage_runner: StageRunner | None = None
        self._post_detect_runner: StageRunner | None = None
        self._pre_world_runner: StageRunner | None = None
        self._post_world_runner: StageRunner | None = None
        self._world_tracking_stage: WorldTrackingStage | None = None
        self._fetch_stage: FetchStage | None = None
        self._detect_stage: DetectStage | None = None
        self._camera_room_map = CameraRoomMap()
        self._room_polygon_map = RoomPolygonMap()

    @property
    def is_running(self) -> bool:
        return self._running

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
        gallery = deps.gallery_repo or InMemoryGalleryRepository()
        self._gallery_repo = gallery
        self._gallery_cache = GalleryCache(gallery)
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

        # World-coordinate person tracker (replaces per-camera + cross-camera).
        self._ph_repo = deps.ph_repo or InMemoryPHRepository()
        self._obs_repo = deps.obs_repo or InMemoryWorldObservationRepository()

        # ---- Identity resolution ----
        self._floor_projector = FloorProjector(calibration_state)
        self._spatial_projection = SpatialProjectionService(calibration_state)

        self._identity_resolver = IdentityResolver(
            gallery_repo=gallery,
            identities=self._config.known_identities,
            config=self._config.resolver,
            gallery_cache=self._gallery_cache,
        )

        self._revision_publisher = RevisionPublisher(
            redis_url=self._config.transport.redis_url,
            stream=self._config.transport.revisions_stream,
        )
        await self._revision_publisher.connect()
        self._ph_continuation_publisher = PHContinuationPublisher(
            redis_url=self._config.transport.redis_url,
        )
        await self._ph_continuation_publisher.connect()

        # Presence and dwell publishers for CC load decoupling.
        self._presence_publisher = PresencePublisher(
            redis_url=self._config.transport.redis_url,
            stream=self._config.transport.presence_stream,
        )
        await self._presence_publisher.connect()
        self._dwell_publisher = DwellPublisher(
            redis_url=self._config.transport.redis_url,
            stream=self._config.transport.dwell_stream,
        )
        await self._dwell_publisher.connect()

        self._world_tracker = WorldTracker(
            ph_repo=self._ph_repo,
            obs_repo=self._obs_repo,
            config=self._config.world_tracker,
            continuation_publisher=self._ph_continuation_publisher,
            identity_resolver=self._identity_resolver,
            gallery_repo=self._gallery_repo,
            topology_repo=deps.topology_repo if deps else None,
            copresence_repo=deps.copresence_repo if deps else None,
            overlap_groups=deps.overlap_groups if deps else None,
        )

        # ---- Trajectory writer + keyframe sampler ----
        # Use a single in-memory fallback so the signal worker can read
        # trajectories written by the writer.
        _traj_repo = deps.trajectory_repo or InMemoryTrajectoryRepository()
        self._trajectory_writer = TrajectoryWriter(repo=_traj_repo)

        keyframe_repo = deps.keyframe_repo or InMemoryKeyframeRepository(bbox_repo=self._bbox_repo)
        self._keyframe_sampler = KeyframeSampler(
            repo=keyframe_repo,
            config=self._config.sampler,
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
                batch_handler=(
                    self._handle_cross_camera_batch
                    if self._config.cross_camera_detector_batching and self._detector is not None
                    else None
                ),
            )
            logger.info(
                "Frame batcher enabled",
                batch_window_s=self._config.batch_window_s,
                max_batch_size=self._config.max_batch_size,
                cross_camera_detector_batching=self._config.cross_camera_detector_batching,
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
        fetch_stage = FetchStage(frame_fetcher=self._frame_fetcher)
        detect_stage = DetectStage(
            detector=self._detector,  # type: ignore[arg-type]
            iou_dedup_threshold=self._config.detection_iou_dedup_threshold,
        )
        world_tracking_stage = WorldTrackingStage(
            tracker=self._world_tracker,
            camera_room_map=self._camera_room_map,
            room_polygon_map=self._room_polygon_map,
            config=self._config.world_tracker,
        )

        # Named stage groups — replace fragile integer slices so adding
        # or reordering a stage cannot silently corrupt batched paths.
        io_stages = [fetch_stage, detect_stage]

        pre_world_stages = [
            PrivacyStage(),
            SpatialProjectionStage(projection_service=self._spatial_projection),
            InferenceStage(
                reid_embedder=self._reid_embedder,
                pose_estimator=self._pose_estimator,
                pose_enabled=self._config.pose_enabled,
            ),
            FaceIdentityStage(
                face_id_client=self._face_id_client,
                gallery_repo=self._gallery_repo,
                face_id_cooldown_s=self._config.face_id.cooldown_s,
                face_id_min_confidence=self._config.face_id.min_confidence,
                face_id_camera_configs=self._config.face_id.camera_configs,
                last_face_id_by_tracklet=self._last_face_id_by_tracklet,
            ),
        ]

        post_world_stages = [
            DetectionBackfillStage(),
            ClosePHStage(
                trajectory_writer=self._trajectory_writer,
                motion_energy_tracker=self._motion_energy_tracker,
                posture_tracker=self._posture_tracker,
                prev_active_ph_ids=self._prev_active_ph_ids,
                presence_publisher=self._presence_publisher,
                dwell_publisher=self._dwell_publisher,
            ),
            PostureStage(
                camera_room_map=self._camera_room_map,
                posture_strategy=self._posture_strategy,
            ),
            TrajectoryStage(
                trajectory_writer=self._trajectory_writer,
                floor_projector=self._floor_projector,
                motion_energy_tracker=self._motion_energy_tracker,
                posture_tracker=self._posture_tracker,
            ),
            KeyframeStage(
                keyframe_sampler=self._keyframe_sampler,
                scene_publisher=self._scene_publisher,
                min_keyframe_detection_confidence=self._config.min_keyframe_detection_confidence,
            ),
            RevisionsStage(
                revision_publisher=self._revision_publisher,
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
                camera_room_map=self._camera_room_map,
                live_publish_max_hz=self._config.live_publish_max_hz,
            ),
        ]

        stages = io_stages + pre_world_stages + [world_tracking_stage] + post_world_stages

        self._fetch_stage = fetch_stage
        self._detect_stage = detect_stage
        self._world_tracking_stage = world_tracking_stage
        self._stage_runner = StageRunner(stages)
        self._post_detect_runner = StageRunner(
            [*pre_world_stages, world_tracking_stage, *post_world_stages]
        )
        self._pre_world_runner = StageRunner(pre_world_stages)
        self._post_world_runner = StageRunner(post_world_stages)

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
                    "Set pipeline.allow_skeleton=true in settings.yaml for skeleton-only tests."
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

        if self._ph_continuation_publisher:
            await self._ph_continuation_publisher.disconnect()

        if self._presence_publisher:
            await self._presence_publisher.disconnect()

        if self._dwell_publisher:
            await self._dwell_publisher.disconnect()

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

    async def _handle_cross_camera_batch(self, frames: list[FrameReady]) -> None:
        """Fetch and detect a mixed-camera frame batch with one Triton request.

        Per-camera locks are still held while downstream stages run, preserving
        frame order for tracker state. The detector call is shared across all
        frames in the flush, so a static batch-8 detector can be filled with
        real camera frames instead of duplicate padding.
        """
        assert self._transport is not None
        if not frames:
            return
        if (
            self._fetch_stage is None
            or self._detect_stage is None
            or self._pre_world_runner is None
            or self._world_tracking_stage is None
            or self._post_world_runner is None
        ):
            for camera_id, camera_frames in _group_frames_by_camera(frames).items():
                await self._handle_batch(camera_id, camera_frames)
            return
        if self._frame_semaphore is None:
            self._frame_semaphore = asyncio.Semaphore(max(1, self._config.max_concurrent_frames))

        frames_by_camera = _group_frames_by_camera(frames)
        async with self._frame_semaphore, contextlib.AsyncExitStack() as stack:
            for camera_id in sorted(frames_by_camera):
                lock = self._camera_locks.setdefault(camera_id, asyncio.Lock())
                await stack.enter_async_context(lock)

            contexts: list[FrameContext] = []
            for camera_frames in frames_by_camera.values():
                for frame in camera_frames:
                    if self._is_stale(frame):
                        await self._transport.ack_frame(frame)
                        continue
                    contexts.append(self._init_context(frame))

            if not contexts:
                return

            fetch_results = await asyncio.gather(
                *(self._fetch_stage.run(ctx) for ctx in contexts),
                return_exceptions=True,
            )
            fetched_contexts: list[FrameContext] = []
            for ctx, result in zip(contexts, fetch_results, strict=True):
                if isinstance(result, Exception):
                    logger.warning(
                        "Frame fetch failed in detector batch",
                        camera_id=ctx.frame.camera_id,
                        frame_index=ctx.frame.frame_index,
                        error=str(result),
                    )
                    await self._transport.publish_response(
                        ctx.frame,
                        success=False,
                        error_code="fetch_error",
                    )
                else:
                    fetched_contexts.append(ctx)

            if not fetched_contexts:
                return

            try:
                await self._detect_stage.run_batch(fetched_contexts)
            except Exception:
                logger.exception(
                    "Batched detector inference failed",
                    count=len(fetched_contexts),
                )
                for ctx in fetched_contexts:
                    await self._transport.publish_response(
                        ctx.frame,
                        success=False,
                        error_code="detector_error",
                    )
                return

            await self._process_cross_camera_post_detect_batch(fetched_contexts)

    async def _process_cross_camera_post_detect_batch(self, contexts: list[FrameContext]) -> None:
        """Run post-detect stages while preserving cross-camera world tracking.

        Detector batching may include multiple frames per camera. Process one
        ordered round at a time: the earliest remaining frame from each camera
        reaches WorldTrackingStage together, so pre-association dedup can see
        overlapping camera observations in the same tracker step.
        """
        assert self._transport is not None
        assert self._pre_world_runner is not None
        assert self._world_tracking_stage is not None
        assert self._post_world_runner is not None

        by_camera: dict[str, list[FrameContext]] = {}
        for ctx in contexts:
            by_camera.setdefault(ctx.frame.camera_id, []).append(ctx)
        for camera_contexts in by_camera.values():
            camera_contexts.sort(key=lambda ctx: ctx.frame.frame_index)

        while any(by_camera.values()):
            round_contexts: list[FrameContext] = []
            for camera_id in sorted(by_camera):
                camera_contexts = by_camera[camera_id]
                if camera_contexts:
                    round_contexts.append(camera_contexts.pop(0))

            ready_contexts: list[FrameContext] = []
            for ctx in round_contexts:
                try:
                    await self._ensure_camera_row(ctx.frame.camera_id)
                    await self._pre_world_runner.run(ctx)
                    ready_contexts.append(ctx)
                except Exception:
                    logger.exception(
                        "Frame pre-world processing failed",
                        camera_id=ctx.frame.camera_id,
                        frame_index=ctx.frame.frame_index,
                    )
                    await self._transport.publish_response(
                        ctx.frame,
                        success=False,
                        error_code="processing_error",
                    )

            if not ready_contexts:
                continue

            try:
                await self._world_tracking_stage.run_many(ready_contexts)
            except Exception:
                logger.exception(
                    "Cross-camera world tracking failed",
                    count=len(ready_contexts),
                )
                for ctx in ready_contexts:
                    await self._transport.publish_response(
                        ctx.frame,
                        success=False,
                        error_code="processing_error",
                    )
                continue

            await asyncio.gather(
                *(self._process_post_world_context(ctx) for ctx in ready_contexts),
                return_exceptions=True,
            )

    async def _process_post_world_context(self, ctx: FrameContext) -> None:
        """Run camera-local stages after a shared world-tracking round."""
        assert self._transport is not None
        assert self._post_world_runner is not None
        start = time.monotonic()
        try:
            await self._post_world_runner.run(ctx)
            await self._transport.ack_frame(ctx.frame)
            latency_us = int((time.monotonic() - start) * 1e6)
            logger.debug(
                "Frame processed",
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
                latency_us=latency_us,
            )
        except Exception:
            logger.exception(
                "Frame post-world processing failed",
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
            )
            await self._transport.publish_response(
                ctx.frame,
                success=False,
                error_code="processing_error",
            )

    async def _process_post_detect_batch(self, contexts: list[FrameContext]) -> None:
        """Run stages after detection for one camera ordered context group."""
        assert self._transport is not None
        assert self._post_detect_runner is not None
        for ctx in contexts:
            start = time.monotonic()
            try:
                await self._ensure_camera_row(ctx.frame.camera_id)
                await self._post_detect_runner.run(ctx)
                await self._transport.ack_frame(ctx.frame)
                latency_us = int((time.monotonic() - start) * 1e6)
                logger.debug(
                    "Frame processed",
                    camera_id=ctx.frame.camera_id,
                    frame_index=ctx.frame.frame_index,
                    latency_us=latency_us,
                )
            except Exception:
                logger.exception(
                    "Frame processing failed",
                    camera_id=ctx.frame.camera_id,
                    frame_index=ctx.frame.frame_index,
                )
                await self._transport.publish_response(
                    ctx.frame,
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

    def set_camera_room_map(self, camera_room_map: CameraRoomMap) -> None:
        """Inject the live CameraRoomMap into pipeline stages.

        Called at startup after the CCConfigSyncService is created.
        Stages that need room attribution read from this map.
        """
        self._camera_room_map = camera_room_map
        if self._stage_runner is None:
            return
        for stage in self._stage_runner._stages:
            if hasattr(stage, "_camera_room_map"):
                stage._camera_room_map = camera_room_map

    def set_room_polygon_map(self, room_polygon_map: RoomPolygonMap) -> None:
        """Inject the live room polygon map into WorldTrackingStage."""
        self._room_polygon_map = room_polygon_map
        if self._world_tracking_stage is not None:
            self._world_tracking_stage._room_polygon_map = room_polygon_map

    def set_transit_config(
        self,
        transit_detector: object,
        transit_zone_map: TransitZoneMap,
        room_transition_publisher: object,
    ) -> None:
        """Inject transit zone dependencies into WorldTrackingStage.

        Called at startup after the live transit-zone map is created.
        """
        if self._stage_runner is not None:
            for stage in self._stage_runner._stages:
                if stage.name == "world_tracking":
                    stage._transit_detector = transit_detector  # type: ignore[attr-defined]
                    stage._transit_zone_map = transit_zone_map  # type: ignore[attr-defined]
                    stage._room_transition_publisher = room_transition_publisher  # type: ignore[attr-defined]
                    break

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

        assert self._transport is not None
        room_name = await camera_room_name(self._camera_room_map, frame.camera_id)
        await self._transport.publish_event(
            camera_id=frame.camera_id,
            event_time=event_time,
            frame_index=frame.frame_index,
            minio_key=frame.minio_key,
            room_name=room_name if room_name is not None else "",
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
