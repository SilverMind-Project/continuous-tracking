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
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ..calibration.state import calibration_state
from ..domain import (
    BoundingBox,
    CameraConfig,
    Detection,
    FaceAnchor,
    FloorPoint,
    FrameRef,
    GlobalTrack,
    Identity,
    IdentityRevision,
    OverlapGroup,
    PostureType,
    ResolveOutcome,
    TaggedKeyframe,
    TrackingEvent,
)
from ..inference.detector import PersonDetector
from ..inference.face_id_client import FaceIdentificationClient
from ..inference.pose import PoseEstimator
from ..inference.schemas import DetectionBox, Embedding, PoseResult
from ..observability import metrics as _metrics
from ..pipeline.batcher import FrameBatcher
from ..pipeline.privacy import PrivacyZoneFilter
from ..sampling.keyframe_sampler import KeyframeSampler, SamplerConfig
from ..services.identity_rewriter import (
    IdentityRewriter,
    InMemoryIdentityRewriter,
)
from ..storage.base import (
    BehaviorBaselineRepository,
    DementiaSignalRepository,
    GalleryRepository,
    GlobalTrackRepository,
    InMemoryBehaviorBaselineRepository,
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
from ..tracking.identity_committer import IdentityCommitter
from ..tracking.identity_resolver import IdentityResolver, ResolverConfig
from ..tracking.tracker import PerCameraTrackers, TrackerConfig
from ..tracking.tracklet_manager import TrackletConfig, TrackletManager
from ..trajectory.dementia_signals import DementiaSignalWorker, SignalConfig
from ..trajectory.motion_energy import MotionEnergyTracker
from ..trajectory.posture import classify_posture
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


def _iou_dedup_detections(
    boxes: list[DetectionBox],
    iou_threshold: float,
) -> list[DetectionBox]:
    """Greedy IoU dedup: keep the highest-confidence box from overlapping clusters.

    Processes boxes in descending confidence order. A box is suppressed if its
    IoU with any already-kept box exceeds *iou_threshold*.  O(N^2) but N is
    always small (YOLO outputs ≤ 300 post-NMS boxes of which only a handful
    pass the score threshold in a typical room-camera scene).
    """
    if len(boxes) <= 1:
        return list(boxes)

    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[DetectionBox] = []
    suppressed_count = 0
    for box in sorted_boxes:
        b_coords = [box.x1, box.y1, box.x2, box.y2]
        if any(_bbox_iou(b_coords, [k.x1, k.y1, k.x2, k.y2]) > iou_threshold for k in kept):
            suppressed_count += 1
        else:
            kept.append(box)
    if suppressed_count > 0:
        _metrics.metrics.detections_suppressed_total.labels(stage="iou_dedup").inc(suppressed_count)
    return kept


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
    # Dementia signal thresholds (see SignalConfig for rationale).
    signal_stillness_threshold_minutes: int = 60
    signal_stillness_emergency_minutes: int = 120
    signal_stillness_motion_floor: float = 0.02
    signal_pacing_room_threshold: int = 8
    signal_pacing_window_minutes: int = 30
    signal_nighttime_transition_threshold: int = 3
    signal_absence_threshold_minutes: int = 60
    signal_bathroom_absolute_threshold_seconds: int = 2700
    # Face identification via person-identification-service (ArcFace).
    face_id_url: str = ""
    face_id_cooldown_s: float = 5.0
    face_id_timeout_s: float = 2.0
    # Allow running without a detector (skeleton mode). Off by default; enable only in tests.
    allow_skeleton: bool = False
    face_id_min_confidence: float = 0.5
    face_id_enabled: bool = True
    # Per-camera overrides: camera_id -> enabled flag and optional higher threshold.
    # Top-down cameras should set enabled=false; face-level cameras with
    # difficult angles can raise min_confidence above the global default.
    face_id_camera_configs: dict[str, FaceIdCameraConfig] = field(default_factory=dict)
    # Pose estimation (RTMPose) — enabled by default; set False to disable.
    pose_enabled: bool = True

    # --- Phase 1: noise reduction ---

    # Post-decode IoU suppression threshold. Detections whose bboxes overlap
    # an already-kept detection by more than this IoU are dropped before
    # the tracker sees them. Belt-and-braces since the ONNX model bakes NMS.
    detection_iou_dedup_threshold: float = 0.55

    # Per-camera tracker dedup IoU threshold (see TrackerConfig.dedup_iou_threshold).
    tracker_dedup_iou_threshold: float = 0.6

    # Stability gate: tracklets must survive this many frames before being
    # exposed to downstream pipeline stages and publication.
    tracker_min_frames_to_publish: int = 3

    # IdentityCommitter — buffered windowed commit (see IdentityCommitter).
    identity_commit_window_s: float = 3.0
    identity_high_confidence_face_threshold: float = 0.80
    # Feature flag: off by default; enable after one week soak.
    identity_committer_enabled: bool = False
    # Retroactive cross-table rewrite on face-confirmed identity commits.
    # Enabled by default: when any face anchor triggers a revision, the
    # IdentityRewriter updates trajectory, dwell, and signal rows so that
    # history reflects the now-known identity.  Set False only to disable
    # rewrites without disabling the identity_committer_enabled path.
    identity_rewrite_on_face_commit: bool = True
    # Frame batching — buffer frames for a configurable time window,
    # group by camera_id, then flush each camera's batch sequentially
    # while running *different* cameras concurrently.
    # Set batch_window_s=0 to disable batching (default behaviour).
    batch_window_s: float = 0.0
    max_batch_size: int = 4


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
        # Overlap groups fetched from CC at startup; preserved across adjacency reloads.
        self._overlap_groups: list[OverlapGroup] = []
        self._frame_fetcher: FrameImageFetcher | None = None
        self._reid_embedder: ReidEmbedderProtocol | None = None
        # M6
        self._trajectory_writer: TrajectoryWriter | None = None
        self._identity_committer: IdentityCommitter | None = None
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
        # Pose estimation (RTMPose) + motion energy tracking.
        self._pose_estimator: PoseEstimator | None = None
        self._motion_energy_tracker: MotionEnergyTracker | None = None
        # Phase 1: cross-table identity rewriter (orchestrator side).
        self._identity_rewriter: IdentityRewriter | None = None
        # Phase 7: per-tracklet trail deque (last 12 normalised foot-points).
        _trail_maxlen = 12
        self._trail_by_tracklet: dict[str, deque[tuple[float, float]]] = {}
        self._TRAIL_MAXLEN = _trail_maxlen
        # Frame batcher (optional; None when batch_window_s=0).
        self._batcher: FrameBatcher | None = None

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
        # Pose estimator (RTMPose) — optional; if None, posture defaults to "unknown".
        pose_estimator: PoseEstimator | None = None,
        # Identity rewriter — rewrites trajectory/dwell/signal rows on revision.
        # Defaults to InMemory no-op; inject PostgresIdentityRewriter in production.
        identity_rewriter: IdentityRewriter | None = None,
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
            pose_estimator: Triton-backed RTMPose estimator. If None, posture defaults to "unknown".
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

        # Pose estimation
        self._pose_estimator = pose_estimator
        self._motion_energy_tracker = MotionEnergyTracker() if pose_estimator is not None else None

        # Identity rewriter (orchestrator side)
        self._identity_rewriter = identity_rewriter or InMemoryIdentityRewriter()

        # Tracklet manager
        tracker = PerCameraTrackers(
            TrackerConfig(
                dedup_iou_threshold=self._config.tracker_dedup_iou_threshold,
            )
        )

        tracklet_config = TrackletConfig(
            min_hit_ratio=self._config.tracklet.min_hit_ratio,
            close_grace_frames=self._config.tracklet.close_grace_frames,
            gallery_min_quality=self._config.tracklet.gallery_min_quality,
            gallery_max_per_tracklet=self._config.tracklet.gallery_max_per_tracklet,
            min_detection_confidence=self._config.tracklet.min_detection_confidence,
            enabled=self._config.tracklet.enabled,
            min_frames_to_publish=self._config.tracker_min_frames_to_publish,
        )

        self._tracklet_manager = TrackletManager(
            repo=self._repo,
            gallery=gallery,
            config=tracklet_config,
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

        # Phase 5: IdentityCommitter buffers per-frame posterior evidence
        # and emits one commit decision per GT per commit_window_s.
        self._identity_committer = IdentityCommitter(
            commit_window_s=self._config.identity_commit_window_s,
            high_confidence_face_threshold=self._config.identity_high_confidence_face_threshold,
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
            traj_repo = trajectory_repo or InMemoryTrajectoryRepository()
            # Build a baseline repository from the same trajectory source.
            baseline_repo: BehaviorBaselineRepository = InMemoryBehaviorBaselineRepository()
            self._signal_worker = DementiaSignalWorker(
                trajectory_repo=traj_repo,
                signal_repo=self._signal_repo,
                cfg=SignalConfig(
                    tz_name=self._config.timezone,
                    stillness_threshold_minutes=self._config.signal_stillness_threshold_minutes,
                    stillness_emergency_minutes=self._config.signal_stillness_emergency_minutes,
                    stillness_motion_floor=self._config.signal_stillness_motion_floor,
                    pacing_room_threshold=self._config.signal_pacing_room_threshold,
                    pacing_window_minutes=self._config.signal_pacing_window_minutes,
                    nighttime_transition_threshold=self._config.signal_nighttime_transition_threshold,
                    absence_threshold_minutes=self._config.signal_absence_threshold_minutes,
                    bathroom_absolute_threshold_seconds=self._config.signal_bathroom_absolute_threshold_seconds,
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
        new_adjacency.set_overlap_groups(self._overlap_groups)
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

    def set_overlap_groups(self, groups: list[OverlapGroup]) -> None:
        """Apply overlap group data from CC.

        Stores the groups for future adjacency reloads and immediately
        updates the current adjacency graph if it has been built.
        """
        self._overlap_groups = groups
        if self._adjacency is not None:
            self._adjacency.set_overlap_groups(groups)

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
                _metrics.metrics.frames_dropped_stale_total.labels(camera_id=frame.camera_id).inc()
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
        # Guard against resolution mismatch between the MinIO-stored image
        # and the FrameReady-reported dimensions.  If they diverge (e.g.
        # due to EXIF rotation, resized upload, or thumbnail storage), the
        # bbox→pixel and pose→pixel transforms would be offset, producing
        # misaligned overlays in the live view.
        img_h, img_w = image.shape[:2]
        effective_width = frame.width
        effective_height = frame.height
        if img_w != frame.width or img_h != frame.height:
            logger.warning(
                "frame_dimension_mismatch",
                camera_id=frame.camera_id,
                frame_index=frame.frame_index,
                minio_shape=f"{img_h}x{img_w}",
                reported_shape=f"{frame.height}x{frame.width}",
            )
            effective_width = img_w
            effective_height = img_h

        logger.debug(
            "detections_raw",
            camera_id=frame.camera_id,
            frame_index=frame.frame_index,
            count=len(detections),
            image_shape=f"{img_h}x{img_w}",
        )

        # Step 2a: Post-decode IoU dedup — suppresses any near-duplicate
        # detections that survived the model's baked NMS. Keeps the highest-
        # confidence box from each overlapping cluster.
        if detections and self._config.detection_iou_dedup_threshold < 1.0:
            detections = _iou_dedup_detections(
                detections,
                self._config.detection_iou_dedup_threshold,
            )

        # Privacy enforcement: drop detections + apply blur/mask to frame.
        privacy_filter = PrivacyZoneFilter.from_state(
            calibration_state,
            frame.camera_id,
            frame_width=effective_width,
            frame_height=effective_height,
        )
        if privacy_filter.is_active():
            # Apply blur/mask policies to the frame in place (affects crops
            # and the uploaded keyframe).
            image = privacy_filter.apply_blur_mask(image)

        # Filter detections in drop_detection zones.
        if detections and privacy_filter.is_active():
            kept: list[DetectionBox] = []
            dropped_count = 0
            for det in detections:
                foot_x = (det.x1 + det.x2) / 2.0
                foot_y = det.y2
                if privacy_filter.should_drop((foot_x, foot_y)):
                    dropped_count += 1
                    _metrics.metrics.privacy_detections_dropped_total.labels(
                        camera_id=frame.camera_id,
                    ).inc()
                else:
                    kept.append(det)
            if dropped_count > 0:
                logger.debug(
                    "privacy_detections_filtered",
                    camera_id=frame.camera_id,
                    dropped=dropped_count,
                    kept=len(kept),
                )
            detections = kept

        domain_detections: list[Detection] = []
        det_posture: dict[str, PostureType] = {}
        det_pose_result: dict[str, PoseResult] = {}
        embeddings: list[Embedding] = []
        crops: list[npt.NDArray[np.uint8]] = []

        if detections:
            crops = [_crop_detection(image, det) for det in detections]

            # ReID and pose both operate on the same crops — run in parallel.
            async def _do_reid() -> list[Embedding]:
                if self._reid_embedder is not None:
                    return await self._reid_embedder.embed_batch(crops)
                return []

            async def _do_pose() -> list[PoseResult | None]:
                if self._pose_estimator is not None and self._config.pose_enabled:
                    return await self._run_pose(crops, detections)
                return []

            embeddings, pose_results = await asyncio.gather(
                _do_reid(), _do_pose(),
            )

            for det_idx, det in enumerate(detections):
                bbox = BoundingBox(
                    x_min=int(det.x1 * effective_width),
                    y_min=int(det.y1 * effective_height),
                    x_max=int(det.x2 * effective_width),
                    y_max=int(det.y2 * effective_height),
                )
                emb = embeddings[det_idx] if det_idx < len(embeddings) else None

                # Classify posture for this detection.
                posture: PostureType = "unknown"
                pose_result = pose_results[det_idx] if det_idx < len(pose_results) else None
                if pose_result is not None:
                    posture = classify_posture(pose_result, bbox)

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
                det_posture[domain_det.detection_id] = posture
                if pose_result is not None:
                    det_pose_result[domain_det.detection_id] = pose_result

        # Step 3: Per-camera tracking — run even with 0 detections so the
        # BoT-SORT Kalman filter ages out lost tracklets instead of keeping
        # them alive indefinitely when nobody is in frame.
        local_tracks = self._tracker.update(
            camera_id=frame.camera_id,
            detections=domain_detections,
            embeddings=embeddings or None,
            frame_index=frame.frame_index,
        )

        # Emit tracker dedup metric
        dedup_dropped = self._tracker.get_dedup_dropped(frame.camera_id)
        if dedup_dropped > 0:
            _metrics.metrics.tracklets_dedup_dropped_total.labels(camera_id=frame.camera_id).inc(
                dedup_dropped
            )

        # Step 4: Tracklet management — run even with 0 detections so
        # tracklets that have no corresponding detection are marked lost.
        camera_config = CameraConfig(camera_id=frame.camera_id)
        await self._tracklet_manager.step(
            camera=camera_config,
            local_tracks=local_tracks,
            detections=domain_detections,
            embeddings=embeddings,
            event_time=event_time,
            frame_index=frame.frame_index,
        )

        # Step 4b: Face identification (rate-limited call to person-identification-service).
        # Uses person crops (already extracted for ReID at Step 3) at native
        # resolution to give the face detector the best chance at small faces.
        face_anchors: list[FaceAnchor] = []
        now = datetime.now(UTC)
        if (
            self._face_id_client is not None
            and domain_detections
            and crops
            and self._should_call_face_id(frame.camera_id, now)
        ):
            face_anchors = await self._identify_faces_from_crops(
                crops=crops,
                crop_detections=domain_detections,
                frame_width=effective_width,
                frame_height=effective_height,
                camera_id=frame.camera_id,
            )

        # Ensure face-anchor identities exist in the identities table so that
        # downstream FK references (trajectory points, dwells, signals) resolve.
        if face_anchors and self._gallery_repo is not None:
            seen: set[str] = set()
            for fa in face_anchors:
                if fa.person_id and fa.person_id != "unknown" and fa.person_id not in seen:
                    seen.add(fa.person_id)
                    await self._gallery_repo.upsert_identity(
                        Identity(
                            identity_id=fa.person_id,
                            display_name=fa.person_id,
                            enrolled_at=now,
                        )
                    )

        # ---- M5: Cross-camera association ----
        active_tracklets = (
            self._tracklet_manager.get_active_tracklets() if self._tracklet_manager else []
        )

        # Stability gate metric: count tracklets held below the publish threshold.
        if self._tracklet_manager is not None:
            for cam_id, held_count in self._tracklet_manager.get_held_count_by_camera().items():
                _metrics.metrics.tracklets_held_below_stability_gate.labels(camera_id=cam_id).set(
                    held_count
                )
        active_global_tracks: list[GlobalTrack] = []
        new_revisions: list[IdentityRevision] = []
        outcome: ResolveOutcome = ResolveOutcome()

        if active_tracklets:
            assert self._cross_camera is not None
            assert self._identity_resolver is not None

            # Step 5: Cross-camera association
            active_global_tracks = await self._cross_camera.associate(
                active_tracklets,
                captured_at=event_time,
            )

            # Step 6: Identity resolution
            outcome = await self._identity_resolver.resolve(
                global_tracks=active_global_tracks,
                new_face_anchors=face_anchors,
                captured_at=event_time,
            )

            # Apply decisions: update GlobalTrack identity assignments.
            if self._config.identity_committer_enabled and self._identity_committer is not None:
                # Buffered path: ingest per-frame evidence, flush committed decisions.
                for decision in outcome.decisions:
                    _top_id, top_conf = decision.posterior.top_identity()
                    self._identity_committer.ingest(
                        global_track_id=decision.global_track_id,
                        identity_id=decision.identity_id,
                        confidence=top_conf,
                        reason=decision.reason,
                    )
                # High-confidence face fast-path: commit immediately and rewrite history.
                for fa in face_anchors:
                    gt = next(
                        (
                            g
                            for g in active_global_tracks
                            if any(tid == fa.tracklet_id for tid in g.tracklet_ids)
                            if self._tracklet_manager is not None
                        ),
                        None,
                    )
                    if gt is not None:
                        immediate = self._identity_committer.check_high_confidence_face(
                            gt.global_track_id,
                            fa.person_id,
                            fa.confidence,
                            first_seen_at=gt.started_at,
                        )
                        if immediate and self._global_track_repo:
                            await self._global_track_repo.assign_identity(
                                global_track_id=immediate.global_track_id,
                                identity_id=immediate.identity_id,
                            )
                # Flush buffer: emit committed decisions.
                flushed = self._identity_committer.flush()
                if self._global_track_repo:
                    for commit in flushed:
                        if commit.identity_id is not None:
                            await self._global_track_repo.assign_identity(
                                global_track_id=commit.global_track_id,
                                identity_id=commit.identity_id,
                            )
            elif self._global_track_repo:
                # Direct per-frame path (legacy, committer_enabled=False).
                # Only write when the identity actually changes; maintenance
                # carry-forwards set revises_previous=False and must not call
                # assign_identity(None) which would clear a valid assignment.
                for decision in outcome.decisions:
                    if decision.revises_previous and decision.identity_id is not None:
                        await self._global_track_repo.assign_identity(
                            global_track_id=decision.global_track_id,
                            identity_id=decision.identity_id,
                        )

            # Collect revisions to emit.
            new_revisions = list(outcome.revisions)

            # Backfill gallery identity: once the identity resolver commits an
            # identity for a GlobalTrack, stamp that identity onto all gallery
            # entries for the tracklets in that GT.  This creates a virtuous
            # cycle where future ReID gallery searches (for this person on
            # other cameras) find identity-tagged entries and produce identity
            # evidence — instead of mapping to "UNKNOWN".
            if self._gallery_repo is not None and active_global_tracks:
                for gt in active_global_tracks:
                    # Only backfill when the GT has a committed identity on
                    # this frame (either from the direct path or the committer).
                    committed_id = gt.current_identity_id
                    if not committed_id:
                        continue
                    tracklet_ids = set(gt.tracklet_ids)
                    if tracklet_ids:
                        await self._gallery_repo.update_identity_for_tracklets(
                            tracklet_ids=tracklet_ids,
                            identity_id=committed_id,
                        )

        # Step 5b: Close trajectory dwells for terminated global tracks.
        # Runs regardless of whether active_tracklets is empty — when it is
        # empty, every previously-active track is considered terminated.
        current_gt_ids = {gt.global_track_id for gt in active_global_tracks}
        terminated_gt_ids = self._prev_active_gt_ids - current_gt_ids
        if terminated_gt_ids:
            traj_close_time = datetime.now(UTC)
            for gt_id in terminated_gt_ids:
                logger.debug(
                    "Closing terminated global track",
                    global_track_id=gt_id,
                )
                if self._global_track_repo is not None:
                    await self._global_track_repo.close_global_track(gt_id)
                if self._trajectory_writer:
                    await self._trajectory_writer.close_track(gt_id, closed_at=traj_close_time)
                if self._motion_energy_tracker is not None:
                    self._motion_energy_tracker.evict_track(gt_id)
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
                # CR-12: Always write trajectory points, even when identity
                # is UNKNOWN.  The trajectory is keyed on global_track_id;
                # identity is a stamp, not a gate.  When identity later
                # resolves, the rewriter retroactively labels the points.
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

                # Resolve posture + motion energy for this global track
                # via detection_id → tracklet → global_track chain.
                gt_posture: PostureType = "unknown"
                gt_motion_energy: float | None = None
                if self._motion_energy_tracker is not None and gt_bbox_entry is not None:
                    # Find a detection_id belonging to this global_track_id.
                    for domain_det in domain_detections:
                        tid = (
                            self._tracklet_manager.get_tracklet_id_for_detection(
                                domain_det.detection_id
                            )
                            if self._tracklet_manager
                            else ""
                        )
                        if not tid:
                            continue
                        # Check if this tracklet belongs to the current gt decision.
                        gt_for_det = next(
                            (
                                gt.global_track_id
                                for gt in active_global_tracks
                                if tid in gt.tracklet_ids
                            ),
                            "",
                        )
                        if gt_for_det != decision.global_track_id:
                            continue
                        # Use the first matching detection's posture + pose.
                        gt_posture = det_posture.get(domain_det.detection_id, "unknown")
                        pose = det_pose_result.get(domain_det.detection_id)
                        if pose is not None:
                            bbox_diag = (gt_bbox_entry.width**2 + gt_bbox_entry.height**2) ** 0.5
                            me = self._motion_energy_tracker.update(
                                decision.global_track_id,
                                pose,
                                traj_time,
                                bbox_diag_px=bbox_diag,
                            )
                            gt_motion_energy = me.mean_keypoint_velocity_px_s
                        break

                await self._trajectory_writer.write(
                    identity_id=decision.identity_id,
                    global_track_id=decision.global_track_id,
                    room_name=room_name,
                    floor_point=floor_point,
                    captured_at=traj_time,
                    identity_confidence=top_prob,
                    posture=gt_posture,
                    motion_energy=gt_motion_energy,
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
                # Resolve the global track for this tracklet to pick up
                # the committed identity (if any).  The identity resolver
                # writes current_identity_id on the GlobalTrack before
                # keyframe sampling runs, so it is available here.
                gt_for_tracklet = next(
                    (gt for gt in active_global_tracks if tracklet.tracklet_id in gt.tracklet_ids),
                    None,
                )
                identity_id = (
                    gt_for_tracklet.current_identity_id if gt_for_tracklet is not None else ""
                )
                annotations: dict[str, object] = {
                    "tracklet_id": tracklet.tracklet_id,
                    "camera_id": tracklet.camera_id,
                    "identity_id": identity_id or "",
                }
                if tracklet.last_bbox is not None:
                    annotations["bbox"] = {
                        "x_min": tracklet.last_bbox.x_min,
                        "y_min": tracklet.last_bbox.y_min,
                        "x_max": tracklet.last_bbox.x_max,
                        "y_max": tracklet.last_bbox.y_max,
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

        # Step 9a: Retroactive cross-table rewrite. Run when the committer is
        # enabled OR face-commit rewrite is enabled and any revision changes
        # identity (including face-anchor-driven revisions from the direct path).
        if (
            (
                self._config.identity_committer_enabled
                or self._config.identity_rewrite_on_face_commit
            )
            and new_revisions
            and self._identity_rewriter is not None
        ):
            rewrite_time = datetime.now(UTC)
            gt_start_by_id = {gt.global_track_id: gt.started_at for gt in active_global_tracks}
            for rev in new_revisions:
                if rev.previous_identity_id is None or rev.new_identity_id is None:
                    continue
                applies_from = gt_start_by_id.get(rev.global_track_id, rewrite_time)
                await self._identity_rewriter.rewrite(
                    revision_id=str(rev.revision_id),
                    global_track_id=str(rev.global_track_id),
                    old_identity_id=str(rev.previous_identity_id),
                    new_identity_id=str(rev.new_identity_id),
                    applies_from=applies_from,
                    applies_to=rewrite_time,
                )

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

        # Step 9c (Phase 7): Update per-tracklet trail deques with the current
        # foot-point (bbox bottom-centre in normalised camera coords).
        frame_w = float(effective_width) if effective_width else 1.0
        frame_h = float(effective_height) if effective_height else 1.0
        for domain_det in domain_detections:
            if not domain_det.tracklet_id:
                continue
            foot_x = (domain_det.bbox.x_min + domain_det.bbox.x_max) / 2.0 / frame_w
            foot_y = domain_det.bbox.y_max / frame_h
            trail_dq = self._trail_by_tracklet.get(domain_det.tracklet_id)
            if trail_dq is None:
                trail_dq = deque(maxlen=self._TRAIL_MAXLEN)
                self._trail_by_tracklet[domain_det.tracklet_id] = trail_dq
            trail_dq.append((float(foot_x), float(foot_y)))

        # Expire trails for tracklets no longer active.
        active_tids = {d.tracklet_id for d in domain_detections if d.tracklet_id}
        stale_tids = set(self._trail_by_tracklet) - active_tids
        for tid in stale_tids:
            del self._trail_by_tracklet[tid]

        trail_by_tracklet_snapshot: dict[str, list[tuple[float, float]]] = {
            tid: list(dq) for tid, dq in self._trail_by_tracklet.items()
        }

        # Step 10: Publish tracking event with identity + room context so the
        # CC-side TrackingEventSubscriber can write PersonLocationState directly.
        identities: dict[str, tuple[str, float]] = {}
        evidence_by_gt: dict[str, tuple[float, float, bool]] = {}
        if active_tracklets and outcome.decisions:
            for decision in outcome.decisions:
                # Use the posterior's top identity even when the commit rule
                # hasn't fired yet.  If we gate on decision.identity_id alone
                # (which is None until a formal commit), the per-frame
                # TrackingEvent carries no IdentityRevision sub-messages, the
                # CC subscriber defaults identity_confidence to 0.0, the
                # LocationWriter skips the detection, no PersonLocationState
                # row is written, and the Current Presence tab falls through
                # to UNKNOWN for every person.
                top_id, top_prob = decision.posterior.top_identity()
                if top_id == "UNKNOWN" or top_prob <= 0.0:
                    continue
                identities[decision.global_track_id] = (top_id, top_prob)
                # Compute top-2 probability for the evidence chip.
                top_probs = sorted(decision.posterior.distribution.values(), reverse=True)
                top2_prob = top_probs[1] if len(top_probs) > 1 else 0.0
                evidence_by_gt[decision.global_track_id] = (top_prob, top2_prob, False)

        # Carry committed identities forward when the current-frame posterior
        # is below the identification threshold (brief occlusion, bad frame,
        # momentary YOLO miss).  A formal commit is a high-confidence,
        # multi-frame assignment; one ambiguous frame must not let it flicker
        # to UNKNOWN on the live view.  confidence=0.0 is an intentional
        # sentinel meaning "maintained from commit, not freshly evidenced";
        # LocationWriter accepts it and no downstream code gates on > 0.
        for gt in active_global_tracks:
            if gt.global_track_id not in identities and gt.current_identity_id:
                identities[gt.global_track_id] = (gt.current_identity_id, 0.0)

        assert self._transport is not None
        await self._transport.publish_event(
            camera_id=frame.camera_id,
            event_time=event_time,
            frame_index=frame.frame_index,
            detections=domain_detections if detections else None,
            minio_key=frame.minio_key,
            room_name=self._config.camera_room_map.get(frame.camera_id, ""),
            identities=identities or None,
            frame_width=effective_width,
            frame_height=effective_height,
            capture_time_unix_ns=frame.capture_time_unix_ns,
            pose_results=det_pose_result if det_pose_result else None,
            trail_by_tracklet=trail_by_tracklet_snapshot or None,
            evidence_by_gt=evidence_by_gt or None,
            det_posture=cast("dict[str, str] | None", det_posture) if det_posture else None,
        )

        if new_revisions:
            logger.info(
                "Identity revisions emitted",
                camera_id=frame.camera_id,
                frame_index=frame.frame_index,
                revision_count=len(new_revisions),
            )

    async def _run_pose(
        self,
        crops: list[npt.NDArray[np.uint8]],
        detections: list[DetectionBox],
    ) -> list[PoseResult | None]:
        """Run pose estimation on crops, skipping degenerate crops.

        A crop is degenerate when width < 16 or height < 32.
        Degenerate crops get ``None`` in the result list and are logged
        as ``pose_skipped``.
        """
        assert self._pose_estimator is not None
        valid_idxs: list[int] = []
        valid_crops: list[npt.NDArray[np.uint8]] = []
        results: list[PoseResult | None] = [None] * len(crops)
        for i, crop in enumerate(crops):
            h, w = crop.shape[:2]
            if w < 16 or h < 32:
                logger.debug(
                    "pose_skipped",
                    detection_index=i,
                    crop_width=w,
                    crop_height=h,
                )
                continue
            valid_idxs.append(i)
            valid_crops.append(crop)

        if valid_crops:
            batch_results = await self._pose_estimator.infer_batch(valid_crops)
            for vi, pr in zip(valid_idxs, batch_results, strict=True):
                results[vi] = pr
                visible = sum(1 for kp in pr.keypoints if kp.score > 0.2)
                logger.debug(
                    "pose_result",
                    detection_index=vi,
                    visible_keypoints=visible,
                    min_score=round(min(kp.score for kp in pr.keypoints), 3),
                    max_score=round(max(kp.score for kp in pr.keypoints), 3),
                )

        return results

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

    async def _identify_faces_from_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_detections: list[Detection],
        frame_width: int,
        frame_height: int,
        camera_id: str,
    ) -> list[FaceAnchor]:
        """Call person-id-service on person crops and build FaceAnchors.

        Sends each person crop at native resolution (no downscaling) to
        give the face detector the best chance at small/distant faces.
        The face bboxes are returned in frame-normalised [0, 1] space
        by the client, so association with YOLO detections is already
        handled — we just map detection → tracklet_id.

        If no face_id_client is configured, or the call fails, an empty
        list is returned (graceful degradation).
        """
        if self._face_id_client is None or not crops:
            return []

        cam_cfg = self._get_face_id_config(camera_id)
        if not cam_cfg.enabled:
            return []

        # Build crop_bboxes_norm from detections (frame-normalised [0, 1]).
        # Crop i corresponds to crop_detections[i].
        crop_bboxes_norm: list[tuple[float, float, float, float]] = []
        for det in crop_detections:
            crop_bboxes_norm.append(
                (
                    det.bbox.x_min / frame_width,
                    det.bbox.y_min / frame_height,
                    det.bbox.x_max / frame_width,
                    det.bbox.y_max / frame_height,
                )
            )

        try:
            crop_face_results = await self._face_id_client.identify_crops(crops, crop_bboxes_norm)
        except Exception:
            # Connection-level failure (service down, timeout, etc.).  Update the
            # cooldown timestamp so we don't hammer a dead service every frame.
            self._last_face_id_call[camera_id] = datetime.now(UTC)
            logger.warning(
                "face_id_service_error",
                camera_id=camera_id,
                crop_count=len(crops),
            )
            return []

        if not crop_face_results:
            self._last_face_id_call[camera_id] = datetime.now(UTC)
            return []

        # Effective per-camera confidence threshold.
        min_conf = (
            cam_cfg.min_confidence
            if cam_cfg.min_confidence is not None
            else self._config.face_id_min_confidence
        )

        face_anchors: list[FaceAnchor] = []
        for crop_idx, face_results in crop_face_results:
            det = crop_detections[crop_idx]

            for face in face_results:
                if face.person_id == "unknown":
                    continue
                if face.confidence < min_conf:
                    continue

                tracklet_id = ""
                if self._tracklet_manager is not None:
                    tracklet_id = self._tracklet_manager.get_tracklet_id_for_detection(
                        det.detection_id
                    )

                if not tracklet_id:
                    logger.debug(
                        "face_anchor_dropped_no_tracklet",
                        person_id=face.person_id,
                        detection_id=det.detection_id,
                        camera_id=camera_id,
                    )
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
        if face_anchors:
            logger.debug(
                "face_anchors_created",
                camera_id=camera_id,
                anchor_count=len(face_anchors),
                identities=[fa.person_id for fa in face_anchors],
            )
        return face_anchors
