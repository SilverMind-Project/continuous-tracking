"""FastAPI application factory for the tracking orchestrator.

Mirrors the cognitive-companion pattern: create_app() returns the app,
lifespan manages service lifecycle (pipeline start/stop, DB migrations).

Configuration is read from ``config/settings.yaml`` (overridable via
``ORCHESTRATOR_CONFIG_PATH``) with ``${VAR}`` / ``${VAR:-default}``
env-var interpolation — see :mod:`app.config`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from structlog import get_logger

from .config import settings
from .observability.logging_config import configure_logging

# Configure structlog before any logger is first used.
configure_logging(settings.get("logging.level", "INFO") or "INFO")
from .inference.depth import DepthEstimator
from .inference.detector import PersonDetector
from .inference.pose import PoseEstimator
from .inference.reid_embedder import ReidEmbedder
from .inference.triton_client import TritonGrpcClient
from .pipeline import FrameProcessingPipeline
from .pipeline.frame_pipeline import FaceIdCameraConfig, PipelineConfig
from .routers import corrections as corrections_router_mod
from .routers import dashboard as dashboard_router_mod
from .routers import gallery as gallery_router_mod
from .routers import live as live_router_mod
from .routers.calibration import router as calibration_router
from .routers.calibration import set_auto_calibration_context
from .routers.corrections import router as corrections_router
from .routers.dashboard import router as dashboard_router
from .routers.gallery import router as gallery_router
from .routers.live import router as live_router
from .routers.trajectory import router as trajectory_router
from .routers.trajectory import set_context as set_trajectory_context
from .sampling.keyframe_sampler import SamplerConfig
from .services.cc_client import CognitiveCompanionClient
from .services.identity_rewriter import InMemoryIdentityRewriter, PostgresIdentityRewriter
from .services.overlap_group_sync import fetch_adjacency_edges, fetch_overlap_groups
from .storage.migrations import MigrationRunner
from .storage.postgres.bbox_annotations import PostgresBboxAnnotationRepository
from .storage.postgres.do_not_fuse import PostgresDoNotFuseRepository
from .storage.postgres.gallery_repo import PostgresGalleryRepository
from .storage.postgres.global_track_repo import PostgresGlobalTrackRepository
from .storage.postgres.keyframe_repo import PostgresKeyframeRepository
from .storage.postgres.settings_repo import PostgresSettingsRepository
from .storage.postgres.signal_repo import PostgresDementiaSignalRepository
from .storage.postgres.tracking_repo import PostgresTrackingRepository
from .storage.postgres.trajectory_repo import PostgresTrajectoryRepository
from .tracking.cross_camera import CrossCamConfig
from .tracking.global_track_merger import GlobalTrackMerger
from .tracking.identity_resolver import ResolverConfig
from .tracking.tracklet_manager import TrackletConfig
from .trajectory.depth_posture_strategy import DepthPostureStrategy
from .trajectory.fused_posture_strategy import FusedPostureStrategy
from .trajectory.posture_strategy import RTMPosePostureStrategy
from .transport.minio_frames import MinioFrameConfig, MinioFrameFetcher
from .transport.redis_streams import TransportConfig

logger = get_logger(__name__)

# Module-level pipeline singleton, initialized in lifespan.
_pipeline: FrameProcessingPipeline | None = None

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _normalize_dsn(dsn: str) -> str:
    """Strip ``+asyncpg`` SQLAlchemy scheme prefix if present."""
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("+asyncpg", "", 1)
    return dsn


async def _fetch_cc_camera_configs(
    client: CognitiveCompanionClient,
) -> dict[str, FaceIdCameraConfig]:
    """Fetch per-camera face-id configs from cognitive-companion's camera API.

    Returns an empty dict if CC is unreachable (graceful degradation).
    CC is the primary source for per-camera face-id settings (enabled flag
    and min_confidence). The ``FACE_ID_CAMERA_CONFIDENCE`` env var provides
    a deployment-level fallback for cameras not in CC.
    """
    try:
        cameras: list[dict[str, Any]] = await client.get("/api/v1/cts/cameras")
    except Exception:
        logger.warning("Failed to fetch camera configs from CC", exc_info=True)
        return {}

    cfgs: dict[str, FaceIdCameraConfig] = {}
    for cam in cameras:
        cam_id = cam.get("id", "")
        if not cam_id:
            continue
        enabled = cam.get("face_id_enabled", True)
        min_conf = cam.get("face_id_min_confidence")
        cfgs[cam_id] = FaceIdCameraConfig(
            enabled=enabled if isinstance(enabled, bool) else True,
            min_confidence=float(min_conf) if min_conf is not None else None,
        )

    # Apply FACE_ID_CAMERA_CONFIDENCE env var as fallback for cameras
    # that were not found in CC's camera list.
    _apply_confidence_fallback(cfgs)

    logger.info("Fetched face-id configs from CC", camera_count=len(cfgs))
    return cfgs


def _apply_confidence_fallback(cfgs: dict[str, FaceIdCameraConfig]) -> None:
    """Apply ``FACE_ID_CAMERA_CONFIDENCE`` env var as a fallback.

    Only affects cameras already present in *cfgs* (from CC).  If a camera
    has no ``min_confidence`` set in CC, the env-var value fills the gap.
    The env var format is ``cam_id:confidence,cam_id:confidence,...``.
    """
    import os

    cam_conf = os.environ.get("FACE_ID_CAMERA_CONFIDENCE", "")
    if not cam_conf:
        return

    for pair in cam_conf.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        cam_id, conf_str = pair.split(":", 1)
        try:
            conf = float(conf_str)
        except ValueError:
            logger.warning("Invalid FACE_ID_CAMERA_CONFIDENCE entry", entry=pair)
            continue

        cam_id = cam_id.strip()
        existing = cfgs.get(cam_id)
        if existing is not None:
            # Only fill if CC didn't set a value.
            if existing.min_confidence is None:
                cfgs[cam_id] = FaceIdCameraConfig(
                    enabled=existing.enabled,
                    min_confidence=conf,
                )
        else:
            logger.info(
                "FACE_ID_CAMERA_CONFIDENCE ignored for unknown camera",
                camera_id=cam_id,
            )


def _str_to_bool(v: object) -> bool:
    """Convert a string or bool to a Python bool."""
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes")


def _build_resolver_config(s: Any) -> ResolverConfig:
    """Build ResolverConfig from parsed settings (dot-notation access)."""
    r = s.get("resolver", {})
    return ResolverConfig(
        commit_prob=float(r.get("commit_prob", 0.65)),
        commit_margin=float(r.get("commit_margin", 0.15)),
        reid_decision_sim=float(r.get("reid_decision_sim", 0.70)),
        revision_horizon_s=float(r.get("revision_horizon_s", 600.0)),
        max_revisions_per_gt_per_minute=int(r.get("max_revisions_per_gt_per_minute", 3)),
        unknown_mass=float(r.get("unknown_mass", 0.05)),
        prior_weight=float(r.get("prior_weight", 0.6)),
        face_weight_multiplier=float(r.get("face_weight_multiplier", 3.0)),
        commit_prob_dense=float(r.get("commit_prob_dense", 0.80)),
        commit_margin_dense=float(r.get("commit_margin_dense", 0.20)),
        prior_maintenance_max_age_s=float(r.get("prior_maintenance_max_age_s", 120.0)),
        identified_entry_boost_min_sim=float(r.get("identified_entry_boost_min_sim", 0.65)),
        identified_entry_min_likelihood=float(r.get("identified_entry_min_likelihood", 0.80)),
        enable_embedding_coherence_boost=_str_to_bool(
            r.get("enable_embedding_coherence_boost", False)
        ),
        embedding_coherence_window=int(r.get("embedding_coherence_window", 5)),
        embedding_coherence_min_sim=float(r.get("embedding_coherence_min_sim", 0.70)),
        embedding_coherence_boost=float(r.get("embedding_coherence_boost", 2.0)),
        face_commit_min_confidence=float(r.get("face_commit_min_confidence", 0.70)),
        face_lock_maintenance_max_age_s=float(r.get("face_lock_maintenance_max_age_s", 300)),
        cross_gt_face_propagation_threshold=float(
            r.get("cross_gt_face_propagation_threshold", 0.65)
        ),
        cross_gt_face_propagation_max_gts=int(r.get("cross_gt_face_propagation_max_gts", 4)),
    )


def _build_tracklet_config(s: Any) -> TrackletConfig:
    """Build TrackletConfig from parsed settings."""
    t = s.get("tracklet", {})
    return TrackletConfig(
        min_hit_ratio=float(t.get("min_hit_ratio", 0.5)),
        close_grace_frames=int(t.get("close_grace_frames", 15)),
        gallery_min_quality=float(t.get("gallery_min_quality", 0.5)),
        gallery_max_per_tracklet=int(t.get("gallery_max_per_tracklet", 20)),
        min_detection_confidence=float(t.get("min_detection_confidence", 0.3)),
        enabled=_str_to_bool(t.get("enabled", True)),
        min_frames_to_publish=int(s.get("pipeline.tracker.min_frames_to_publish", 3)),
    )


def _build_cross_cam_config(s: Any) -> CrossCamConfig:
    """Build CrossCamConfig from parsed settings."""
    cc = s.get("cross_camera", {})
    return CrossCamConfig(
        alpha=float(cc.get("alpha", 0.7)),
        floor_sigma_m=float(cc.get("floor_sigma_m", 1.5)),
        max_floor_distance_m=float(cc.get("max_floor_distance_m", 8.0)),
        min_link_score=float(cc.get("min_link_score", 0.55)),
        unknown_merge_appearance_threshold=float(
            cc.get("unknown_merge_appearance_threshold", 0.92)
        ),
        within_group_min_score=float(cc.get("within_group_min_score", 0.35)),
        inter_gt_consolidation_appearance_threshold=float(
            cc.get("inter_gt_consolidation_appearance_threshold", 0.88)
        ),
        known_identity_reentry_threshold=float(cc.get("known_identity_reentry_threshold", 0.72)),
        same_camera_reentry_max_gap_s=float(cc.get("same_camera_reentry_max_gap_s", 30.0)),
    )


def _build_sampler_config(s: Any) -> SamplerConfig:
    """Build SamplerConfig from parsed settings."""
    sp = s.get("sampler", {})
    return SamplerConfig(
        keyframe_min_interval_s=float(sp.get("keyframe_min_interval_s", 30.0)),
        periodic_expires_hours=int(sp.get("periodic_expires_hours", 72)),
        trigger_expires_days=int(sp.get("trigger_expires_days", 30)),
    )


# Module-level asyncpg pool so shutdown can close it.
_pool: Any = None  # asyncpg.Pool | None
_triton_client: TritonGrpcClient | None = None
_frame_fetcher: MinioFrameFetcher | None = None


def get_pipeline() -> FrameProcessingPipeline | None:
    """Access the running pipeline from dependency injection."""
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline, _pool, _triton_client, _frame_fetcher

    # -------------------------------------------------------------------
    # Startup: build config, connect services, start pipeline
    # -------------------------------------------------------------------

    redis_url = settings.get("redis.url", "redis://localhost:6379/0")
    face_id_url = settings.get("face_id.url", "")

    # Per-camera face-id config: CC is the primary source.
    cc_url = settings.get("cognitive_companion.url", "")
    cc_api_key = settings.get("cognitive_companion.api_key", "")
    cc_client = CognitiveCompanionClient(cc_url, cc_api_key)
    face_id_camera_configs = await _fetch_cc_camera_configs(cc_client)
    overlap_groups = await fetch_overlap_groups(cc_client)
    adjacency_edges_raw = await fetch_adjacency_edges(cc_client)

    # Build config objects from settings (env-interpolated YAML).
    resolver_config = _build_resolver_config(settings)
    tracklet_config = _build_tracklet_config(settings)
    cross_cam_config = _build_cross_cam_config(settings)
    sampler_config = _build_sampler_config(settings)

    config = PipelineConfig(
        transport=TransportConfig(redis_url=redis_url),
        resolver=resolver_config,
        tracklet=tracklet_config,
        cross_cam=cross_cam_config,
        sampler=sampler_config,
        face_id_url=face_id_url,
        face_id_cooldown_s=float(settings.get("face_id.cooldown_s", "5.0")),
        face_id_timeout_s=float(settings.get("face_id.timeout_s", "2.0")),
        face_id_min_confidence=float(settings.get("face_id.min_confidence", "0.4")),
        face_id_enabled=bool(face_id_url),
        face_id_camera_configs=face_id_camera_configs,
        timezone=settings.get("app.timezone", "UTC"),
        signal_stillness_threshold_minutes=int(
            settings.get("signal.stillness_threshold_minutes", "60")
        ),
        signal_stillness_emergency_minutes=int(
            settings.get("signal.stillness_emergency_minutes", "120")
        ),
        signal_stillness_motion_floor=float(settings.get("signal.stillness_motion_floor", "0.02")),
        signal_pacing_room_threshold=int(settings.get("signal.pacing_room_threshold", "8")),
        signal_pacing_window_minutes=int(settings.get("signal.pacing_window_minutes", "30")),
        signal_nighttime_transition_threshold=int(
            settings.get("signal.nighttime_transition_threshold", "3")
        ),
        signal_absence_threshold_minutes=int(
            settings.get("signal.absence_threshold_minutes", "60")
        ),
        signal_bathroom_absolute_threshold_seconds=int(
            settings.get("signal.bathroom_absolute_threshold_seconds", "2700")
        ),
        allow_skeleton=str(settings.get("pipeline.allow_skeleton", "false")).lower()
        in ("1", "true", "yes"),
        # Phase 1: noise reduction
        detection_iou_dedup_threshold=float(
            settings.get("pipeline.detection.iou_dedup_threshold", "0.55")
        ),
        tracker_dedup_iou_threshold=float(
            settings.get("pipeline.tracker.dedup_iou_threshold", "0.7")
        ),
        tracker_min_frames_to_publish=int(
            settings.get("pipeline.tracker.min_frames_to_publish", "3")
        ),
        identity_commit_window_s=float(settings.get("pipeline.identity.commit_window_s", "3.0")),
        identity_high_confidence_face_threshold=float(
            settings.get("pipeline.identity.high_confidence_face_threshold", "0.85")
        ),
        identity_committer_enabled=str(
            settings.get("pipeline.identity.committer_enabled", "false")
        ).lower()
        in ("1", "true", "yes"),
    )
    _pipeline = FrameProcessingPipeline(config)

    # -- Database --
    database_url = settings.get("database.url", "")

    if not database_url:
        env = settings.get("env", "production")
        if env in ("dev", "development", "test"):
            logger.info("No database.url; using in-memory storage for this environment.")
        else:
            logger.warning(
                "No database.url configured; running with in-memory storage. "
                "State will be lost on restart.",
            )

    if database_url:
        import asyncpg  # type: ignore[import-untyped]

        _pool = await asyncpg.create_pool(
            dsn=_normalize_dsn(database_url),
            min_size=2,
            max_size=20,
            command_timeout=10.0,
            server_settings={"search_path": "continuous_tracking, public"},
        )
        runner = MigrationRunner(_pool, MIGRATIONS_DIR)
        await runner.migrate()
        tracking_repo = PostgresTrackingRepository(_pool)
        gallery_repo = PostgresGalleryRepository(_pool)
        global_track_repo = PostgresGlobalTrackRepository(_pool)
        trajectory_repo = PostgresTrajectoryRepository(_pool)
        keyframe_repo = PostgresKeyframeRepository(_pool)
        signal_repo = PostgresDementiaSignalRepository(_pool)
        settings_repo = PostgresSettingsRepository(_pool)
        bbox_repo = PostgresBboxAnnotationRepository(_pool)
        dnf_repo = PostgresDoNotFuseRepository(_pool)
    else:
        tracking_repo = None
        gallery_repo = None
        global_track_repo = None
        trajectory_repo = None
        keyframe_repo = None
        signal_repo = None
        settings_repo = None
        bbox_repo = None

    # -- Triton --
    triton_url = settings.get("triton.url", "")
    detector = None
    reid_embedder = None
    pose_estimator = None
    env = settings.get("env", "production")
    depth_estimator: DepthEstimator | None = None
    if triton_url:
        try:
            triton_timeout_ms = int(settings.get("triton.timeout_ms", "5000") or "5000")
            _triton_client = TritonGrpcClient(triton_url, timeout_ms=triton_timeout_ms)
            await _triton_client.__aenter__()
            detector = PersonDetector(
                _triton_client,
                conf_threshold=float(settings.get("pipeline.detector_confidence", "0.35")),
            )
            reid_embedder = ReidEmbedder(
                _triton_client,
                model_name=settings.get("triton.reid_model", "reid-solider"),
            )
            pose_enabled_str = settings.get("triton.pose_enabled", "true")
            if str(pose_enabled_str).lower() not in ("0", "false", "no"):
                pose_estimator = PoseEstimator(
                    _triton_client,
                    model_name=settings.get("triton.pose_model", "pose-rtmpose"),
                )
            depth_enabled_str = settings.get("triton.depth_enabled", "true")
            if str(depth_enabled_str).lower() not in ("0", "false", "no"):
                depth_model_name = settings.get("triton.depth_model", "depth-anything-v2")
                if await _triton_client.is_model_ready(depth_model_name):
                    depth_estimator = DepthEstimator(
                        _triton_client,
                        model_name=depth_model_name,
                    )
                else:
                    logger.warning(
                        "depth_model_not_ready",
                        model=depth_model_name,
                        hint="export_depth_anything_v2.py must be run to produce model.onnx",
                    )
        except Exception:
            logger.exception("Failed to connect to Triton Inference Server")
            if env in ("production", "staging"):
                raise RuntimeError(
                    "Triton Inference Server is required in production/staging. "
                    "Set PIPELINE_ALLOW_SKELETON=true to override for testing."
                ) from None
            _triton_client = None

    # -- MinIO --
    minio_endpoint = settings.get("minio.endpoint", "")
    minio_bucket = settings.get("minio.bucket", "")
    minio_access_key = settings.get("minio.access_key", "")
    minio_secret_key = settings.get("minio.secret_key", "")
    if minio_endpoint and minio_bucket and minio_access_key and minio_secret_key:
        secure_str = settings.get("minio.secure", "false")
        secure = str(secure_str).lower() in {"1", "true", "yes"}
        _frame_fetcher = MinioFrameFetcher(
            MinioFrameConfig(
                endpoint_url=minio_endpoint,
                bucket=minio_bucket,
                access_key_id=minio_access_key,
                secret_access_key=minio_secret_key,
                region_name=settings.get("minio.region", "us-east-1"),
                secure=secure,
            )
        )
        await _frame_fetcher.connect()
    elif detector is not None:
        logger.warning(
            "Triton configured without MinIO frame storage; inference will run on blank frames.",
        )

    # -- Auto-calibration (depth-based homography estimation) --
    if depth_estimator is not None:
        from .calibration.auto_calibrator import AutoCalibrator

        auto_calibrator = AutoCalibrator(depth_estimator=depth_estimator)
        set_auto_calibration_context(auto_calibrator=auto_calibrator, frame_fetcher=_frame_fetcher)
        logger.info("auto_calibrator_ready", model="depth-anything-v2")
    else:
        set_auto_calibration_context(auto_calibrator=None, frame_fetcher=_frame_fetcher)
        logger.info("auto_calibrator_disabled", reason="triton_not_connected_or_depth_disabled")

    # -- Wire posture strategy --
    fast_strategy = RTMPosePostureStrategy()

    depth_slow_path_enabled = str(
        settings.get("pipeline.posture.depth_slow_path_enabled", "false")
    ).lower() in ("1", "true", "yes")

    if depth_estimator is not None and depth_slow_path_enabled:
        slow_strategy = DepthPostureStrategy(depth_estimator)
        posture_strategy: RTMPosePostureStrategy | FusedPostureStrategy = FusedPostureStrategy(
            fast=fast_strategy,
            slow=slow_strategy,
            slow_path_min_interval_s=float(
                settings.get("pipeline.posture.depth_slow_path_min_interval_s", "15.0")
            ),
            slow_path_max_age_s=float(
                settings.get("pipeline.posture.depth_slow_path_max_age_s", "60.0")
            ),
        )
        logger.info(
            "Posture strategy: fused (RTMPose + Depth slow-path)",
            slow_path_interval_s=settings.get(
                "pipeline.posture.depth_slow_path_min_interval_s", "15.0"
            ),
        )
    else:
        posture_strategy = fast_strategy
        logger.info("Posture strategy: RTMPose only (depth slow-path disabled)")

    # -- Wire everything and start --
    identity_rewriter = (
        PostgresIdentityRewriter(_pool) if _pool is not None else InMemoryIdentityRewriter()
    )
    await _pipeline.initialize(
        detector=detector,
        tracking_repo=tracking_repo,
        gallery_repo=gallery_repo,
        global_track_repo=global_track_repo,
        trajectory_repo=trajectory_repo,
        keyframe_repo=keyframe_repo,
        signal_repo=signal_repo,
        settings_repo=settings_repo,
        frame_fetcher=_frame_fetcher,
        reid_embedder=reid_embedder,
        pose_estimator=pose_estimator,
        posture_strategy=posture_strategy,
        identity_rewriter=identity_rewriter,
        bbox_repo=bbox_repo,
        dnf_repo=dnf_repo,
    )
    _pipeline.set_overlap_groups(overlap_groups)

    # Restore persisted adjacency edges from CC DB into in-memory calibration state.
    if adjacency_edges_raw:
        from .calibration.state import AdjacencyEdge as _AdjacencyEdge
        from .calibration.state import calibration_state

        edges = [
            _AdjacencyEdge(
                from_camera=e.get("from", ""),
                to_camera=e.get("to", ""),
                min_transit_s=float(e.get("min_transit_s", 0.5)),
                max_transit_s=float(e.get("max_transit_s", 30.0)),
                overlap=bool(e.get("overlap", False)),
            )
            for e in adjacency_edges_raw
            if e.get("from") and e.get("to")
        ]
        await calibration_state.set_adjacency(edges)
        logger.info("Restored adjacency edges from CC", edge_count=len(edges))

    await _pipeline.start()

    # Wire router modules to share the pipeline's repositories.
    if _pipeline.tracking_repo is not None and _pipeline.global_track_repo is not None:
        global_track_merger = GlobalTrackMerger(_pool) if _pool is not None else None
        corrections_router_mod.set_context(
            tracking_repo=_pipeline.tracking_repo,
            global_track_repo=_pipeline.global_track_repo,
            publisher=_pipeline.revision_publisher,
            dnf_repo=dnf_repo,
            merger=global_track_merger,
        )
        live_router_mod.set_context(
            global_track_repo=_pipeline.global_track_repo,
            keyframe_repo=keyframe_repo,
            gallery_repo=gallery_repo,
        )

    if trajectory_repo is not None:
        set_trajectory_context(trajectory_repo=trajectory_repo)

    if gallery_repo is not None:
        gallery_router_mod.set_context(gallery_repo=gallery_repo, reid_embedder=reid_embedder)

    if signal_repo is not None and trajectory_repo is not None and keyframe_repo is not None:
        dashboard_router_mod.set_repos(
            signal=signal_repo,
            trajectory=trajectory_repo,
            keyframe=keyframe_repo,
        )

    if bbox_repo is not None:
        dashboard_router_mod.set_bbox_repo(bbox_repo)

    yield

    # -------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------
    if _pipeline is not None:
        await _pipeline.stop()
        _pipeline = None

    if _frame_fetcher is not None:
        await _frame_fetcher.disconnect()
        _frame_fetcher = None

    if _triton_client is not None:
        await _triton_client.__aexit__(None, None, None)
        _triton_client = None

    if _pool is not None:
        await _pool.close()
        _pool = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Continuous Tracking — Orchestrator",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Internal calibration endpoints consumed by the CC BFF.
    # Phase-0 §0.27 will move these to a separate :8510 listener; for M7
    # they live on the same app under /internal/ to keep the routing simple.
    app.include_router(calibration_router)
    app.include_router(dashboard_router)
    app.include_router(corrections_router)
    app.include_router(gallery_router)
    app.include_router(live_router)
    app.include_router(trajectory_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        status = "starting"
        if _pipeline is not None:
            status = "running" if _pipeline.is_running else "stopped"
        return {
            "status": status,
            "service": "tracking-orchestrator",
            "version": "0.1.0",
        }

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Prometheus scrape target.

        Exposes the default ``prometheus_client`` registry so the
        observability stack can pull counters, gauges, and histograms
        registered by :mod:`app.observability.metrics`.
        """
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
