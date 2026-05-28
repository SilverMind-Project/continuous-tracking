"""FastAPI application factory for the tracking orchestrator.

Mirrors the cognitive-companion pattern: create_app() returns the app,
lifespan manages service lifecycle (pipeline start/stop, DB migrations).

Configuration is read from ``config/settings.yaml`` (overridable via
``ORCHESTRATOR_CONFIG_PATH``) with ``${VAR}`` / ``${VAR:-default}``
env-var interpolation — see :mod:`app.config`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from structlog import get_logger

from .config import Settings, settings
from .observability.logging_config import configure_logging

# Configure structlog before any logger is first used.
configure_logging(settings.as_str("logging.level"))
from .inference.depth import DepthEstimator
from .inference.detector import PersonDetector
from .inference.pose import PoseEstimator
from .inference.reid_embedder import ReidEmbedder
from .inference.triton_client import TritonGrpcClient
from .pipeline import FrameProcessingPipeline
from .pipeline.frame_pipeline import (
    FaceIdConfig,
    PipelineConfig,
    PipelineDependencies,
    SignalConfig,
)
from .pipeline.types import FaceIdCameraConfig
from .routers import dashboard as dashboard_router_mod
from .routers import gallery as gallery_router_mod
from .routers.calibration import router as calibration_router
from .routers.calibration import set_auto_calibration_context
from .routers.corrections import router as corrections_router
from .routers.dashboard import router as dashboard_router
from .routers.gallery import router as gallery_router
from .routers.live import router as live_router
from .routers.ph import router as ph_router
from .routers.ph import set_ph_repository, set_revision_publisher
from .routers.trajectory import router as trajectory_router
from .routers.trajectory import set_context as set_trajectory_context
from .sampling.keyframe_sampler import SamplerConfig
from .services.cc_client import CognitiveCompanionClient
from .services.identity_rewriter import InMemoryIdentityRewriter, PostgresIdentityRewriter
from .services.overlap_group_sync import fetch_adjacency_edges, fetch_overlap_groups
from .storage.migrations import MigrationRunner
from .storage.postgres.bbox_annotations import PostgresBboxAnnotationRepository
from .storage.postgres.gallery_repo import PostgresGalleryRepository
from .storage.postgres.keyframe_repo import PostgresKeyframeRepository
from .storage.postgres.ph_repo import PostgresPHRepository
from .storage.postgres.settings_repo import PostgresSettingsRepository
from .storage.postgres.signal_repo import PostgresDementiaSignalRepository
from .storage.postgres.trajectory_repo import PostgresTrajectoryRepository
from .tracking.identity_resolver import ResolverConfig
from .tracking.world.config import WorldTrackerConfig
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
        cam_id = cam.get("id")
        enabled = cam.get("face_id_enabled")
        if not cam_id or not isinstance(enabled, bool):
            logger.warning("Skipped malformed CC camera config", camera=cam)
            continue
        min_conf = cam.get("face_id_min_confidence")
        cfgs[str(cam_id)] = FaceIdCameraConfig(
            enabled=enabled,
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


def _build_transport_config(s: Settings) -> TransportConfig:
    """Build TransportConfig from required settings.yaml keys."""
    return TransportConfig(
        redis_url=s.as_str("redis.url"),
        consumer_group=s.as_str("redis.consumer_group"),
        consumer_name=s.as_str("redis.consumer_name"),
        frames_stream=s.as_str("redis.frames_stream"),
        events_stream=s.as_str("redis.events_stream"),
        responses_stream=s.as_str("redis.responses_stream"),
        batch_max_wait_ms=s.as_int("redis.batch_max_wait_ms"),
        batch_max_size=s.as_int("redis.batch_max_size"),
        xack_timeout_ms=s.as_int("redis.xack_timeout_ms"),
        ack_ttl_seconds=s.as_int("redis.ack_ttl_seconds"),
        revisions_stream=s.as_str("redis.revisions_stream"),
        signals_stream=s.as_str("redis.signals_stream"),
        scene_samples_stream=s.as_str("redis.scene_samples_stream"),
    )


def _build_resolver_config(s: Settings) -> ResolverConfig:
    """Build ResolverConfig from required settings.yaml keys."""
    r = s.section("resolver")
    return ResolverConfig(
        commit_prob=r.as_float("commit_prob"),
        commit_margin=r.as_float("commit_margin"),
        reid_decision_sim=r.as_float("reid_decision_sim"),
        revision_horizon_s=r.as_float("revision_horizon_s"),
        max_revisions_per_ph_per_minute=r.as_int("max_revisions_per_ph_per_minute"),
        unknown_mass=r.as_float("unknown_mass"),
        prior_weight=r.as_float("prior_weight"),
        face_weight_multiplier=r.as_float("face_weight_multiplier"),
        propagated_face_weight_multiplier=r.as_float("propagated_face_weight_multiplier"),
        height_weight_multiplier=r.as_float("height_weight_multiplier"),
        commit_prob_dense=r.as_float("commit_prob_dense"),
        commit_margin_dense=r.as_float("commit_margin_dense"),
        prior_maintenance_max_age_s=r.as_float("prior_maintenance_max_age_s"),
        identified_entry_boost_min_sim=r.as_float("identified_entry_boost_min_sim"),
        identified_entry_min_likelihood=r.as_float("identified_entry_min_likelihood"),
        enable_embedding_coherence_boost=r.as_bool("enable_embedding_coherence_boost"),
        embedding_coherence_window=r.as_int("embedding_coherence_window"),
        embedding_coherence_min_sim=r.as_float("embedding_coherence_min_sim"),
        embedding_coherence_boost=r.as_float("embedding_coherence_boost"),
        face_commit_min_confidence=r.as_float("face_commit_min_confidence"),
        face_lock_maintenance_max_age_s=r.as_float("face_lock_maintenance_max_age_s"),
        cross_gt_face_propagation_threshold=r.as_float("cross_gt_face_propagation_threshold"),
        cross_gt_face_propagation_max_gts=r.as_int("cross_gt_face_propagation_max_gts"),
    )


def _build_world_tracker_config(s: Settings) -> WorldTrackerConfig:
    """Build WorldTrackerConfig from required settings.yaml keys."""
    wt = s.section("world_tracker")
    return WorldTrackerConfig(
        gate_chi2=wt.as_float("gate_chi2"),
        alpha_geo=wt.as_float("alpha_geo"),
        alpha_app=wt.as_float("alpha_app"),
        alpha_height=wt.as_float("alpha_height"),
        observation_noise_m=wt.as_float("observation_noise_m"),
        process_noise_accel_m_s2=wt.as_float("process_noise_accel_m_s2"),
        ph_close_grace_s=wt.as_float("ph_close_grace_s"),
        min_observations_to_publish=wt.as_int("min_observations_to_publish"),
        face_conflict_threshold=wt.as_float("face_conflict_threshold"),
        inferred_handoff_max_s=wt.as_float("inferred_handoff_max_s"),
        inferred_handoff_max_distance_m=wt.as_float("inferred_handoff_max_distance_m"),
        initial_position_sigma_m=wt.as_float("initial_position_sigma_m"),
        initial_velocity_sigma_m_s=wt.as_float("initial_velocity_sigma_m_s"),
        velocity_decay_s=wt.as_float("velocity_decay_s"),
        height_sigma_m=wt.as_float("height_sigma_m"),
        evidence_window_s=wt.as_float("evidence_window_s"),
    )


def _build_sampler_config(s: Settings) -> SamplerConfig:
    """Build SamplerConfig from required settings.yaml keys."""
    sp = s.section("sampler")
    return SamplerConfig(
        keyframe_min_interval_s=sp.as_float("keyframe_min_interval_s"),
        periodic_expires_hours=sp.as_int("periodic_expires_hours"),
        trigger_expires_days=sp.as_int("trigger_expires_days"),
    )


def _build_signal_config(s: Settings) -> SignalConfig:
    sig = s.section("signal")
    return SignalConfig(
        interval_s=sig.as_int("interval_s"),
        enabled=sig.as_bool("enabled"),
        timezone=s.as_str("app.timezone"),
        stillness_threshold_minutes=sig.as_int("stillness_threshold_minutes"),
        stillness_emergency_minutes=sig.as_int("stillness_emergency_minutes"),
        stillness_motion_floor=sig.as_float("stillness_motion_floor"),
        pacing_room_threshold=sig.as_int("pacing_room_threshold"),
        pacing_window_minutes=sig.as_int("pacing_window_minutes"),
        nighttime_transition_threshold=sig.as_int("nighttime_transition_threshold"),
        absence_threshold_minutes=sig.as_int("absence_threshold_minutes"),
        bathroom_absolute_threshold_seconds=sig.as_int("bathroom_absolute_threshold_seconds"),
    )


def _build_face_id_config(
    s: Settings, camera_configs: dict[str, FaceIdCameraConfig]
) -> FaceIdConfig:
    fi = s.section("face_id")
    url = s.as_str("face_id.url")
    return FaceIdConfig(
        url=url,
        cooldown_s=fi.as_float("cooldown_s"),
        timeout_s=fi.as_float("timeout_s"),
        min_confidence=fi.as_float("min_confidence"),
        enabled=bool(url),
        camera_configs=camera_configs,
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

    transport_config = _build_transport_config(settings)

    # Per-camera face-id config: CC is the primary source.
    cc_url = settings.as_str("cognitive_companion.url")
    cc_api_key = settings.as_str("cognitive_companion.api_key")
    cc_client = CognitiveCompanionClient(cc_url, cc_api_key)
    face_id_camera_configs = await _fetch_cc_camera_configs(cc_client)
    overlap_groups = await fetch_overlap_groups(cc_client)
    adjacency_edges_raw = await fetch_adjacency_edges(cc_client)

    # Build config objects from settings (env-interpolated YAML).
    resolver_config = _build_resolver_config(settings)
    world_tracker_config = _build_world_tracker_config(settings)
    sampler_config = _build_sampler_config(settings)

    config = PipelineConfig(
        transport=transport_config,
        resolver=resolver_config,
        world_tracker=world_tracker_config,
        sampler=sampler_config,
        signals=_build_signal_config(settings),
        face_id=_build_face_id_config(settings, face_id_camera_configs),
        max_concurrent_frames=settings.as_int("pipeline.max_concurrent_frames"),
        shutdown_timeout=settings.as_float("pipeline.shutdown_timeout_s"),
        pose_enabled=settings.as_bool("triton.pose_enabled"),
        allow_skeleton=settings.as_bool("pipeline.allow_skeleton"),
        # Phase 1: noise reduction
        detection_iou_dedup_threshold=settings.as_float("pipeline.detection.iou_dedup_threshold"),
        identity_commit_window_s=settings.as_float("pipeline.identity.commit_window_s"),
        identity_high_confidence_face_threshold=settings.as_float(
            "pipeline.identity.high_confidence_face_threshold"
        ),
        gallery_identity_backfill_delay_s=settings.as_float(
            "pipeline.identity.gallery_backfill_delay_s"
        ),
        identity_rewrite_on_face_commit=settings.as_bool(
            "pipeline.identity.rewrite_on_face_commit"
        ),
        batch_window_s=settings.as_float("pipeline.batch_window_s"),
        max_batch_size=settings.as_int("pipeline.max_batch_size"),
    )
    _pipeline = FrameProcessingPipeline(config)

    # -- Database --
    database_url = settings.as_str("database.url")

    if not database_url:
        env = settings.as_str("env")
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
        gallery_repo = PostgresGalleryRepository(_pool)
        trajectory_repo = PostgresTrajectoryRepository(_pool)
        keyframe_repo = PostgresKeyframeRepository(_pool)
        signal_repo = PostgresDementiaSignalRepository(_pool)
        settings_repo = PostgresSettingsRepository(_pool)
        bbox_repo = PostgresBboxAnnotationRepository(_pool)
    else:
        gallery_repo = None
        trajectory_repo = None
        keyframe_repo = None
        signal_repo = None
        settings_repo = None
        bbox_repo = None

    # -- Triton --
    triton_url = settings.as_str("triton.url")
    detector = None
    reid_embedder = None
    pose_estimator = None
    env = settings.as_str("env")
    triton_timeout_ms = settings.as_int("triton.timeout_ms")
    detector_model_name = settings.as_str("triton.person_detector_model")
    detector_confidence = settings.as_float("pipeline.detector_confidence")
    reid_model_name = settings.as_str("triton.reid_model")
    pose_enabled = settings.as_bool("triton.pose_enabled")
    pose_model_name = settings.as_str("triton.pose_model")
    depth_enabled = settings.as_bool("triton.depth_enabled")
    depth_model_name = settings.as_str("triton.depth_model")
    depth_estimator: DepthEstimator | None = None
    if triton_url:
        startup_wait_s = settings.as_int("triton.startup_wait_s")
        startup_retry_s = settings.as_int("triton.startup_retry_interval_s")
        deadline = asyncio.get_event_loop().time() + startup_wait_s
        last_exc: Exception | None = None
        connected = False
        while asyncio.get_event_loop().time() < deadline:
            try:
                _triton_client = TritonGrpcClient(triton_url, timeout_ms=triton_timeout_ms)
                await _triton_client.__aenter__()
                connected = True
                break
            except Exception as exc:
                last_exc = exc
                remaining = deadline - asyncio.get_event_loop().time()
                logger.warning(
                    "triton_not_ready",
                    url=triton_url,
                    remaining_s=round(remaining, 1),
                    retry_interval_s=startup_retry_s,
                )
                await asyncio.sleep(startup_retry_s)

        if not connected:
            logger.exception("Failed to connect to Triton Inference Server", exc_info=last_exc)
            if env in ("production", "staging"):
                raise RuntimeError(
                    "Triton Inference Server is required in production/staging. "
                    "Set PIPELINE_ALLOW_SKELETON=true to override for testing."
                ) from None
            _triton_client = None
        else:
            assert _triton_client is not None  # narrow for mypy
            detector = PersonDetector(
                _triton_client,
                model_name=detector_model_name,
                conf_threshold=detector_confidence,
            )
            reid_embedder = ReidEmbedder(
                _triton_client,
                model_name=reid_model_name,
            )
            if pose_enabled:
                pose_estimator = PoseEstimator(
                    _triton_client,
                    model_name=pose_model_name,
                )
            if depth_enabled:
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

    # -- MinIO --
    minio_endpoint = settings.as_str("minio.endpoint")
    minio_bucket = settings.as_str("minio.bucket")
    minio_access_key = settings.as_str("minio.access_key")
    minio_secret_key = settings.as_str("minio.secret_key")
    if minio_endpoint and minio_bucket and minio_access_key and minio_secret_key:
        secure = settings.as_bool("minio.secure")
        _frame_fetcher = MinioFrameFetcher(
            MinioFrameConfig(
                endpoint_url=minio_endpoint,
                bucket=minio_bucket,
                access_key_id=minio_access_key,
                secret_access_key=minio_secret_key,
                region_name=settings.as_str("minio.region"),
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

    depth_slow_path_enabled = settings.as_bool("pipeline.posture.depth_slow_path_enabled")

    if depth_estimator is not None and depth_slow_path_enabled:
        slow_strategy = DepthPostureStrategy(depth_estimator)
        posture_strategy: RTMPosePostureStrategy | FusedPostureStrategy = FusedPostureStrategy(
            fast=fast_strategy,
            slow=slow_strategy,
            slow_path_min_interval_s=settings.as_float(
                "pipeline.posture.depth_slow_path_min_interval_s"
            ),
            slow_path_max_age_s=settings.as_float("pipeline.posture.depth_slow_path_max_age_s"),
        )
        logger.info(
            "Posture strategy: fused (RTMPose + Depth slow-path)",
            slow_path_interval_s=settings.as_float(
                "pipeline.posture.depth_slow_path_min_interval_s"
            ),
        )
    else:
        posture_strategy = fast_strategy
        logger.info("Posture strategy: RTMPose only (depth slow-path disabled)")

    # -- N1: PH repository (Postgres or in-memory fallback) --
    from .storage.base import InMemoryPHRepository as _InMemPH

    if _pool is not None:
        _ph_repo_n1: _InMemPH | PostgresPHRepository = PostgresPHRepository(_pool)
    else:
        _ph_repo_n1 = _InMemPH()
    set_ph_repository(_ph_repo_n1)  # type: ignore[arg-type]
    deps_ph_repo = _ph_repo_n1

    # WTR6: wire revision publisher for manual PH corrections.
    if _pipeline._revision_publisher is not None:
        set_revision_publisher(_pipeline._revision_publisher)

    # -- Wire everything and start --
    identity_rewriter = (
        PostgresIdentityRewriter(_pool) if _pool is not None else InMemoryIdentityRewriter()
    )
    deps = PipelineDependencies(
        detector=detector,
        gallery_repo=gallery_repo,
        ph_repo=deps_ph_repo,  # type: ignore[arg-type] # N1: Postgres-backed PH repository
        trajectory_repo=trajectory_repo,
        keyframe_repo=keyframe_repo,
        signal_repo=signal_repo,
        settings_repo=settings_repo,
        frame_fetcher=_frame_fetcher,
        reid_embedder=reid_embedder,
        pose_estimator=pose_estimator,
        posture_strategy=posture_strategy,
        identity_rewriter=identity_rewriter,
        bbox_repo=bbox_repo,  # type: ignore[arg-type]
    )
    await _pipeline.initialize(deps)
    _pipeline.set_overlap_groups(overlap_groups)

    # Restore persisted adjacency edges from CC DB into in-memory calibration state.
    from .calibration.state import AdjacencyEdge as _AdjacencyEdge
    from .calibration.state import calibration_state

    if adjacency_edges_raw:
        edges: list[_AdjacencyEdge] = []
        for e in adjacency_edges_raw:
            try:
                overlap_raw = e["overlap"]
                if isinstance(overlap_raw, bool):
                    overlap = overlap_raw
                else:
                    normalized_overlap = str(overlap_raw).strip().lower()
                    if normalized_overlap in {"1", "true", "yes", "on"}:
                        overlap = True
                    elif normalized_overlap in {"0", "false", "no", "off"}:
                        overlap = False
                    else:
                        raise ValueError(f"Invalid adjacency overlap value: {overlap_raw!r}")
                edges.append(
                    _AdjacencyEdge(
                        from_camera=str(e["from"]),
                        to_camera=str(e["to"]),
                        min_transit_s=float(e["min_transit_s"]),
                        max_transit_s=float(e["max_transit_s"]),
                        overlap=overlap,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipped malformed CC adjacency edge", edge=e, error=str(exc))
        await calibration_state.set_adjacency(edges)
        logger.info("Restored adjacency edges from CC", edge_count=len(edges))

    await _pipeline.start()

    # M2: CC config sync service (camera→room map + homography sync).
    from .services.camera_room_map import CameraRoomMap
    from .services.cc_config_sync import CCConfigSyncService

    camera_room_map = CameraRoomMap()
    _poll_s = settings.as_float("cc_sync.poll_interval_s")
    cc_sync_service = CCConfigSyncService(
        client=cc_client,
        calibration_state=calibration_state,
        camera_room_map=camera_room_map,
        poll_interval_s=_poll_s,
    )
    app.state.cc_sync_service = cc_sync_service
    app.state.cc_sync_task = asyncio.create_task(cc_sync_service.run())
    app.state.camera_room_map = camera_room_map
    # Wire CameraRoomMap into the pipeline so stages read room bindings.
    _pipeline.set_camera_room_map(camera_room_map)

    # WTR5: load transit zones from CC and wire into the world tracker.
    from .domain import TransitZone as _TransitZone
    from .tracking.world.transit_detector import TransitDetector as _TransitDetector
    from .transport.room_transition_publisher import (
        RoomTransitionPublisher as _RoomTransitionPublisher,
    )

    _transit_zones: list[_TransitZone] = []
    try:
        _tz_data = await cc_client.get("/api/v1/cts/transit-zones")
        if isinstance(_tz_data, list):
            for tz in _tz_data:
                if not isinstance(tz, dict):
                    continue
                _poly_raw = tz.get("polygon", [])
                _polygon: list[tuple[float, float]] = [
                    (float(p[0]), float(p[1]))
                    for p in _poly_raw
                    if isinstance(p, list) and len(p) >= 2
                ]
                if len(_polygon) < 3:
                    continue
                _dir_raw = tz.get("direction_vec", [0.0, 0.0])
                _transit_zones.append(
                    _TransitZone(
                        zone_id=str(tz.get("id", "")),
                        name=str(tz.get("name", "")),
                        kind=str(tz.get("kind", "door")),
                        polygon=_polygon,
                        inside_room_id=str(tz.get("inside_room_id", "")),
                        outside_room_id=str(tz.get("outside_room_id", "")),
                        direction_vec=(
                            float(_dir_raw[0]) if len(_dir_raw) > 0 else 0.0,
                            float(_dir_raw[1]) if len(_dir_raw) > 1 else 0.0,
                        ),
                    )
                )
        logger.info("transit_zones_loaded", count=len(_transit_zones))
    except Exception:
        logger.exception("transit_zone_load_failed")

    _transit_detector = _TransitDetector()
    _room_transition_pub = _RoomTransitionPublisher(
        redis_url=settings.as_str("redis.url"),
    )
    await _room_transition_pub.connect()
    _pipeline.set_transit_config(
        transit_detector=_transit_detector,
        transit_zones=_transit_zones,  # type: ignore[arg-type]
        room_transition_publisher=_room_transition_pub,
    )

    # WTR2: CC identity assertion subscriber (bidirectional identity flow).
    cc_assertion_cache: object | None = None
    cc_assertion_subscriber: object | None = None
    redis_url = settings.as_str("redis.url")
    if redis_url:
        try:
            from .services.cc_identity_assertion_subscriber import (
                CCIdentityAssertionSubscriber,
                IdentityAssertionCache,
            )

            cc_assertion_cache = IdentityAssertionCache()
            # Use a dedicated Redis client for the subscriber so it does not
            # share the pipeline's pub/sub connection.
            import redis.asyncio as aioredis

            _cc_redis = aioredis.from_url(redis_url, decode_responses=False)
            cc_assertion_subscriber = CCIdentityAssertionSubscriber(
                redis_client=_cc_redis,
                cache=cc_assertion_cache,
            )
            await cc_assertion_subscriber.start()
            logger.info("cc_assertion_subscriber_started", stream="cc.identity_assertions")
        except Exception:
            logger.exception("cc_assertion_subscriber_init_failed")
            cc_assertion_cache = None
            cc_assertion_subscriber = None

    # Inject assertion cache into the WorldTrackingStage for CC identity flow.
    if cc_assertion_cache is not None and _pipeline._stage_runner is not None:
        for stage in _pipeline._stage_runner._stages:
            if stage.name == "world_tracking":
                stage._assertion_cache = cc_assertion_cache  # type: ignore[attr-defined]
                break

    # M3: keyframe revalidator background task.
    if bbox_repo is not None:
        from .services.keyframe_revalidator import KeyframeRevalidator

        _revalidator = KeyframeRevalidator(
            bbox_repo=bbox_repo,
            threshold=settings.as_float("pipeline.min_keyframe_detection_confidence"),
        )
        app.state.keyframe_revalidator = _revalidator
        app.state.keyframe_revalidator_task = asyncio.create_task(_revalidator.run())

    # NOTE(N0): corrections and live router wiring removed since their
    # Postgres repositories (tracking_repo, global_track_repo, dnf_repo)
    # were deleted. These routers need PH-native replacements (N1+N2+N3).

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
        dashboard_router_mod.set_bbox_repo(bbox_repo)  # type: ignore[arg-type]

    yield

    # -------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------
    # M3: stop keyframe revalidator.
    _m3_revalidator = getattr(app.state, "keyframe_revalidator", None)
    if _m3_revalidator is not None:
        await _m3_revalidator.stop()

    # WTR2: stop CC assertion subscriber.
    if cc_assertion_subscriber is not None:
        await cc_assertion_subscriber.stop()  # type: ignore[attr-defined]

    # WTR5: disconnect room transition publisher.
    if "_room_transition_pub" in dir():
        await _room_transition_pub.disconnect()

    # M2: stop CC config sync before pipeline.
    _cc_sync = getattr(app.state, "cc_sync_service", None)
    if _cc_sync is not None:
        await _cc_sync.stop()

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
    app.include_router(ph_router)
    app.include_router(trajectory_router)

    @app.get("/health")
    async def health() -> dict[str, object]:
        """WTR10: Rollout health check — reports subsystem readiness.

        Returns 'healthy' only when all required subsystems are operational.
        Returns 'degraded' when optional subsystems are missing.
        Returns 'unhealthy' when required subsystems are absent.
        """
        checks: dict[str, dict[str, object]] = {}
        overall = "healthy"

        # Pipeline
        pipeline_ok = _pipeline is not None and _pipeline.is_running
        checks["pipeline"] = {
            "status": "ok" if pipeline_ok else "degraded",
            "message": "running" if pipeline_ok else "stopped or not initialized",
        }
        if not pipeline_ok and _pipeline is None:
            overall = "degraded"

        # PH repository
        _ph_repo_ref = _pipeline._ph_repo if _pipeline else None
        ph_repo_ok = _ph_repo_ref is not None
        checks["ph_repository"] = {
            "status": "ok" if ph_repo_ok else "degraded",
            "message": "wired" if ph_repo_ok else "not configured",
        }

        # Observation repository (wired through pipeline)
        obs_repo_ok = _pipeline is not None and _pipeline._obs_repo is not None
        checks["observation_repository"] = {
            "status": "ok" if obs_repo_ok else "degraded",
            "message": "wired" if obs_repo_ok else "not configured",
        }

        # Database pool
        db_ok = _pool is not None
        checks["database"] = {
            "status": "ok" if db_ok else "degraded",
            "message": "connected" if db_ok else "in-memory mode (state lost on restart)",
        }

        # Identity resolver (wired through pipeline)
        resolver_ok = _pipeline is not None and _pipeline._identity_resolver is not None
        checks["identity_resolver"] = {
            "status": "ok" if resolver_ok else "degraded",
            "message": "wired" if resolver_ok else "not configured",
        }

        # World tracker
        world_tracker_ok = _pipeline is not None and _pipeline._world_tracker is not None
        checks["world_tracker"] = {
            "status": "ok" if world_tracker_ok else "degraded",
            "message": "wired" if world_tracker_ok else "not configured",
        }

        # Calibration sync
        cc_sync = getattr(app.state, "cc_sync_service", None)
        calibration_ok = cc_sync is not None
        checks["calibration_sync"] = {
            "status": "ok" if calibration_ok else "degraded",
            "message": "running" if calibration_ok else "not configured",
        }

        # Transit detector (WTR5)
        transit_ok = getattr(app.state, "cc_sync_service", None) is not None
        checks["transit_detector"] = {
            "status": "ok" if transit_ok else "degraded",
            "message": "configured" if transit_ok else "transit zones not loaded",
        }

        # Face ID
        face_id_ok = _pipeline is not None and _pipeline._face_id_client is not None
        checks["face_identity"] = {
            "status": "ok" if face_id_ok else "info",
            "message": "configured" if face_id_ok else "not configured (optional)",
        }

        return {
            "status": overall,
            "service": "tracking-orchestrator",
            "version": "0.1.0",
            "checks": checks,
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
