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
from .inference.detector import PersonDetector
from .inference.reid_embedder import ReidEmbedder
from .inference.triton_client import TritonGrpcClient
from .pipeline import FrameProcessingPipeline
from .pipeline.frame_pipeline import FaceIdCameraConfig, PipelineConfig
from .routers import corrections as corrections_router_mod
from .routers import dashboard as dashboard_router_mod
from .routers import gallery as gallery_router_mod
from .routers import live as live_router_mod
from .routers.calibration import router as calibration_router
from .routers.corrections import router as corrections_router
from .routers.dashboard import router as dashboard_router
from .routers.gallery import router as gallery_router
from .routers.live import router as live_router
from .routers.trajectory import router as trajectory_router
from .routers.trajectory import set_context as set_trajectory_context
from .storage.migrations import MigrationRunner
from .storage.postgres.gallery_repo import PostgresGalleryRepository
from .storage.postgres.global_track_repo import PostgresGlobalTrackRepository
from .storage.postgres.keyframe_repo import PostgresKeyframeRepository
from .storage.postgres.settings_repo import PostgresSettingsRepository
from .storage.postgres.signal_repo import PostgresDementiaSignalRepository
from .storage.postgres.tracking_repo import PostgresTrackingRepository
from .storage.postgres.trajectory_repo import PostgresTrajectoryRepository
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


async def _fetch_cc_camera_configs(cc_url: str, api_key: str = "") -> dict[str, FaceIdCameraConfig]:
    """Fetch per-camera face-id configs from cognitive-companion's camera API.

    Returns an empty dict if CC is unreachable (graceful degradation).
    CC is the primary source for per-camera face-id settings (enabled flag
    and min_confidence). The ``FACE_ID_CAMERA_CONFIDENCE`` env var provides
    a deployment-level fallback for cameras not in CC.
    """
    if not cc_url:
        return {}
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; cannot fetch camera configs from CC")
        return {}

    url = cc_url.rstrip("/") + "/api/v1/cts/cameras"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            cameras: list[dict[str, Any]] = resp.json()
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
    face_id_camera_configs = await _fetch_cc_camera_configs(cc_url, cc_api_key)

    config = PipelineConfig(
        transport=TransportConfig(redis_url=redis_url),
        face_id_url=face_id_url,
        face_id_cooldown_s=float(settings.get("face_id.cooldown_s", "5.0")),
        face_id_timeout_s=float(settings.get("face_id.timeout_s", "2.0")),
        face_id_min_confidence=float(settings.get("face_id.min_confidence", "0.4")),
        face_id_enabled=bool(face_id_url),
        face_id_camera_configs=face_id_camera_configs,
        timezone=settings.get("app.timezone", "UTC"),
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
    else:
        tracking_repo = None
        gallery_repo = None
        global_track_repo = None
        trajectory_repo = None
        keyframe_repo = None
        signal_repo = None
        settings_repo = None

    # -- Triton --
    triton_url = settings.get("triton.url", "")
    detector = None
    reid_embedder = None
    env = settings.get("env", "production")
    if triton_url:
        try:
            triton_timeout_ms = int(settings.get("triton.timeout_ms", "5000") or "5000")
            _triton_client = TritonGrpcClient(triton_url, timeout_ms=triton_timeout_ms)
            await _triton_client.__aenter__()
            detector = PersonDetector(
                _triton_client,
                conf_threshold=float(settings.get("pipeline.detector_confidence", "0.25")),
            )
            reid_embedder = ReidEmbedder(_triton_client)
        except Exception:
            logger.exception("Failed to connect to Triton Inference Server")
            if env in ("production", "staging"):
                raise RuntimeError(
                    "Triton Inference Server is required in production/staging. "
                    "Set PIPELINE_ALLOW_SKELETON=true to override for testing."
                ) from None
            _triton_client = None
    elif env in ("production", "staging"):
        allow_skeleton = settings.get("pipeline.allow_skeleton", "false")
        if str(allow_skeleton).lower() not in ("1", "true", "yes"):
            raise RuntimeError(
                "Triton Inference Server (TRITON_GRPC_URL) is required in production/staging. "
                "Set PIPELINE_ALLOW_SKELETON=true to override for testing."
            )

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

    # -- Wire everything and start --
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
    )
    await _pipeline.start()

    # Wire router modules to share the pipeline's repositories.
    if _pipeline.tracking_repo is not None and _pipeline.global_track_repo is not None:
        corrections_router_mod.set_context(
            tracking_repo=_pipeline.tracking_repo,
            global_track_repo=_pipeline.global_track_repo,
            publisher=_pipeline.revision_publisher,
        )
        live_router_mod.set_context(
            global_track_repo=_pipeline.global_track_repo,
            keyframe_repo=keyframe_repo,
            gallery_repo=gallery_repo,
        )

    if trajectory_repo is not None:
        set_trajectory_context(trajectory_repo=trajectory_repo)

    if gallery_repo is not None:
        gallery_router_mod.set_context(gallery_repo=gallery_repo)

    if signal_repo is not None and trajectory_repo is not None and keyframe_repo is not None:
        dashboard_router_mod.set_repos(
            signal=signal_repo,
            trajectory=trajectory_repo,
            keyframe=keyframe_repo,
        )

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
