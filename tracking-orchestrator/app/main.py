"""FastAPI application factory for the tracking orchestrator.

Mirrors the cognitive-companion pattern: create_app() returns the app,
lifespan manages service lifecycle (pipeline start/stop, DB migrations).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from structlog import get_logger

from .inference.detector import PersonDetector
from .inference.reid_embedder import ReidEmbedder
from .inference.triton_client import TritonGrpcClient
from .pipeline import FrameProcessingPipeline
from .pipeline.frame_pipeline import PipelineConfig
from .routers import corrections as corrections_router_mod
from .routers import live as live_router_mod
from .routers.calibration import router as calibration_router
from .routers.corrections import router as corrections_router
from .routers.dashboard import router as dashboard_router
from .routers.live import router as live_router
from .storage.migrations import MigrationRunner
from .storage.postgres.gallery_repo import PostgresGalleryRepository
from .storage.postgres.global_track_repo import PostgresGlobalTrackRepository
from .storage.postgres.keyframe_repo import PostgresKeyframeRepository
from .storage.postgres.tracking_repo import PostgresTrackingRepository
from .storage.postgres.trajectory_repo import PostgresTrajectoryRepository
from .transport.minio_frames import MinioFrameConfig, MinioFrameFetcher

logger = get_logger(__name__)

# Module-level pipeline singleton, initialized in lifespan.
_pipeline: FrameProcessingPipeline | None = None

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _normalize_dsn(dsn: str) -> str:
    """Strip ``+asyncpg`` SQLAlchemy scheme prefix if present."""
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("+asyncpg", "", 1)
    return dsn


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

    # Startup: initialize pipeline and start background tasks
    config = PipelineConfig()
    _pipeline = FrameProcessingPipeline(config)

    # Read database URL from environment, not from never-populated app.state.config.
    database_url = os.environ.get("DATABASE_URL") or os.environ.get("CTS_DATABASE_URL")

    if not database_url:
        env = os.environ.get("CTS_ENV", "")
        if env not in ("dev", "development", "test"):
            logger.warning(
                "DATABASE_URL/CTS_DATABASE_URL not set; running with in-memory storage. "
                "State will be lost on restart.",
            )
        else:
            logger.info("DATABASE_URL not set; using in-memory storage for this environment.")

    if database_url:
        import asyncpg  # type: ignore[import-untyped]

        _pool = await asyncpg.create_pool(
            dsn=_normalize_dsn(database_url),
            min_size=2,
            max_size=20,
            command_timeout=10.0,
        )
        runner = MigrationRunner(_pool, MIGRATIONS_DIR)
        await runner.migrate()
        tracking_repo = PostgresTrackingRepository(_pool)
        gallery_repo = PostgresGalleryRepository(_pool)
        global_track_repo = PostgresGlobalTrackRepository(_pool)
        trajectory_repo = PostgresTrajectoryRepository(_pool)
        keyframe_repo = PostgresKeyframeRepository(_pool)
    else:
        tracking_repo = None
        gallery_repo = None
        global_track_repo = None
        trajectory_repo = None
        keyframe_repo = None

    triton_url = os.environ.get("TRITON_GRPC_URL") or os.environ.get("TRITON_URL")
    detector = None
    reid_embedder = None
    if triton_url:
        _triton_client = TritonGrpcClient(triton_url)
        await _triton_client.__aenter__()
        detector = PersonDetector(
            _triton_client,
            conf_threshold=config.detector_confidence,
        )
        reid_embedder = ReidEmbedder(_triton_client)

    minio_endpoint = os.environ.get("MINIO_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL")
    minio_bucket = os.environ.get("MINIO_BUCKET") or os.environ.get("CTS_FRAME_BUCKET")
    minio_access_key = os.environ.get("MINIO_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    minio_secret_key = os.environ.get("MINIO_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if minio_endpoint and minio_bucket and minio_access_key and minio_secret_key:
        _frame_fetcher = MinioFrameFetcher(
            MinioFrameConfig(
                endpoint_url=minio_endpoint,
                bucket=minio_bucket,
                access_key_id=minio_access_key,
                secret_access_key=minio_secret_key,
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                secure=os.environ.get("MINIO_SECURE", "0").lower() in {"1", "true", "yes"},
            )
        )
        await _frame_fetcher.connect()
    elif detector is not None:
        logger.warning(
            "Triton configured without MinIO frame storage; inference will run on blank frames.",
        )

    await _pipeline.initialize(
        detector=detector,
        tracking_repo=tracking_repo,
        gallery_repo=gallery_repo,
        global_track_repo=global_track_repo,
        trajectory_repo=trajectory_repo,
        keyframe_repo=keyframe_repo,
        frame_fetcher=_frame_fetcher,
        reid_embedder=reid_embedder,
    )
    await _pipeline.start()

    # Wire the corrections + live routers to share the pipeline's repositories
    # and revision publisher so manual overrides produce real revisions on the
    # same stream the automatic ones do.
    if _pipeline.tracking_repo is not None and _pipeline.global_track_repo is not None:
        corrections_router_mod.set_context(
            tracking_repo=_pipeline.tracking_repo,
            global_track_repo=_pipeline.global_track_repo,
            publisher=_pipeline.revision_publisher,
        )
        live_router_mod.set_context(global_track_repo=_pipeline.global_track_repo)

    yield

    # Shutdown: stop pipeline and close the DB pool.
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
    app.include_router(live_router)

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
