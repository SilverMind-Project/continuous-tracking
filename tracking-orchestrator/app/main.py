"""FastAPI application factory for the tracking orchestrator.

Mirrors the cognitive-companion pattern: create_app() returns the app,
lifespan manages service lifecycle (pipeline start/stop).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .pipeline import FrameProcessingPipeline
from .pipeline.frame_pipeline import PipelineConfig
from .routers.calibration import router as calibration_router
from .routers.dashboard import router as dashboard_router
from .storage.postgres.tracking_repo import PostgresTrackingRepository

# Module-level pipeline singleton, initialized in lifespan.
_pipeline: FrameProcessingPipeline | None = None


def get_pipeline() -> FrameProcessingPipeline | None:
    """Access the running pipeline from dependency injection."""
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline

    # Startup: initialize pipeline and start background tasks
    config = PipelineConfig()
    _pipeline = FrameProcessingPipeline(config)

    # Optionally inject a Postgres repo if DATABASE_URL is set
    repo = None
    database_url = app.state.config.get("database_url") if hasattr(app.state, "config") else None
    if database_url:
        import asyncpg  # type: ignore[import-untyped]

        pool = await asyncpg.create_pool(dsn=database_url)
        repo = PostgresTrackingRepository(pool)

    await _pipeline.initialize(repo=repo)
    await _pipeline.start()

    yield

    # Shutdown: stop pipeline gracefully
    if _pipeline is not None:
        await _pipeline.stop()
        _pipeline = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Continuous Tracking — Orchestrator",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Internal calibration endpoints consumed by the CC BFF.
    # Phase-0 §0.27 will move these to a separate :8310 listener; for M7
    # they live on the same app under /internal/ to keep the routing simple.
    app.include_router(calibration_router)
    app.include_router(dashboard_router)

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

    return app
