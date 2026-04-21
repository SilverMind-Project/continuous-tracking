"""FastAPI application factory for the tracking orchestrator.

Mirrors the cognitive-companion pattern: create_app() returns the app,
lifespan manages service lifecycle (pipeline start/stop).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .pipeline import FrameProcessingPipeline
from .pipeline.frame_pipeline import PipelineConfig
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
    # TODO(phase-0 0.27): add the separate internal admin API listener on :8310.

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
