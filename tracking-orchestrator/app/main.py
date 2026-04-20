"""FastAPI application factory for the tracking orchestrator.

Mirrors the cognitive-companion pattern: create_app() returns the app,
lifespan manages service lifecycle.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize DB connections, Redis, load camera configs
    yield
    # Shutdown: drain queues, close connections


def create_app() -> FastAPI:
    app = FastAPI(
        title="Continuous Tracking — Orchestrator",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "tracking-orchestrator", "version": "0.1.0"}

    return app
