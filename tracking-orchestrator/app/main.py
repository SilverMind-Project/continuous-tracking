"""FastAPI application factory for the tracking orchestrator.

Mirrors the cognitive-companion pattern: create_app() returns the app,
lifespan manages service lifecycle.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup order follows phase-0 section 0.6:
    # database pools, Redis clients, model/runtime handles, then routers.
    yield
    # Shutdown drains in reverse order and eventually adds the mTLS admin port.


def create_app() -> FastAPI:
    app = FastAPI(
        title="Continuous Tracking — Orchestrator",
        version="0.1.0",
        lifespan=lifespan,
    )
    # TODO(phase-0 0.27): add the separate internal admin API listener on :8310.

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "tracking-orchestrator", "version": "0.1.0"}

    return app
