"""Tests for the live-view internal router."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import PersonHypothesis
from app.routers.live import router as live_router
from app.routers.live import set_context
from app.storage.base import InMemoryPHRepository


@pytest.fixture
def client():
    ph_repo = InMemoryPHRepository()

    import asyncio

    async def _seed() -> None:
        now = datetime.now(UTC)
        ph = PersonHypothesis(
            ph_id="gt-alpha",
            state_mean=(1.0, 1.0, 0.0, 0.0),
            state_cov=tuple([0.1] * 16),
            born_at=now,
            last_seen_at=now,
            last_seen_camera="kitchen-1",
            observation_count=5,
            current_identity_id="grandma",
            active_cameras=frozenset(["kitchen-1"]),
        )
        await ph_repo.save(ph)

    asyncio.run(_seed())
    set_context(ph_repo=ph_repo)

    app = FastAPI()
    app.include_router(live_router)
    return TestClient(app)


def test_health_ok(client: TestClient):
    resp = client.get("/internal/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_features_returns_flags(client: TestClient):
    resp = client.get("/internal/features")
    assert resp.status_code == 200
    flags = resp.json()["flags"]
    assert "retroactive_revision_enabled" in flags
