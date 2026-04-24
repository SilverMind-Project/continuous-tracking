"""Tests for the live-view internal router."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import GlobalTrack
from app.routers.live import router as live_router
from app.routers.live import set_context
from app.storage.base import InMemoryGlobalTrackRepository


@pytest.fixture
def client():
    gtr = InMemoryGlobalTrackRepository()

    import asyncio

    async def _seed() -> None:
        await gtr.save(
            GlobalTrack(
                global_track_id="gt-alpha",
                camera_ids=["kitchen-1"],
                tracklet_ids=["t-1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                current_identity_id="grandma",
            )
        )

    asyncio.run(_seed())
    set_context(global_track_repo=gtr)

    app = FastAPI()
    app.include_router(live_router)
    return TestClient(app)


def test_global_tracks_returns_seeded_track(client: TestClient):
    resp = client.get("/internal/global_tracks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["tracks"][0]["current_identity_id"] == "grandma"


def test_health_ok(client: TestClient):
    resp = client.get("/internal/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_features_returns_flags(client: TestClient):
    resp = client.get("/internal/features")
    assert resp.status_code == 200
    flags = resp.json()["flags"]
    assert "retroactive_revision_enabled" in flags


def test_global_track_404_on_missing(client: TestClient):
    resp = client.get("/internal/global_tracks/missing")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "global_track.not_found"
