"""Tests for the manual-override corrections router."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import GlobalTrack
from app.routers.corrections import router as corrections_router
from app.routers.corrections import set_context
from app.storage.base import (
    InMemoryGlobalTrackRepository,
    InMemoryTrackingRepository,
)


class _FakePublisher:
    """In-memory stand-in for :class:`RevisionPublisher`."""

    def __init__(self) -> None:
        self.published: list = []
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def publish(self, revision) -> str:
        self.published.append(revision)
        return f"msg-{len(self.published)}"


@pytest.fixture
def client_and_publisher(monkeypatch):
    tracking = InMemoryTrackingRepository()
    gtr = InMemoryGlobalTrackRepository()
    pub = _FakePublisher()

    # Seed an active global track with a committed identity.
    import asyncio

    async def _seed() -> None:
        track = GlobalTrack(
            global_track_id="gt-1",
            camera_ids=["kitchen-1"],
            tracklet_ids=["t-1"],
            started_at=datetime.now(UTC),
            last_seen_at=datetime.now(UTC),
            current_identity_id="grandma",
        )
        await gtr.save(track)

    asyncio.run(_seed())

    set_context(tracking_repo=tracking, global_track_repo=gtr, publisher=pub)

    app = FastAPI()
    app.include_router(corrections_router)
    return TestClient(app), pub, tracking, gtr


def test_correction_applies_and_publishes(client_and_publisher):
    client, pub, _tracking, _gtr = client_and_publisher
    resp = client.post(
        "/internal/corrections",
        json={
            "global_track_id": "gt-1",
            "new_identity_id": "grandpa",
            "actor": "caregiver@home",
            "reason": "manual",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["previous_identity_id"] == "grandma"
    assert body["new_identity_id"] == "grandpa"
    assert body["revision_id"]

    # Revision was published.
    assert len(pub.published) == 1
    rev = pub.published[0]
    assert rev.new_identity_id == "grandpa"
    assert rev.previous_identity_id == "grandma"
    assert rev.reason == "manual"
    assert rev.evidence["actor"] == "caregiver@home"


def test_correction_unknown_track_returns_404(client_and_publisher):
    client, *_ = client_and_publisher
    resp = client.post(
        "/internal/corrections",
        json={
            "global_track_id": "does-not-exist",
            "new_identity_id": "grandpa",
            "actor": "caregiver@home",
        },
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "global_track.not_found"


def test_clearing_identity_publishes_unknown_revision(client_and_publisher):
    client, pub, *_ = client_and_publisher
    resp = client.post(
        "/internal/corrections",
        json={
            "global_track_id": "gt-1",
            "new_identity_id": None,
            "actor": "caregiver@home",
        },
    )
    assert resp.status_code == 200
    rev = pub.published[0]
    assert rev.new_identity_id is None
    assert rev.map_identity_id == "UNKNOWN"


def test_disconnected_publisher_still_returns_200(client_and_publisher):
    client, pub, _tracking, _gtr = client_and_publisher
    pub._connected = False
    resp = client.post(
        "/internal/corrections",
        json={
            "global_track_id": "gt-1",
            "new_identity_id": "grandpa",
            "actor": "caregiver@home",
        },
    )
    assert resp.status_code == 200
    assert not pub.published  # never published because disconnected
