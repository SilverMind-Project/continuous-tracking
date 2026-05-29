"""Tests for the manual-override corrections router."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import PersonHypothesis
from app.routers.corrections import router as corrections_router
from app.routers.corrections import set_context
from app.storage.base import InMemoryPHRepository


class _FakePublisher:
    """In-memory stand-in for RevisionPublisher."""

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
def client_and_publisher():
    ph_repo = InMemoryPHRepository()
    pub = _FakePublisher()

    import asyncio

    async def _seed() -> None:
        now = datetime.now(UTC)
        ph = PersonHypothesis(
            ph_id="gt-1",
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

    set_context(ph_repo=ph_repo, publisher=pub)

    app = FastAPI()
    app.include_router(corrections_router)
    return TestClient(app), pub, ph_repo


def test_correction_applies_and_publishes(client_and_publisher):
    client, pub, _ph_repo = client_and_publisher
    resp = client.post(
        "/internal/corrections",
        json={
            "ph_id": "gt-1",
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
    assert rev.actor == "caregiver@home"


def test_correction_unknown_track_returns_404(client_and_publisher):
    client, *_ = client_and_publisher
    resp = client.post(
        "/internal/corrections",
        json={
            "ph_id": "does-not-exist",
            "new_identity_id": "grandpa",
            "actor": "caregiver@home",
        },
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "ph.not_found"


def test_clearing_identity_publishes_unknown_revision(client_and_publisher):
    client, pub, *_ = client_and_publisher
    resp = client.post(
        "/internal/corrections",
        json={
            "ph_id": "gt-1",
            "new_identity_id": None,
            "actor": "caregiver@home",
        },
    )
    assert resp.status_code == 200
    rev = pub.published[0]
    assert rev.new_identity_id is None
    assert rev.reason == "manual"


def test_disconnected_publisher_still_returns_200(client_and_publisher):
    client, pub, _ph_repo = client_and_publisher
    pub._connected = False
    resp = client.post(
        "/internal/corrections",
        json={
            "ph_id": "gt-1",
            "new_identity_id": "grandpa",
            "actor": "caregiver@home",
        },
    )
    assert resp.status_code == 200
    assert not pub.published  # never published because disconnected
