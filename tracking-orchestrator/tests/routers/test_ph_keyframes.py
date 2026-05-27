"""FP1: Keyframes endpoint tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.domain import (
    BoundingBox,
    FloorPoint,
    PersonHypothesis,
    WorldObservation,
)
from app.main import create_app
from app.routers.ph import set_ph_repository
from app.storage.base import InMemoryPHRepository


@pytest.fixture
def repo() -> InMemoryPHRepository:
    return InMemoryPHRepository()


@pytest.fixture
def client(repo: InMemoryPHRepository) -> TestClient:
    set_ph_repository(repo)
    app = create_app()
    return TestClient(app)


def _make_ph(ph_id: str) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now - timedelta(minutes=30),
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=15,
        current_identity_id=None,
        current_identity_committed_at=None,
        active_cameras=frozenset(["cam-1"]),
    )


def _make_observation(camera_id: str, frame_idx: int, x_m: float, y_m: float) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_idx,
        captured_at=datetime.now(UTC) - timedelta(seconds=frame_idx),
        floor_point=FloorPoint(int(x_m * 1000), int(y_m * 1000), calibrated=True),
        bbox=BoundingBox(10, 20, 30, 40),
        embedding=[0.0] * 4,
        detection_confidence=0.9,
    )


class TestPHKeyframes:
    @pytest.mark.asyncio
    async def test_get_keyframes_returns_empty_when_no_observations(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1"))
        resp = client.get("/ph/ph-1/keyframes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ph_id"] == "ph-1"
        assert data["items"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_get_keyframes_404_on_missing_ph(self, client: TestClient) -> None:
        resp = client.get("/ph/nonexistent/keyframes")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_keyframes_pagination(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1"))
        resp = client.get("/ph/ph-1/keyframes?limit=5&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ph_id"] == "ph-1"
        assert isinstance(data["items"], list)
        assert isinstance(data["count"], int)
