"""FP1: Co-present endpoint tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.domain import PersonHypothesis
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


def _make_ph(
    ph_id: str,
    identity_id: str | None = None,
    *,
    last_seen_offset_s: float = 0,
    closed: bool = False,
    x_m: float = 1.0,
    y_m: float = 2.0,
) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(x_m, y_m, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now - timedelta(minutes=30),
        last_seen_at=now - timedelta(seconds=last_seen_offset_s),
        last_seen_camera="cam-1",
        observation_count=15,
        current_identity_id=identity_id,
        current_identity_committed_at=now if identity_id else None,
        active_cameras=frozenset(["cam-1"]),
        closed_at=now if closed else None,
    )


class TestPHCoPresent:
    @pytest.mark.asyncio
    async def test_co_present_returns_overlapping_ph(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", last_seen_offset_s=0))
        await repo.save(_make_ph("ph-2", last_seen_offset_s=5))
        resp = client.get("/ph/ph-1/co_present")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ph_id"] == "ph-1"
        assert len(data["co_present"]) == 1
        assert data["co_present"][0]["ph_id"] == "ph-2"

    @pytest.mark.asyncio
    async def test_co_present_excludes_non_overlapping(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", last_seen_offset_s=0))
        await repo.save(_make_ph("ph-2", last_seen_offset_s=120))
        resp = client.get("/ph/ph-1/co_present")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["co_present"]) == 0

    @pytest.mark.asyncio
    async def test_co_present_excludes_phs_outside_radius(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", last_seen_offset_s=0, x_m=1.0, y_m=2.0))
        await repo.save(_make_ph("ph-2", last_seen_offset_s=5, x_m=20.0, y_m=2.0))

        resp = client.get("/ph/ph-1/co_present?radius_m=5")

        assert resp.status_code == 200
        data = resp.json()
        assert data["co_present"] == []

    @pytest.mark.asyncio
    async def test_co_present_excludes_self(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", last_seen_offset_s=0))
        resp = client.get("/ph/ph-1/co_present")
        assert resp.status_code == 200
        data = resp.json()
        for item in data["co_present"]:
            assert item["ph_id"] != "ph-1"

    @pytest.mark.asyncio
    async def test_co_present_excludes_closed_phs(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", last_seen_offset_s=0))
        await repo.save(_make_ph("ph-2", closed=True, last_seen_offset_s=5))
        resp = client.get("/ph/ph-1/co_present")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["co_present"]) == 0

    @pytest.mark.asyncio
    async def test_co_present_empty_when_nobody_else(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1"))
        resp = client.get("/ph/ph-1/co_present")
        assert resp.status_code == 200
        data = resp.json()
        assert data["co_present"] == []
