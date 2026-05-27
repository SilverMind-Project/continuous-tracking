"""FP1: PH list filter parameter tests."""

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
    born_offset_s: int = 1800,
    closed: bool = False,
    metadata: dict | None = None,
) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now - timedelta(seconds=born_offset_s),
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=15,
        current_identity_id=identity_id,
        current_identity_committed_at=now if identity_id else None,
        active_cameras=frozenset(["cam-1"]),
        closed_at=now if closed else None,
        metadata=metadata or {},
    )


class TestPHListFilters:
    @pytest.mark.asyncio
    async def test_list_phs_no_filters_returns_all(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1"))
        await repo.save(_make_ph("ph-2", identity_id="alice"))
        resp = client.get("/ph?include_transient=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_list_phs_filter_by_state_active(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1"))
        await repo.save(_make_ph("ph-2", closed=True))
        resp = client.get("/ph?state=active&include_transient=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["ph_id"] == "ph-1"

    @pytest.mark.asyncio
    async def test_list_phs_filter_by_state_ended(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1"))
        await repo.save(_make_ph("ph-2", closed=True))
        resp = client.get("/ph?state=ended&include_transient=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["ph_id"] == "ph-2"

    @pytest.mark.asyncio
    async def test_list_phs_filter_by_until(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        now = datetime.now(UTC)
        await repo.save(_make_ph("ph-old", born_offset_s=7200))
        await repo.save(_make_ph("ph-new", born_offset_s=60))
        until_str = (now - timedelta(seconds=3600)).strftime("%Y-%m-%dT%H:%M:%SZ")
        resp = client.get(f"/ph?until={until_str}&include_transient=true")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_phs_filter_by_min_duration(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", born_offset_s=60))
        resp = client.get("/ph?min_duration_s=3600")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_phs_filter_search_matches_display_name(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", metadata={"display_name": "Alice"}))
        await repo.save(_make_ph("ph-2", metadata={"display_name": "Bob"}))
        resp = client.get("/ph?search=Ali&include_transient=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["ph_id"] == "ph-1"

    @pytest.mark.asyncio
    async def test_list_phs_filter_search_no_match_returns_empty(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", metadata={"display_name": "Alice"}))
        resp = client.get("/ph?search=XYZ&include_transient=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_phs_include_transient_false_excludes_short_lived(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", born_offset_s=0))
        resp = client.get("/ph?include_transient=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_phs_include_transient_true_includes_short_lived(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", born_offset_s=0))
        resp = client.get("/ph?include_transient=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_list_phs_pagination_limit_offset(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        for i in range(5):
            await repo.save(_make_ph(f"ph-{i}"))
        resp = client.get("/ph?limit=2&offset=1&include_transient=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2

    @pytest.mark.asyncio
    async def test_list_phs_combined_filters(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", identity_id="alice"))
        await repo.save(_make_ph("ph-2", identity_id="bob"))
        resp = client.get("/ph?identity_id=alice&state=active&include_transient=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["ph_id"] == "ph-1"
