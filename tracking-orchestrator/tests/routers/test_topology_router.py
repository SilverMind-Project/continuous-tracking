"""Learned camera transitions are readable for the CC admin."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import CameraTopologyEdge
from app.routers.topology import get_topology_repo, router
from app.storage.base import InMemoryCameraTopologyRepository


@pytest.fixture
def repo() -> InMemoryCameraTopologyRepository:
    return InMemoryCameraTopologyRepository()


@pytest.fixture
def client(repo: InMemoryCameraTopologyRepository) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_topology_repo] = lambda: repo
    yield TestClient(app)


@pytest.mark.asyncio
async def test_transitions_sorted_by_observations(
    repo: InMemoryCameraTopologyRepository, client: TestClient
) -> None:
    at = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
    await repo.upsert_edge(CameraTopologyEdge("cam-a", "cam-b", 2, 4.0, 1.0, at))
    await repo.upsert_edge(CameraTopologyEdge("cam-b", "cam-a", 9, 6.0, 4.0, at))

    body = client.get("/internal/topology/transitions").json()

    first, second = body["transitions"]
    assert (first["from_camera"], first["to_camera"], first["observation_count"]) == (
        "cam-b",
        "cam-a",
        9,
    )
    assert first["std_transit_s"] == pytest.approx(2.0)
    assert second["observation_count"] == 2


def test_no_transitions_learned_yet(client: TestClient) -> None:
    assert client.get("/internal/topology/transitions").json()["transitions"] == []
