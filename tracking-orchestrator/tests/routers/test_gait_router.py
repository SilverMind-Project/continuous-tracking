"""Tests for the gait daily aggregate internal router."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.gait import router as gait_router
from app.routers.gait import set_context
from app.storage.gait import InMemoryGaitDailyRepository
from app.trajectory.gait import GaitDailyRecord


def _make_record(
    identity_id: str,
    local_date: date,
    median_speed_m_s: float = 0.9,
    bout_count: int = 5,
    total_walking_s: float = 120.0,
) -> GaitDailyRecord:
    return GaitDailyRecord(
        identity_id=identity_id,
        local_date=local_date,
        bout_count=bout_count,
        total_walking_s=total_walking_s,
        total_distance_m=total_walking_s * median_speed_m_s,
        median_speed_m_s=median_speed_m_s,
        mad_speed_m_s=0.05,
        p95_speed_m_s=median_speed_m_s + 0.2,
        sample_bout_ids=["bout-1"],
        computed_at=datetime.now(UTC),
    )


@pytest.fixture
def client():
    repo = InMemoryGaitDailyRepository()
    set_context(gait_daily_repo=repo)
    app = FastAPI()
    app.include_router(gait_router)
    return TestClient(app), repo


@pytest.mark.asyncio
async def test_happy_path_returns_rows(client):
    test_client, repo = client
    d1 = date(2026, 5, 1)
    d2 = date(2026, 5, 2)
    await repo.upsert_day(_make_record("alice", d1))
    await repo.upsert_day(_make_record("alice", d2))

    resp = test_client.get(
        "/internal/gait/daily",
        params={"identity_id": "alice", "since": "2026-04-01", "until": "2026-06-01"},
    )
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert rows[0]["identity_id"] == "alice"
    assert rows[0]["median_speed_m_s"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_empty_identity_returns_empty_list(client):
    test_client, _repo = client
    resp = test_client.get(
        "/internal/gait/daily",
        params={"identity_id": "nobody"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_date_filtering(client):
    test_client, repo = client
    await repo.upsert_day(_make_record("bob", date(2026, 3, 1)))
    await repo.upsert_day(_make_record("bob", date(2026, 5, 1)))
    await repo.upsert_day(_make_record("bob", date(2026, 6, 1)))

    resp = test_client.get(
        "/internal/gait/daily",
        params={"identity_id": "bob", "since": "2026-04-01", "until": "2026-05-31"},
    )
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["local_date"] == "2026-05-01"
