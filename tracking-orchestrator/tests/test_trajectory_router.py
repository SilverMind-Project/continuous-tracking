"""Tests for the trajectory read router."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.domain import RoomDwell
from app.main import create_app
from app.routers import trajectory as trajectory_router_mod
from app.storage.base import InMemoryTrajectoryRepository


def _make_client() -> TestClient:
    app = create_app()
    return TestClient(app)


class TestTrajectoryRouter:
    def test_recent_returns_points(self) -> None:
        client = _make_client()
        resp = client.get("/internal/trajectory/recent", params={"limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert "points" in data
        assert "count" in data

    def test_recent_with_identity_filter(self) -> None:
        client = _make_client()
        resp = client.get(
            "/internal/trajectory/recent",
            params={"identity_id": "alice", "limit": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0  # InMemory repo starts empty

    def test_recent_with_since(self) -> None:
        client = _make_client()
        since = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
        resp = client.get("/internal/trajectory/recent", params={"since": since, "limit": 10})
        assert resp.status_code == 200


class TestDwellRangeEndpoint:
    """GET /internal/trajectory/dwells (identity-continuity M04)."""

    def test_happy_path_returns_dwells_in_range(self) -> None:
        repo = InMemoryTrajectoryRepository()
        trajectory_router_mod.set_context(trajectory_repo=repo)
        t0 = datetime(2026, 7, 20, 8, 0, 0, tzinfo=UTC)
        dwell = RoomDwell(
            dwell_id="d-1",
            identity_id="alice",
            ph_id="ph-1",
            room_name="living_room",
            entered_at=t0 + timedelta(minutes=10),
            exited_at=t0 + timedelta(minutes=20),
            entry_confidence=0.8,
        )
        repo._closed_dwells.append(dwell)  # test-only direct seed

        client = _make_client()
        resp = client.get(
            "/internal/trajectory/dwells",
            params={
                "ph_id": "ph-1",
                "start": t0.isoformat(),
                "end": (t0 + timedelta(hours=1)).isoformat(),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["dwells"]) == 1
        assert data["dwells"][0]["room_name"] == "living_room"
        assert data["dwells"][0]["identity_id"] == "alice"
        assert data["dwells"][0]["ph_id"] == "ph-1"

    def test_empty_range_returns_no_dwells(self) -> None:
        repo = InMemoryTrajectoryRepository()
        trajectory_router_mod.set_context(trajectory_repo=repo)
        t0 = datetime(2026, 7, 20, 8, 0, 0, tzinfo=UTC)

        client = _make_client()
        resp = client.get(
            "/internal/trajectory/dwells",
            params={
                "ph_id": "ph-does-not-exist",
                "start": t0.isoformat(),
                "end": (t0 + timedelta(hours=1)).isoformat(),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["dwells"] == []

    def test_bad_params_missing_required_query_arg(self) -> None:
        client = _make_client()
        resp = client.get("/internal/trajectory/dwells", params={"ph_id": "ph-1"})
        assert resp.status_code == 422

    def test_bad_params_unparseable_datetime(self) -> None:
        client = _make_client()
        resp = client.get(
            "/internal/trajectory/dwells",
            params={"ph_id": "ph-1", "start": "not-a-date", "end": "also-not-a-date"},
        )
        assert resp.status_code == 422
