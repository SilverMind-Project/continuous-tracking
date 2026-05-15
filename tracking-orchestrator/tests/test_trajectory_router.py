"""Tests for the trajectory read router."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.domain import PersonTrajectoryPoint
from app.main import create_app
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
        resp = client.get(
            "/internal/trajectory/recent", params={"since": since, "limit": 10}
        )
        assert resp.status_code == 200
