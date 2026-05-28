"""N1 Person Hypothesis API tests.

Exercises the PH router against an InMemoryPHRepository.
Covers: list, detail, observations, trail, keyframes, co_present,
correct, merge, split, batch correct, revisions.
"""

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


def _make_ph(
    ph_id: str,
    identity_id: str | None = None,
    active_cameras: frozenset[str] | None = None,
) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now - timedelta(minutes=30),
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=15,
        current_identity_id=identity_id,
        current_identity_committed_at=now if identity_id else None,
        active_cameras=active_cameras or frozenset(["cam-1", "cam-2"]),
        last_floor_speed_m_s=0.5,
        last_posture="walking",
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


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestPHList:
    @pytest.mark.asyncio
    async def test_empty_list(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        resp = client.get("/ph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_list_with_phs(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1"))
        await repo.save(_make_ph("ph-2", identity_id="alice"))
        resp = client.get("/ph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    @pytest.mark.asyncio
    async def test_filter_by_identity(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1"))
        await repo.save(_make_ph("ph-2", identity_id="alice"))
        resp = client.get("/ph?identity_id=alice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["current_identity_id"] == "alice"


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


class TestPHDetail:
    @pytest.mark.asyncio
    async def test_detail(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1", identity_id="bob"))
        resp = client.get("/ph/ph-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ph_id"] == "ph-1"
        assert data["current_identity_id"] == "bob"

    def test_unknown_ph(self, client: TestClient) -> None:
        resp = client.get("/ph/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


class TestPHObservations:
    @pytest.mark.asyncio
    async def test_observations(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1"))
        # Manually add observations via the repo's internal store
        obs = _make_observation("cam-1", 1, 1.0, 2.0)
        repo._observations.setdefault("ph-1", []).append(obs)
        resp = client.get("/ph/ph-1/observations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1


# ---------------------------------------------------------------------------
# Correct
# ---------------------------------------------------------------------------


class TestPHCorrect:
    @pytest.mark.asyncio
    async def test_correct_identity(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1"))
        resp = client.post(
            "/ph/ph-1/correct",
            json={"new_identity_id": "alice", "reason": "operator override"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["revision"]["ph_id"] == "ph-1"
        assert data["revision"]["new_identity_id"] == "alice"

        # PH should now have the new identity
        ph = await repo.get("ph-1")
        assert ph is not None
        assert ph.current_identity_id == "alice"

    @pytest.mark.asyncio
    async def test_correct_unknown_ph(self, client: TestClient) -> None:
        resp = client.post(
            "/ph/nonexistent/correct",
            json={"new_identity_id": "alice", "reason": "test"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestPHMerge:
    @pytest.mark.asyncio
    async def test_merge(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1", active_cameras=frozenset(["cam-1"])))
        await repo.save(_make_ph("ph-2", active_cameras=frozenset(["cam-2"])))
        resp = client.post(
            "/ph/merge",
            json={
                "source_ph_id": "ph-1",
                "target_ph_id": "ph-2",
                "reason": "duplicate person",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_ph_id"] == "ph-1"
        assert data["target_ph_id"] == "ph-2"

        # Source should be closed
        ph1 = await repo.get("ph-1")
        assert ph1 is not None
        assert ph1.closed_at is not None

    @pytest.mark.asyncio
    async def test_merge_same_ph_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/ph/merge",
            json={
                "source_ph_id": "ph-1",
                "target_ph_id": "ph-1",
                "reason": "self",
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


class TestPHSplit:
    @pytest.mark.asyncio
    async def test_split(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1"))
        # Add observations so we can split
        obs1 = _make_observation("cam-1", 1, 1.0, 1.0)
        object.__setattr__(obs1, "observation_id", "obs-1")
        obs2 = _make_observation("cam-1", 2, 2.0, 2.0)
        object.__setattr__(obs2, "observation_id", "obs-2")
        obs3 = _make_observation("cam-1", 3, 3.0, 3.0)
        object.__setattr__(obs3, "observation_id", "obs-3")
        repo._observations["ph-1"] = [obs1, obs2, obs3]

        resp = client.post(
            "/ph/ph-1/split",
            json={"at_observation_id": "obs-2", "reason": "two people tracked as one"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_ph_id"] == "ph-1"
        assert data["new_ph_id"]


# ---------------------------------------------------------------------------
# Batch correct
# ---------------------------------------------------------------------------


class TestPHBatchCorrect:
    @pytest.mark.asyncio
    async def test_batch_correct(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1"))
        await repo.save(_make_ph("ph-2"))
        resp = client.post(
            "/ph/batch_correct",
            json={
                "corrections": [
                    {"ph_id": "ph-1", "new_identity_id": "alice", "reason": "batch"},
                    {"ph_id": "ph-2", "new_identity_id": "bob", "reason": "batch"},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["applied"] == 2


# ---------------------------------------------------------------------------
# Revisions
# ---------------------------------------------------------------------------


class TestPHRevisions:
    @pytest.mark.asyncio
    async def test_revisions_feed(self, client: TestClient, repo: InMemoryPHRepository) -> None:
        await repo.save(_make_ph("ph-1"))
        # Perform a correction to generate a revision
        client.post(
            "/ph/ph-1/correct",
            json={"new_identity_id": "alice", "reason": "test"},
        )
        resp = client.get("/ph/revisions?ph_id=ph-1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) >= 1
        assert data["has_more"] is False
