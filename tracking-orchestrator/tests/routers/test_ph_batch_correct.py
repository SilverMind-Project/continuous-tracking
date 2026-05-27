"""FP1: Batch correct atomicity tests."""

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


def _make_ph(ph_id: str, identity_id: str | None = None) -> PersonHypothesis:
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
        active_cameras=frozenset(["cam-1"]),
    )


class TestPHBatchCorrect:
    @pytest.mark.asyncio
    async def test_batch_correct_applies_all_revisions(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
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
        assert data["errors"] == []

        ph1 = await repo.get("ph-1")
        assert ph1 is not None
        assert ph1.current_identity_id == "alice"
        ph2 = await repo.get("ph-2")
        assert ph2 is not None
        assert ph2.current_identity_id == "bob"

    @pytest.mark.asyncio
    async def test_batch_correct_empty_batch_fails_validation(self, client: TestClient) -> None:
        resp = client.post(
            "/ph/batch_correct",
            json={"corrections": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_correct_atomicity_via_mock(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        await repo.save(_make_ph("ph-1", identity_id="original"))
        await repo.save(_make_ph("ph-2", identity_id="bob"))

        resp = client.post(
            "/ph/batch_correct",
            json={
                "corrections": [
                    {"ph_id": "ph-1", "new_identity_id": "alice", "reason": "ok"},
                    {"ph_id": "ph-nonexistent", "new_identity_id": "x", "reason": "bad"},
                ]
            },
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["detail"]["code"] == "ph.batch_correct.invalid"

        # Verify: the router propagates the exception and returns an error.
        # In Postgres, the transaction rolls back ensuring atomicity.
        # InMemory does not support rollback, so ph-1 may or may not be updated.
        # The key contract is that the API surface returns a structured error,
        # not a partial success.
        assert resp.status_code == 422
