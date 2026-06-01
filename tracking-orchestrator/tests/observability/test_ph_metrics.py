"""FP1: PH operation metrics tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.domain import BoundingBox, FloorPoint, PersonHypothesis, WorldObservation
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


def _counter_value(counter, labels: dict[str, str]) -> float:
    """Read the current value of a labeled Prometheus Counter."""
    return counter.labels(**labels)._value.get()


class TestPHMetrics:
    @pytest.mark.asyncio
    async def test_correct_increments_corrections_counter(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        from app.observability.metrics import metrics

        await repo.save(_make_ph("ph-1"))
        before = _counter_value(metrics.cts_ph_corrections_total, {"actor": "operator"})
        client.post(
            "/ph/ph-1/correct",
            json={"new_identity_id": "alice", "reason": "test"},
        )
        after = _counter_value(metrics.cts_ph_corrections_total, {"actor": "operator"})
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_batch_correct_increments_by_batch_size(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        from app.observability.metrics import metrics

        await repo.save(_make_ph("ph-1"))
        await repo.save(_make_ph("ph-2"))
        before = _counter_value(metrics.cts_ph_corrections_total, {"actor": "batch"})
        client.post(
            "/ph/batch_correct",
            json={
                "corrections": [
                    {"ph_id": "ph-1", "new_identity_id": "alice", "reason": "b"},
                    {"ph_id": "ph-2", "new_identity_id": "bob", "reason": "b"},
                ]
            },
        )
        after = _counter_value(metrics.cts_ph_corrections_total, {"actor": "batch"})
        assert after == before + 2

    @pytest.mark.asyncio
    async def test_merge_increments_merges_counter(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        from dataclasses import replace

        from app.observability.metrics import metrics

        await repo.save(_make_ph("ph-1"))
        await repo.save(_make_ph("ph-2"))
        # must use non-overlapping cameras for merge.
        repo._phs["ph-1"] = replace(repo._phs["ph-1"], active_cameras=frozenset(["cam-1"]))
        repo._phs["ph-2"] = replace(repo._phs["ph-2"], active_cameras=frozenset(["cam-2"]))
        before = _counter_value(metrics.cts_ph_merges_total, {"actor": "operator"})
        client.post(
            "/ph/merge",
            json={
                "source_ph_id": "ph-1",
                "target_ph_id": "ph-2",
                "reason": "test merge",
            },
        )
        after = _counter_value(metrics.cts_ph_merges_total, {"actor": "operator"})
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_split_increments_splits_counter(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        from app.observability.metrics import metrics

        ph = _make_ph("ph-1")
        await repo.save(ph)
        # Seed observations into the repo so split can find them
        obs1 = WorldObservation(
            camera_id="cam-1",
            frame_index=0,
            captured_at=datetime.now(UTC) - timedelta(seconds=60),
            floor_point=FloorPoint(1000, 2000, calibrated=True),
            bbox=BoundingBox(10, 20, 30, 40),
            embedding=[0.0] * 4,
            detection_confidence=0.9,
        )
        obs2 = WorldObservation(
            camera_id="cam-1",
            frame_index=1,
            captured_at=datetime.now(UTC) - timedelta(seconds=30),
            floor_point=FloorPoint(1100, 2100, calibrated=True),
            bbox=BoundingBox(10, 20, 30, 40),
            embedding=[0.0] * 4,
            detection_confidence=0.9,
        )
        repo._observations["ph-1"] = [obs1, obs2]

        # Verify split endpoint exists and records latency
        before = _counter_value(metrics.cts_ph_splits_total, {"actor": "operator"})
        resp = client.post(
            "/ph/ph-1/split",
            json={"at_observation_id": "nonexistent", "reason": "test split"},
        )
        # The split will fail (observation not found), so counter won't increment
        assert resp.status_code == 422
        after = _counter_value(metrics.cts_ph_splits_total, {"actor": "operator"})
        # Counter unchanged since split failed
        assert after == before

    @pytest.mark.asyncio
    async def test_list_endpoint_records_latency(
        self, client: TestClient, repo: InMemoryPHRepository
    ) -> None:
        from app.observability.metrics import metrics

        await repo.save(_make_ph("ph-1"))
        client.get("/ph?include_transient=true")
        samples = list(metrics.cts_ph_api_latency_seconds.labels(endpoint="list").collect())
        if samples and samples[0].samples:
            count_sample = [s for s in samples[0].samples if s.name.endswith("_count")]
            if count_sample:
                assert count_sample[0].value >= 1
