"""Operator merge, split, and correction drop the face templates built before them."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.domain import BoundingBox, FloorPoint, PersonHypothesis, WorldObservation
from app.main import create_app
from app.routers.ph import set_face_template_reset, set_ph_repository
from app.storage.base import InMemoryPHRepository


class _RecordingReset:
    def __init__(self) -> None:
        self.reset: list[str] = []

    def reset_ph(self, ph_id: str) -> None:
        self.reset.append(ph_id)


@pytest.fixture
def repo() -> InMemoryPHRepository:
    return InMemoryPHRepository()


@pytest.fixture
def resets() -> Iterator[_RecordingReset]:
    recorder = _RecordingReset()
    set_face_template_reset(recorder)
    yield recorder
    set_face_template_reset(None)


@pytest.fixture
def client(repo: InMemoryPHRepository, resets: _RecordingReset) -> TestClient:
    set_ph_repository(repo)
    return TestClient(create_app())


def _ph(ph_id: str, camera: str = "cam-1") -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now - timedelta(minutes=30),
        last_seen_at=now,
        last_seen_camera=camera,
        observation_count=15,
        active_cameras=frozenset([camera]),
    )


def _observation(observation_id: str, frame_index: int) -> WorldObservation:
    return replace(
        WorldObservation(
            camera_id="cam-1",
            frame_index=frame_index,
            captured_at=datetime.now(UTC) - timedelta(seconds=frame_index),
            floor_point=FloorPoint(1000 * frame_index, 1000, calibrated=True),
            bbox=BoundingBox(10, 20, 30, 40),
            embedding=[0.0] * 4,
            detection_confidence=0.9,
        ),
        observation_id=observation_id,
    )


async def test_merge_resets_source_and_target(
    client: TestClient, repo: InMemoryPHRepository, resets: _RecordingReset
) -> None:
    await repo.save(_ph("ph-1", "cam-1"))
    await repo.save(_ph("ph-2", "cam-2"))
    resp = client.post(
        "/ph/merge", json={"source_ph_id": "ph-1", "target_ph_id": "ph-2", "reason": "same"}
    )
    assert resp.status_code == 200
    assert resets.reset == ["ph-1", "ph-2"]


async def test_batch_merge_resets_every_ph(
    client: TestClient, repo: InMemoryPHRepository, resets: _RecordingReset
) -> None:
    for ph_id, camera in (("ph-1", "cam-1"), ("ph-2", "cam-2"), ("ph-3", "cam-3")):
        await repo.save(_ph(ph_id, camera))
    resp = client.post(
        "/ph/batch_merge",
        json={"source_ph_ids": ["ph-1", "ph-2"], "target_ph_id": "ph-3", "reason": "same"},
    )
    assert resp.status_code == 200
    assert resets.reset == ["ph-1", "ph-2", "ph-3"]


async def test_split_resets_both_halves(
    client: TestClient, repo: InMemoryPHRepository, resets: _RecordingReset
) -> None:
    await repo.save(_ph("ph-1"))
    repo._observations["ph-1"] = [_observation(f"obs-{i}", i) for i in (1, 2, 3)]
    resp = client.post("/ph/ph-1/split", json={"at_observation_id": "obs-2", "reason": "two"})
    assert resp.status_code == 200
    assert resets.reset == ["ph-1", resp.json()["new_ph_id"]]


async def test_corrections_reset_the_corrected_phs(
    client: TestClient, repo: InMemoryPHRepository, resets: _RecordingReset
) -> None:
    await repo.save(_ph("ph-1"))
    await repo.save(_ph("ph-2"))
    single = client.post("/ph/ph-1/correct", json={"new_identity_id": "alice", "reason": "fix"})
    batch = client.post(
        "/ph/batch_correct",
        json={"corrections": [{"ph_id": "ph-2", "new_identity_id": "bob", "reason": "fix"}]},
    )
    assert (single.status_code, batch.status_code) == (200, 200)
    assert resets.reset == ["ph-1", "ph-2"]


async def test_a_rejected_merge_resets_nothing(client: TestClient, resets: _RecordingReset) -> None:
    resp = client.post(
        "/ph/merge", json={"source_ph_id": "ph-1", "target_ph_id": "ph-1", "reason": "self"}
    )
    assert resp.status_code == 422
    assert resets.reset == []
