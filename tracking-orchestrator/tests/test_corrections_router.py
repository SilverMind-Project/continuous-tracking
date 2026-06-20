"""Tests for the identity-corrections router (M06 service-backed)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import (
    BoundingBox,
    FloorPoint,
    PersonHypothesis,
    WorldObservation,
)
from app.routers.corrections import router as corrections_router
from app.routers.corrections import set_context
from app.services.identity_correction_service import (
    CorrectionConfig,
    IdentityCorrectionService,
)
from app.storage.base import InMemoryPHRepository
from app.storage.corrections import InMemoryIdentityCorrectionRepository

T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


class _FakePublisher:
    def __init__(self) -> None:
        self.published: list = []
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def publish(self, revision) -> str:
        self.published.append(revision)
        return f"msg-{len(self.published)}"


def _obs(obs_id: str, at: datetime) -> WorldObservation:
    return WorldObservation(
        camera_id="kitchen-1",
        frame_index=0,
        captured_at=at,
        floor_point=FloorPoint(1000, 2000, calibrated=True),
        bbox=BoundingBox(10, 20, 30, 40),
        embedding=[0.0] * 4,
        detection_confidence=0.9,
        observation_id=obs_id,
    )


@pytest.fixture
def client_and_publisher():
    ph_repo = InMemoryPHRepository()
    corr_repo = InMemoryIdentityCorrectionRepository()
    pub = _FakePublisher()
    obs = [_obs(f"obs-{i}", T0 + timedelta(seconds=i)) for i in range(4)]

    async def _seed() -> None:
        ph = PersonHypothesis(
            ph_id="gt-1",
            state_mean=(1.0, 1.0, 0.0, 0.0),
            state_cov=tuple([0.1] * 16),
            born_at=T0,
            last_seen_at=obs[-1].captured_at,
            last_seen_camera="kitchen-1",
            observation_count=0,
            current_identity_id="grandma",
            active_cameras=frozenset(["kitchen-1"]),
        )
        await ph_repo.save(ph)
        ph_repo._observations["gt-1"] = obs

    asyncio.run(_seed())

    service = IdentityCorrectionService(
        ph_repo=ph_repo,
        correction_repo=corr_repo,
        publisher=pub,
        config=CorrectionConfig(prior_window_s=30.0, discontinuity_gap_s=10.0),
    )
    set_context(service)

    app = FastAPI()
    app.include_router(corrections_router)
    return TestClient(app), pub, ph_repo, corr_repo


def test_propose_then_apply(client_and_publisher):
    client, pub, _ph_repo, _corr = client_and_publisher
    prop = client.post("/internal/corrections/propose", json={"ph_id": "gt-1"})
    assert prop.status_code == 200
    proposal = prop.json()
    assert proposal["ph_version"] == 0
    assert proposal["observation_ids"]

    resp = client.post(
        "/internal/corrections/apply",
        json={
            "ph_id": "gt-1",
            "actor": "caregiver@home",
            "reason_code": "wrong_person",
            "observation_start": proposal["start"]["captured_at"],
            "observation_end": proposal["end"]["captured_at"],
            "base_ph_version": proposal["ph_version"],
            "target_identity_id": "grandpa",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["previous_identity_id"] == "grandma"
    assert body["new_identity_id"] == "grandpa"
    assert body["revision_id"]
    assert len(pub.published) == 1
    assert pub.published[0].new_identity_id == "grandpa"


def test_apply_stale_version_returns_409(client_and_publisher):
    client, _pub, *_ = client_and_publisher
    resp = client.post(
        "/internal/corrections/apply",
        json={
            "ph_id": "gt-1",
            "actor": "caregiver@home",
            "reason_code": "wrong_person",
            "observation_start": T0.isoformat(),
            "observation_end": (T0 + timedelta(seconds=3)).isoformat(),
            "base_ph_version": 999,
            "target_identity_id": "grandpa",
        },
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "correction.stale_version"


def test_apply_empty_identity_returns_422(client_and_publisher):
    client, *_ = client_and_publisher
    resp = client.post(
        "/internal/corrections/apply",
        json={
            "ph_id": "gt-1",
            "actor": "caregiver@home",
            "reason_code": "wrong_person",
            "observation_start": T0.isoformat(),
            "observation_end": (T0 + timedelta(seconds=3)).isoformat(),
            "base_ph_version": 0,
            "target_identity_id": None,
            "set_unknown": False,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "correction.empty_identity"


def test_compensate_restores(client_and_publisher):
    client, _pub, _ph_repo, _corr = client_and_publisher
    prop = client.post("/internal/corrections/propose", json={"ph_id": "gt-1"}).json()
    applied = client.post(
        "/internal/corrections/apply",
        json={
            "ph_id": "gt-1",
            "actor": "caregiver@home",
            "reason_code": "wrong_person",
            "observation_start": prop["start"]["captured_at"],
            "observation_end": prop["end"]["captured_at"],
            "base_ph_version": prop["ph_version"],
            "target_identity_id": "grandpa",
        },
    ).json()
    comp = client.post(
        f"/internal/corrections/{applied['correction_id']}/compensate",
        json={"actor": "caregiver@home"},
    )
    assert comp.status_code == 200
    assert comp.json()["correction_id"] != applied["correction_id"]


def test_projection_ack_completes_job(client_and_publisher):
    client, _pub, *_ = client_and_publisher
    prop = client.post("/internal/corrections/propose", json={"ph_id": "gt-1"}).json()
    applied = client.post(
        "/internal/corrections/apply",
        json={
            "ph_id": "gt-1",
            "actor": "caregiver@home",
            "reason_code": "wrong_person",
            "observation_start": prop["start"]["captured_at"],
            "observation_end": prop["end"]["captured_at"],
            "base_ph_version": prop["ph_version"],
            "target_identity_id": "grandpa",
        },
    ).json()
    ack = client.post(
        "/internal/projection-acks",
        json={
            "revision_id": applied["revision_id"],
            "consumer": "cc",
            "schema_version": "1",
        },
    )
    assert ack.status_code == 200
    assert ack.json()["completed"] is True


def test_legacy_endpoint_marks_deprecated_and_applies(client_and_publisher):
    client, pub, _ph_repo, _corr = client_and_publisher
    resp = client.post(
        "/internal/corrections",
        json={
            "ph_id": "gt-1",
            "new_identity_id": "grandpa",
            "actor": "caregiver@home",
            "reason": "manual",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("Deprecation") == "true"
    assert resp.json()["new_identity_id"] == "grandpa"
    assert len(pub.published) == 1


def test_legacy_unknown_ph_returns_404(client_and_publisher):
    client, *_ = client_and_publisher
    resp = client.post(
        "/internal/corrections",
        json={"ph_id": "nope", "new_identity_id": "grandpa", "actor": "x"},
    )
    assert resp.status_code == 404
