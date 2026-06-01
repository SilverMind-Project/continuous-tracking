"""Tests for internal dashboard bbox endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import BboxAnnotation
from app.routers.dashboard import get_bbox_repo, router
from app.storage.base import InMemoryBboxAnnotationRepository


def _client(*, bbox_repo: InMemoryBboxAnnotationRepository) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_bbox_repo] = lambda: bbox_repo
    return TestClient(app)


def test_keyframe_bboxes_returns_persisted_annotations() -> None:
    bbox_repo = InMemoryBboxAnnotationRepository()
    now = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
    annotation = BboxAnnotation(
        keyframe_id="kf-001",
        ph_id="ph-001",
        camera_id="cam-a",
        x1=10,
        y1=20,
        x2=110,
        y2=220,
        detection_confidence=0.92,
        frame_width=1920,
        frame_height=1080,
        identity_id="alice",
        created_at=now,
    )

    import asyncio

    asyncio.run(bbox_repo.save_bbox_annotations([annotation]))
    client = _client(bbox_repo=bbox_repo)

    resp = client.get("/internal/keyframes/kf-001/bboxes")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    bbox = data["bboxes"][0]
    assert bbox["id"]
    assert bbox["keyframe_id"] == "kf-001"
    assert bbox["ph_id"] == "ph-001"
    assert bbox["camera_id"] == "cam-a"
    assert bbox["identity_id"] == "alice"
    assert bbox["x1"] == 10
    assert bbox["y1"] == 20
    assert bbox["x2"] == 110
    assert bbox["y2"] == 220
    assert bbox["detection_confidence"] == 0.92
