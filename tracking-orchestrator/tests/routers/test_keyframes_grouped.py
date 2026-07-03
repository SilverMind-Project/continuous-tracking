"""M07: /internal/keyframes/grouped endpoint."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import BboxAnnotation, TaggedKeyframe
from app.routers.dashboard import get_keyframe_read_service, router
from app.services.keyframe_read_model import (
    KeyframeReadModelService,
    KeyframeReadRepositoryBundle,
)
from app.storage.base import (
    InMemoryBboxAnnotationRepository,
    InMemoryIdentityDecisionRepository,
    InMemoryKeyframeRepository,
)
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from app.storage.gallery import InMemoryGalleryRepository

_T0 = datetime(2026, 6, 19, 12, 0, 0, tzinfo=UTC)
_CAM = "camera-a"
_KEY = "frames/camera-a/0001-0.jpg"


def _client_with_data() -> TestClient:
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()

    async def _seed() -> None:
        for reason, ph in (("identity_changed", "ph-alpha"), ("periodic", "ph-beta")):
            kf_id = str(uuid.uuid4())
            await keyframe_repo.save_keyframe(
                TaggedKeyframe(
                    keyframe_id=kf_id,
                    ph_id=ph,
                    camera_id=_CAM,
                    minio_key=_KEY,
                    captured_at=_T0,
                    annotations={},
                    tag_reason=reason,  # type: ignore[arg-type]
                    expires_at=_T0 + timedelta(days=1),
                )
            )
            await bbox_repo.save_bbox_annotations(
                [
                    BboxAnnotation(
                        keyframe_id=kf_id,
                        ph_id="ph-alpha",
                        camera_id=_CAM,
                        x1=10,
                        y1=10,
                        x2=110,
                        y2=210,
                        detection_confidence=0.9,
                        frame_width=1920,
                        frame_height=1080,
                        identity_id="amma",
                    ),
                    BboxAnnotation(
                        keyframe_id=kf_id,
                        ph_id="ph-beta",
                        camera_id=_CAM,
                        x1=200,
                        y1=10,
                        x2=300,
                        y2=210,
                        detection_confidence=0.9,
                        frame_width=1920,
                        frame_height=1080,
                        identity_id="grandma",
                    ),
                ]
            )

    import asyncio

    asyncio.run(_seed())

    bundle = KeyframeReadRepositoryBundle(
        keyframe_repo=keyframe_repo,
        bbox_repo=bbox_repo,
        decision_repo=InMemoryIdentityDecisionRepository(),
        correction_repo=InMemoryIdentityCorrectionRepository(),
        gallery_repo=InMemoryGalleryRepository(),
    )
    svc = KeyframeReadModelService(bundle)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_keyframe_read_service] = lambda: svc
    return TestClient(app)


def test_grouped_endpoint_returns_one_card_with_all_bboxes() -> None:
    client = _client_with_data()
    resp = client.get("/internal/keyframes/grouped")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    frame = data["frames"][0]
    assert len(frame["triggers"]) == 2
    assert sorted(frame["trigger_reasons"]) == ["identity_changed", "periodic"]
    assert {b["effective_identity_id"] for b in frame["bboxes"]} == {"amma", "grandma"}


def test_grouped_endpoint_identity_filter() -> None:
    client = _client_with_data()
    resp = client.get("/internal/keyframes/grouped", params={"effective_identity_id": "amma"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    # Context preserved: grandma still present on the matched frame.
    assert {b["effective_identity_id"] for b in data["frames"][0]["bboxes"]} == {
        "amma",
        "grandma",
    }

    resp_none = client.get(
        "/internal/keyframes/grouped", params={"effective_identity_id": "nobody"}
    )
    assert resp_none.json()["total"] == 0


def test_grouped_endpoint_rejects_bad_timestamp() -> None:
    client = _client_with_data()
    resp = client.get("/internal/keyframes/grouped", params={"after": "not-a-date"})
    assert resp.status_code == 422
