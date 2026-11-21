"""The identity readiness route reports policies, gallery counts, and camera calibration only."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.state import CalibrationState
from app.domain import GalleryEmbedding, GallerySeedSource, Identity
from app.routers.readiness import router
from app.services.camera_room_map import CameraRoomBinding, CameraRoomMap
from app.services.identity_readiness import IdentityPolicySnapshot, IdentityReadinessService
from app.storage.gallery import InMemoryGalleryRepository

T0 = datetime(2026, 11, 14, 9, 0, 0, tzinfo=UTC)
_IDENTITY = "resident-one"
_MATRIX = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@dataclass
class _Groups:
    count: int | None

    def overlap_group_count(self) -> int | None:
        return self.count


def _policy(face_evidence_policy: str = "shadow", templates: str = "off") -> IdentityPolicySnapshot:
    return IdentityPolicySnapshot(
        face_evidence_policy=face_evidence_policy,
        face_templates_mode=templates,
        face_quality_mode="shadow",
        cc_assertion_mode="shadow",
        expected_arcface_model_version=None,
        expected_quality_version="heuristic-v1",
    )


async def _cameras(*camera_ids: str) -> CameraRoomMap:
    room_map = CameraRoomMap()
    await room_map.set_all(
        [
            CameraRoomBinding(camera_id=cid, room_id="1", room_name="Room", bound_at=T0)
            for cid in camera_ids
        ]
    )
    return room_map


async def _gallery() -> InMemoryGalleryRepository:
    repo = InMemoryGalleryRepository()
    await repo.upsert_identity(
        Identity(identity_id=_IDENTITY, display_name="Someone", enrolled_at=T0, is_active=True)
    )
    for index, (state, source) in enumerate(
        (
            ("pending_review", GallerySeedSource.FACE),
            ("pending_review", GallerySeedSource.WEARABLE_ROOM_ATTESTED),
            ("auto_verified", GallerySeedSource.WEARABLE_ATTESTED),
        )
    ):
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"entry-{index}",
                identity_id=_IDENTITY,
                embedding=[1.0, 0.0],
                seen_at=T0,
                quality=0.9,
                origin_tracklet_id=f"tracklet-{index}",
                state=state,
                seed_source=source,
            )
        )
    return repo


def _client(service: IdentityReadinessService | None) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.state.identity_readiness_service = service
    yield TestClient(app)


@pytest.fixture
async def calibration() -> CalibrationState:
    state = CalibrationState()
    await state.set_homography("kitchen", _MATRIX)
    # A homography for a camera CC no longer lists as enabled is not reported.
    await state.set_homography("retired", _MATRIX)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["legacy", "shadow", "calibrated"])
async def test_reports_policies_counts_and_camera_calibration(
    calibration: CalibrationState, policy: str
) -> None:
    service = IdentityReadinessService(
        policy=_policy(face_evidence_policy=policy, templates="shadow"),
        calibration_state=calibration,
        camera_room_map=await _cameras("kitchen", "hall"),
        overlap_groups=_Groups(count=2),
        gallery_repo=await _gallery(),
    )
    client = next(_client(service))

    response = client.get("/internal/identity/readiness")

    assert response.status_code == 200
    body = response.json()
    assert body["face_evidence_policy"] == policy
    assert body["face_templates_mode"] == "shadow"
    assert body["expected_arcface_model_version"] is None
    assert body["reid_gallery_counts"] == {
        "pending_review": 2,
        "auto_verified": 1,
        "operator_verified": 0,
        "rejected": 0,
    }
    # Rows a wearable taught, by state, for the readiness page's body-sample step.
    assert body["wearable_gallery_counts"] == {
        "pending_review": 1,
        "auto_verified": 1,
        "operator_verified": 0,
        "rejected": 0,
    }
    assert body["cameras"] == [
        {"camera_id": "hall", "calibrated": False},
        {"camera_id": "kitchen", "calibrated": True},
    ]
    assert body["overlap_group_count"] == 2
    assert body["identities_known"] == 1
    # Counts and settings only: no identity id ever leaves this route.
    assert _IDENTITY not in response.text


@pytest.mark.asyncio
async def test_without_gallery_or_tracker_reports_null_not_zero(
    calibration: CalibrationState,
) -> None:
    service = IdentityReadinessService(
        policy=_policy(),
        calibration_state=calibration,
        camera_room_map=await _cameras(),
        overlap_groups=_Groups(count=None),
    )
    body = next(_client(service)).get("/internal/identity/readiness").json()

    assert body["reid_gallery_counts"] is None
    assert body["wearable_gallery_counts"] is None
    assert body["identities_known"] is None
    assert body["overlap_group_count"] is None
    assert body["cameras"] == []


def test_unwired_service_is_503() -> None:
    response = next(_client(None)).get("/internal/identity/readiness")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "identity_readiness.unavailable"
