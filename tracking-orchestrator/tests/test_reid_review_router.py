"""Tests for the M09 ReID review-queue router (service-backed, InMemory repo)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import ReviewCandidate
from app.routers.reid_review import router as reid_review_router
from app.routers.reid_review import set_context
from app.services.reid_review_service import ReIDReviewService
from app.storage.gallery import InMemoryGalleryRepository

T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


def _candidate(
    candidate_id: str,
    *,
    state: str = "pending_review",
    identity_id: str = "resident-1",
    camera_id: str = "kitchen-1",
    model_version: str = "v1",
    reason: str = "multiview",
    is_truncated: bool = False,
    is_occluded: bool = False,
    created_offset_s: int = 0,
    audit_version: int = 1,
) -> ReviewCandidate:
    return ReviewCandidate(
        candidate_id=candidate_id,
        identity_id=identity_id,
        proposed_identity_id=identity_id,
        effective_identity_id=identity_id,
        state=state,
        label_source="reid",
        candidate_reason=reason,
        model_version=model_version,
        preprocessing_version="v1",
        dimension=768,
        crop_key=f"reid-candidates/{model_version}/{candidate_id}.jpg",
        source_frame_key=f"reid-candidates-frames/{model_version}/{candidate_id}.jpg",
        crop_hash="abc",
        frame_hash="def",
        bbox={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        crop_width=128,
        crop_height=256,
        ph_id="11111111-1111-1111-1111-111111111111",
        observation_id="22222222-2222-2222-2222-222222222222",
        keyframe_id="33333333-3333-3333-3333-333333333333",
        camera_id=camera_id,
        capture_time=T0 + timedelta(seconds=created_offset_s),
        confidence=0.9,
        orientation=4,
        quality=0.8,
        is_truncated=is_truncated,
        is_occluded=is_occluded,
        source_episode_id="44444444-4444-4444-4444-444444444444",
        created_actor="system",
        created_at=T0 + timedelta(seconds=created_offset_s),
        seen_at=T0 + timedelta(seconds=created_offset_s),
        reviewed_actor=None,
        reviewed_time=None,
        review_reason=None,
        review_note=None,
        audit_version=audit_version,
    )


@pytest.fixture
def client_repo():
    repo = InMemoryGalleryRepository()
    service = ReIDReviewService(repo, active_model_versions={"v1"})
    set_context(service)
    app = FastAPI()
    app.include_router(reid_review_router)
    return TestClient(app), repo


def test_list_filters_and_pagination(client_repo):
    client, repo = client_repo
    for i in range(5):
        repo.seed_review_candidate(_candidate(f"c{i}", created_offset_s=i))
    repo.seed_review_candidate(_candidate("other", camera_id="bath-1", created_offset_s=9))

    r = client.get("/internal/reid-review/candidates", params={"limit": 2, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 6
    assert len(body["candidates"]) == 2
    # Oldest-first ordering: c0 then c1.
    assert [c["candidate_id"] for c in body["candidates"]] == ["c0", "c1"]

    r2 = client.get("/internal/reid-review/candidates", params={"camera_id": "bath-1"})
    assert r2.json()["total"] == 1
    assert r2.json()["candidates"][0]["candidate_id"] == "other"


def test_detail_includes_eligibility_and_events(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("c1"))
    r = client.get("/internal/reid-review/candidates/c1")
    assert r.status_code == 200
    body = r.json()
    assert body["candidate"]["candidate_id"] == "c1"
    assert body["eligibility"]["eligible"] is True
    assert body["eligibility"]["model_compatible"] is True
    assert body["events"] == []


def test_detail_404(client_repo):
    client, _ = client_repo
    assert client.get("/internal/reid-review/candidates/missing").status_code == 404


def test_approve_individual_then_history(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("c1"))
    r = client.post(
        "/internal/reid-review/candidates/c1/approve",
        json={"actor": "alice", "base_audit_version": 1},
    )
    assert r.status_code == 200
    assert r.json()["state"] == "operator_verified"
    events = client.get("/internal/reid-review/candidates/c1/events").json()["events"]
    assert len(events) == 1
    assert events[0]["new_state"] == "operator_verified"
    assert events[0]["actor"] == "alice"


def test_stale_audit_version_blocks_approval(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("c1", audit_version=3))
    r = client.post(
        "/internal/reid-review/candidates/c1/approve",
        json={"actor": "alice", "base_audit_version": 1},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "reid_review.stale"


def test_ineligible_candidate_cannot_be_approved(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("trunc", is_truncated=True))
    r = client.post(
        "/internal/reid-review/candidates/trunc/approve",
        json={"actor": "alice", "base_audit_version": 1},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "reid_review.ineligible"


def test_incompatible_model_surfaced_and_blocks(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("old", model_version="v0"))
    detail = client.get("/internal/reid-review/candidates/old").json()
    assert detail["eligibility"]["model_compatible"] is False
    assert detail["eligibility"]["eligible"] is False
    r = client.post(
        "/internal/reid-review/candidates/old/approve",
        json={"actor": "alice", "base_audit_version": 1},
    )
    assert r.status_code == 409


def test_relabel_requires_target(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("c1"))
    r = client.post(
        "/internal/reid-review/candidates/c1/relabel",
        json={"actor": "alice", "base_audit_version": 1, "target_identity_id": "resident-2"},
    )
    assert r.status_code == 200
    assert r.json()["identity_id"] == "resident-2"
    assert r.json()["state"] == "operator_verified"


def test_batch_reject_reports_per_item(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("ok1"))
    repo.seed_review_candidate(_candidate("ok2"))
    repo.seed_review_candidate(_candidate("stale", audit_version=5))
    r = client.post(
        "/internal/reid-review/reject-batch",
        json={
            "actor": "alice",
            "items": [
                {"candidate_id": "ok1", "base_audit_version": 1, "reason": "wrong_person"},
                {"candidate_id": "ok2", "base_audit_version": 1, "reason": "wrong_person"},
                {"candidate_id": "stale", "base_audit_version": 1, "reason": "wrong_person"},
                {"candidate_id": "ghost", "base_audit_version": 1, "reason": "wrong_person"},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["rejected"] == 2
    assert body["failed"] == 2
    by_id = {item["candidate_id"]: item for item in body["results"]}
    assert by_id["ok1"]["ok"] is True
    assert by_id["stale"]["error_code"] == "conflict"
    assert by_id["ghost"]["error_code"] == "not_found"


def test_reject_then_rejected_audit_remains(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("c1"))
    r = client.post(
        "/internal/reid-review/candidates/c1/reject",
        json={"actor": "alice", "base_audit_version": 1, "reason": "wrong_person"},
    )
    assert r.status_code == 200
    assert r.json()["state"] == "rejected"
    # Rejected candidates are still visible by explicit state filter with audit intact.
    rejected = client.get("/internal/reid-review/candidates", params={"state": "rejected"}).json()
    assert rejected["total"] == 1
    events = client.get("/internal/reid-review/candidates/c1/events").json()["events"]
    assert events[0]["new_state"] == "rejected"


def test_no_bulk_approve_endpoint(client_repo):
    client, _ = client_repo
    # Batch endpoint only rejects; there is no batch approve route.
    assert client.post("/internal/reid-review/approve-batch", json={}).status_code == 404


def test_counts(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("c1"))
    repo.seed_review_candidate(_candidate("c2", state="operator_verified"))
    repo.seed_review_candidate(_candidate("c3", state="rejected"))
    counts = client.get("/internal/reid-review/counts").json()
    assert counts == {"pending_review": 1, "operator_verified": 1, "rejected": 1}


def test_compensate_unverifies(client_repo):
    client, repo = client_repo
    repo.seed_review_candidate(_candidate("c1"))
    client.post(
        "/internal/reid-review/candidates/c1/approve",
        json={"actor": "alice", "base_audit_version": 1},
    )
    r = client.post(
        "/internal/reid-review/candidates/c1/compensate",
        json={"actor": "bob"},
    )
    assert r.status_code == 200
    assert r.json()["state"] == "pending_review"
    # Original approval event is retained, plus the compensating event.
    events = client.get("/internal/reid-review/candidates/c1/events").json()["events"]
    assert [e["new_state"] for e in events] == ["operator_verified", "pending_review"]
