"""Unit tests for GalleryRetentionService and expired lifecycle state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from app.domain import Identity, NewReviewCandidate
from app.observability import metrics as metrics_mod
from app.observability.metrics import build_metrics
from app.services.gallery_retention import GalleryRetentionConfig, GalleryRetentionService
from app.services.reid_review_service import ReIDReviewService
from app.storage.gallery import PENDING_AND_VERIFIED, VOTING_STATES, InMemoryGalleryRepository

_NOW = datetime(2026, 6, 30, 12, 0, 0, tzinfo=UTC)


class _FlakyCropDeleter:
    def __init__(self, *, fail_keys: set[str] | None = None) -> None:
        self.fail_keys: set[str] = set(fail_keys or ())
        self.deleted: list[str] = []

    async def delete_object(self, key: str) -> None:
        if key in self.fail_keys:
            raise OSError(f"transient storage error for {key}")
        self.deleted.append(key)


def _cand(
    candidate_id: str,
    *,
    seen_at: datetime,
    state: str = "pending_review",
) -> NewReviewCandidate:
    return NewReviewCandidate(
        candidate_id=candidate_id,
        identity_id="alice",
        embedding=[1.0] + [0.0] * 7,
        quality=0.9,
        orientation=0,
        camera_id="cam-1",
        capture_time=seen_at,
        ph_id="ph-1",
        observation_id=f"obs-{candidate_id}",
        origin_tracklet_id=f"obs-{candidate_id}",
        keyframe_id=None,
        crop_key=f"reid/{candidate_id}.jpg",
        source_frame_key=None,
        crop_hash="hash",
        frame_hash=None,
        dimensions=(64, 128),
        is_truncated=False,
        is_occluded=False,
        candidate_reason="face_derived",
        source_episode_id=f"ep-{candidate_id}",
        created_actor="pipeline",
        model_version="reid",
        preprocessing_version="v1",
        confidence=0.95,
        state=state,
    )


async def _seeded_repo() -> InMemoryGalleryRepository:
    repo = InMemoryGalleryRepository()
    await repo.upsert_identity(
        Identity(identity_id="alice", display_name="Alice", enrolled_at=_NOW)
    )
    # Auto-verified: 15d old (past 14d) vs 10d old (kept)
    await repo.create_review_candidate(
        _cand("auto-old", seen_at=_NOW - timedelta(days=15), state="auto_verified")
    )
    await repo.create_review_candidate(
        _cand("auto-fresh", seen_at=_NOW - timedelta(days=10), state="auto_verified")
    )
    # Pending: 31d old (past 30d) vs 20d old (kept)
    await repo.create_review_candidate(
        _cand("pend-old", seen_at=_NOW - timedelta(days=31), state="pending_review")
    )
    await repo.create_review_candidate(
        _cand("pend-fresh", seen_at=_NOW - timedelta(days=20), state="pending_review")
    )
    # Operator-verified: 91d old (past 90d) vs 60d old (kept)
    await repo.create_review_candidate(
        _cand("op-old", seen_at=_NOW - timedelta(days=91), state="pending_review")
    )
    await repo.apply_review_action("op-old", action="approve", actor="op", base_audit_version=1)
    await repo.create_review_candidate(
        _cand("op-fresh", seen_at=_NOW - timedelta(days=60), state="pending_review")
    )
    await repo.apply_review_action("op-fresh", action="approve", actor="op", base_audit_version=1)
    # Rejected tombstone: 120d old (must never be touched by retention)
    await repo.create_review_candidate(
        _cand("rej-old", seen_at=_NOW - timedelta(days=120), state="pending_review")
    )
    await repo.apply_review_action(
        "rej-old", action="reject", actor="op", base_audit_version=1, reason="blurry"
    )
    return repo


@pytest.fixture(autouse=True)
def _fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod, "metrics", build_metrics(CollectorRegistry()))


@pytest.mark.asyncio
async def test_shadow_mode_counts_without_mutating_or_deleting() -> None:
    repo = await _seeded_repo()
    deleter = _FlakyCropDeleter()
    svc = GalleryRetentionService(
        gallery_repo=repo,
        config=GalleryRetentionConfig(mode="shadow"),
        crop_storage=deleter,
        clock=lambda: _NOW,
    )

    res = await svc.run_once()
    assert res.mode == "shadow"
    assert res.by_state == {
        "auto_verified": 1,
        "pending_review": 1,
        "operator_verified": 1,
    }
    assert deleter.deleted == []
    counts = await repo.count_review_queue()
    assert counts["expired"] == 0
    assert counts["auto_verified"] == 2
    assert counts["pending_review"] == 2
    assert counts["operator_verified"] == 2
    assert counts["rejected"] == 1


@pytest.mark.asyncio
async def test_enabled_mode_expires_nulls_vectors_deletes_crops_and_keeps_audit() -> None:
    repo = await _seeded_repo()
    deleter = _FlakyCropDeleter()
    svc = GalleryRetentionService(
        gallery_repo=repo,
        config=GalleryRetentionConfig(mode="enabled"),
        crop_storage=deleter,
        clock=lambda: _NOW,
    )

    res = await svc.run_once()
    assert res.mode == "enabled"
    assert res.by_state == {
        "auto_verified": 1,
        "pending_review": 1,
        "operator_verified": 1,
    }
    assert set(deleter.deleted) == {
        "reid/auto-old.jpg",
        "reid/pend-old.jpg",
        "reid/op-old.jpg",
    }

    counts = await repo.count_review_queue()
    assert counts == {
        "auto_verified": 1,
        "pending_review": 1,
        "operator_verified": 1,
        "rejected": 1,
        "expired": 3,
    }

    # Expired rows never vote and never count against the orientation cap.
    hits = await repo.search_similar([1.0] + [0.0] * 7, limit=20, states=VOTING_STATES)
    assert {e.gallery_entry_id for e, _ in hits} == {"auto-fresh", "op-fresh"}
    assert await repo.count_gallery_entries("alice", 0, states=PENDING_AND_VERIFIED) == 3

    # Vector is nulled, audit history survives and records system:retention.
    entries = await repo.list_gallery_entries(identity_id="alice", active_only=False, states=None)
    by_id = {e.gallery_entry_id: e for e in entries}
    for expired_id in ("auto-old", "pend-old", "op-old"):
        assert by_id[expired_id].state == "expired"
        assert by_id[expired_id].embedding == []
        events = await repo.list_review_events(expired_id)
        assert events[-1].actor == "system:retention"
        assert events[-1].new_state == "expired"
        assert events[-1].reason == "retention_expired"
    # Prior operator approval event on op-old survived.
    op_events = await repo.list_review_events("op-old")
    assert any(ev.new_state == "operator_verified" and ev.actor == "op" for ev in op_events)

    # Rejected row was untouched.
    assert by_id["rej-old"].state == "rejected"

    # ReIDReviewService marks expired candidates as terminal.
    review_svc = ReIDReviewService(repo)
    expired_cand = await repo.get_review_candidate("op-old")
    assert expired_cand is not None
    elig = review_svc.eligibility(expired_cand)
    assert elig.eligible is False
    assert any("expired" in r for r in elig.reasons)


@pytest.mark.asyncio
async def test_crop_delete_failure_is_counted_and_retried_next_tick() -> None:
    repo = await _seeded_repo()
    deleter = _FlakyCropDeleter(fail_keys={"reid/auto-old.jpg"})
    svc = GalleryRetentionService(
        gallery_repo=repo,
        config=GalleryRetentionConfig(mode="enabled"),
        crop_storage=deleter,
        clock=lambda: _NOW,
    )

    first = await svc.run_once()
    assert first.crop_delete_failures == 1
    assert "reid/auto-old.jpg" in svc.pending_crop_deletes
    assert metrics_mod.metrics.cts_gallery_retention_crop_delete_failures_total._value.get() == 1.0

    # Storage recovers on next tick: pending crop key is retried and cleared.
    deleter.fail_keys.clear()
    second = await svc.run_once()
    assert second.crops_deleted == 1
    assert second.crop_delete_failures == 0
    assert svc.pending_crop_deletes == frozenset()
    assert "reid/auto-old.jpg" in deleter.deleted
