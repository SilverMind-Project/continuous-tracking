"""Tests for TrackletUnmergeService."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import GlobalTrack
from app.storage.base import InMemoryDoNotFuseRepository, InMemoryGlobalTrackRepository
from app.tracking.tracklet_unmerge_service import TrackletUnmergeService


@pytest.fixture()
def gt_repo() -> InMemoryGlobalTrackRepository:
    repo = InMemoryGlobalTrackRepository()
    yield repo
    repo._tracks.clear()
    repo._by_tracklet.clear()


@pytest.fixture()
def dnf_repo() -> InMemoryDoNotFuseRepository:
    return InMemoryDoNotFuseRepository()


@pytest.fixture()
def svc(
    gt_repo: InMemoryGlobalTrackRepository, dnf_repo: InMemoryDoNotFuseRepository
) -> TrackletUnmergeService:
    return TrackletUnmergeService(global_track_repo=gt_repo, dnf_repo=dnf_repo)


@pytest.mark.asyncio
async def test_unmerge_creates_new_global_track(
    gt_repo: InMemoryGlobalTrackRepository,
    dnf_repo: InMemoryDoNotFuseRepository,
    svc: TrackletUnmergeService,
) -> None:
    # Pre-populate: one global track with a single tracklet.
    now = datetime.now(UTC)
    gt = GlobalTrack(
        global_track_id="original-gt",
        camera_ids=["cam_a"],
        tracklet_ids=["tr1"],
        started_at=now,
        last_seen_at=now,
    )
    await gt_repo.save(gt)

    result = await svc.unmerge(tracklet_id="tr1", requested_by="test")

    assert result.original_global_track_id == "original-gt"
    assert result.new_global_track_id != "original-gt"

    # New GT should contain the tracklet.
    new_gt = await gt_repo.get(result.new_global_track_id)
    assert new_gt is not None
    assert "tr1" in new_gt.tracklet_ids

    # Original GT should no longer have the tracklet.
    orig_gt = await gt_repo.get("original-gt")
    assert orig_gt is not None
    assert "tr1" not in orig_gt.tracklet_ids


@pytest.mark.asyncio
async def test_unmerge_adds_do_not_fuse_hint(
    gt_repo: InMemoryGlobalTrackRepository,
    dnf_repo: InMemoryDoNotFuseRepository,
    svc: TrackletUnmergeService,
) -> None:
    now = datetime.now(UTC)
    gt = GlobalTrack(
        global_track_id="original-gt",
        camera_ids=["cam_a"],
        tracklet_ids=["tr1"],
        started_at=now,
        last_seen_at=now,
    )
    await gt_repo.save(gt)

    await svc.unmerge(tracklet_id="tr1", requested_by="test")

    assert await dnf_repo.is_blocked("tr1", "original-gt") is True


@pytest.mark.asyncio
async def test_unmerge_raises_for_unknown_tracklet(
    svc: TrackletUnmergeService,
) -> None:
    with pytest.raises(ValueError, match="not associated with any global track"):
        await svc.unmerge(tracklet_id="nonexistent", requested_by="test")


@pytest.mark.asyncio
async def test_unmerge_closes_original_if_last_tracklet(
    gt_repo: InMemoryGlobalTrackRepository,
    svc: TrackletUnmergeService,
) -> None:
    now = datetime.now(UTC)
    gt = GlobalTrack(
        global_track_id="original-gt",
        camera_ids=["cam_a"],
        tracklet_ids=["tr1"],
        started_at=now,
        last_seen_at=now,
    )
    await gt_repo.save(gt)

    result = await svc.unmerge(tracklet_id="tr1", requested_by="test")

    orig_gt = await gt_repo.get(result.original_global_track_id)
    assert orig_gt is not None
    assert orig_gt.state == "closed"
    assert len(orig_gt.tracklet_ids) == 0
