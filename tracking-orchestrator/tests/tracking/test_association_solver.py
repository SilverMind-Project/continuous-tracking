"""Tests for AssociationSolver, GlobalTrackService, and hard rejection rules."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import GalleryEmbedding, GlobalTrack, Tracklet
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
)
from app.tracking.association_evidence import (
    AssociationCandidate,
)
from app.tracking.association_solver import (
    HARD_REJECT_CHECKS,
    AssociationConfig,
    AssociationSolver,
    _check_do_not_fuse,
    _check_identity_conflict,
    _check_temporal_infeasible,
)
from app.tracking.camera_adjacency import CameraAdjacency
from app.tracking.global_track_service import GlobalTrackService

_NOW = datetime.now(UTC)


def _tracklet(tid: str, cam: str, floor_point=None, last_bbox=None) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        camera_id=cam,
        detection_ids=[f"det-{tid}"],
        started_at=_NOW,
        state="active",
        last_floor_point=floor_point,
        last_bbox=last_bbox,
    )


def _gt(
    gid: str, cameras: list[str], tracklets: list[str], identity_id: str | None = None
) -> GlobalTrack:
    return GlobalTrack(
        global_track_id=gid,
        camera_ids=cameras,
        tracklet_ids=tracklets,
        started_at=_NOW,
        last_seen_at=_NOW,
        current_identity_id=identity_id,
        state="active",
    )


# ---------------------------------------------------------------------------
# Hard rejection tests
# ---------------------------------------------------------------------------


class TestHardRejection:
    def test_do_not_fuse_rejects(self) -> None:
        c = AssociationCandidate(
            source_tracklet_id="tl-1",
            target_global_track_id="gt-1",
            appearance_sim=0.9,
            floor_distance_m=1.0,
            temporal_feasible=True,
            overlap_group_id=None,
            do_not_fuse=True,
        )
        assert _check_do_not_fuse(c) == "do_not_fuse"

    def test_do_not_fuse_passes_when_false(self) -> None:
        c = AssociationCandidate(
            source_tracklet_id="tl-1",
            target_global_track_id="gt-1",
            appearance_sim=0.9,
            floor_distance_m=1.0,
            temporal_feasible=True,
            overlap_group_id=None,
            do_not_fuse=False,
        )
        assert _check_do_not_fuse(c) is None

    def test_identity_conflict_rejects(self) -> None:
        c = AssociationCandidate(
            source_tracklet_id="tl-1",
            target_global_track_id="gt-1",
            appearance_sim=0.9,
            floor_distance_m=1.0,
            temporal_feasible=True,
            overlap_group_id=None,
            identity_conflict=True,
        )
        assert _check_identity_conflict(c) == "identity_conflict"

    def test_temporal_infeasible_rejects(self) -> None:
        c = AssociationCandidate(
            source_tracklet_id="tl-1",
            target_global_track_id="gt-1",
            appearance_sim=0.9,
            floor_distance_m=1.0,
            temporal_feasible=False,
            overlap_group_id=None,
        )
        assert _check_temporal_infeasible(c) == "temporal_infeasible"

    def test_all_hard_checks_registered(self) -> None:
        """All three hard reject checks must be in the HARD_REJECT_CHECKS list."""
        assert len(HARD_REJECT_CHECKS) == 3


# ---------------------------------------------------------------------------
# AssociationSolver tests
# ---------------------------------------------------------------------------


class TestAssociationSolver:
    def test_one_to_one_assignment_chooses_highest_legal_gt(self) -> None:
        """A single tracklet should be assigned to the highest-scoring legal GT."""
        adjacency = CameraAdjacency()
        solver = AssociationSolver(adjacency=adjacency)

        tl = _tracklet("tl-1", "cam-a")
        gt_a = _gt("gt-a", ["cam-b"], ["tl-old-a"])
        gt_b = _gt("gt-b", ["cam-c"], ["tl-old-b"])

        candidates = [
            AssociationCandidate(
                source_tracklet_id="tl-1",
                target_global_track_id="gt-a",
                appearance_sim=0.85,
                floor_distance_m=2.0,
                temporal_feasible=True,
                overlap_group_id=None,
                score=0.8,
            ),
            AssociationCandidate(
                source_tracklet_id="tl-1",
                target_global_track_id="gt-b",
                appearance_sim=0.60,
                floor_distance_m=5.0,
                temporal_feasible=True,
                overlap_group_id=None,
                score=0.5,
            ),
        ]

        results = solver._hungarian_assign(candidates, [tl], [gt_a, gt_b], None)
        assert len(results) == 1
        assert results[0][0].tracklet_id == "tl-1"
        assert results[0][1].global_track_id == "gt-a"

    def test_two_tracklets_cannot_attach_to_same_gt(self) -> None:
        """Two tracklets from different cameras cannot attach to the same GT
        in the same frame unless both are in overlap group duplicate views."""
        adjacency = CameraAdjacency()
        solver = AssociationSolver(adjacency=adjacency)

        tl_a = _tracklet("tl-1", "cam-a")
        tl_b = _tracklet("tl-2", "cam-b")
        gt = _gt("gt-1", ["cam-c"], ["tl-old"])

        candidates = [
            AssociationCandidate(
                source_tracklet_id="tl-1",
                target_global_track_id="gt-1",
                appearance_sim=0.85,
                floor_distance_m=2.0,
                temporal_feasible=True,
                overlap_group_id=None,
                score=0.8,
            ),
            AssociationCandidate(
                source_tracklet_id="tl-2",
                target_global_track_id="gt-1",
                appearance_sim=0.82,
                floor_distance_m=2.5,
                temporal_feasible=True,
                overlap_group_id=None,
                score=0.75,
            ),
        ]

        results = solver._hungarian_assign(candidates, [tl_a, tl_b], [gt], None)
        # Only one tracklet can be assigned to the single GT.
        assert len(results) <= 1

    def test_floor_plan_mismatch_rejected_by_candidate_scoring(self) -> None:
        """Candidates with unknown geometry should require strong appearance."""
        adjacency = CameraAdjacency()
        config = AssociationConfig(uncalibrated_appearance_threshold=0.65)
        solver = AssociationSolver(adjacency=adjacency, config=config)

        tl = _tracklet("tl-1", "cam-a")
        gt = _gt("gt-1", ["cam-b"], ["tl-old"])

        # Low appearance, unknown geometry → should be skipped.
        candidates = [
            AssociationCandidate(
                source_tracklet_id="tl-1",
                target_global_track_id="gt-1",
                appearance_sim=0.5,
                floor_distance_m=None,
                temporal_feasible=True,
                overlap_group_id=None,
                score=0.4,
            ),
        ]

        results = solver._hungarian_assign(candidates, [tl], [gt], None)
        assert len(results) == 0

    def test_uncalibrated_strong_appearance_passes(self) -> None:
        """Unknown geometry with strong appearance should still be assigned."""
        adjacency = CameraAdjacency()
        config = AssociationConfig(uncalibrated_appearance_threshold=0.65)
        solver = AssociationSolver(adjacency=adjacency, config=config)

        tl = _tracklet("tl-1", "cam-a")
        gt = _gt("gt-1", ["cam-b"], ["tl-old"])

        candidates = [
            AssociationCandidate(
                source_tracklet_id="tl-1",
                target_global_track_id="gt-1",
                appearance_sim=0.85,
                floor_distance_m=None,
                temporal_feasible=True,
                overlap_group_id=None,
                score=0.75,
            ),
        ]

        results = solver._hungarian_assign(candidates, [tl], [gt], None)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# GlobalTrackService integration tests
# ---------------------------------------------------------------------------


class TestGlobalTrackService:
    @pytest.mark.asyncio
    async def test_creates_new_gt_for_unmatched_tracklet(self) -> None:
        """A tracklet with no matching existing GT should create a new one."""
        adjacency = CameraAdjacency()
        gallery = InMemoryGalleryRepository()
        gt_repo = InMemoryGlobalTrackRepository()
        service = GlobalTrackService(
            gallery=gallery,
            adjacency=adjacency,
            global_track_repo=gt_repo,
        )

        tl = _tracklet("tl-new", "cam-a")
        result = await service.associate([tl], _NOW)

        assert len(result) == 1
        assert result[0].global_track_id != ""
        assert "tl-new" in result[0].tracklet_ids

    @pytest.mark.asyncio
    async def test_same_camera_reentry_extends_existing_gt(self) -> None:
        """A tracklet on the same camera as an existing GT should re-extend it."""
        adjacency = CameraAdjacency()
        gallery = InMemoryGalleryRepository()
        gt_repo = InMemoryGlobalTrackRepository()

        # Pre-create an active GT on cam-a.
        await gt_repo.merge_tracklets(tracklet_ids=["tl-old"], camera_ids=["cam-a"])
        # Seed a gallery entry so similarity works.
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-1",
                identity_id="",
                embedding=[0.0] * 768,
                seen_at=_NOW,
                quality=1.0,
                origin_tracklet_id="tl-old",
            )
        )
        # Seed a gallery entry for the new tracklet.
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="ge-2",
                identity_id="",
                embedding=[0.0] * 768,  # identical embedding → max similarity
                seen_at=_NOW,
                quality=1.0,
                origin_tracklet_id="tl-new",
            )
        )

        service = GlobalTrackService(
            gallery=gallery,
            adjacency=adjacency,
            global_track_repo=gt_repo,
        )

        tl = _tracklet("tl-new", "cam-a")
        result = await service.associate([tl], _NOW)

        assert len(result) == 1
        assert "tl-new" in result[0].tracklet_ids

    @pytest.mark.asyncio
    async def test_identity_conflict_prevents_auto_merge(self) -> None:
        """Two GTs with different committed identities must not auto-merge."""
        adjacency = CameraAdjacency()
        gallery = InMemoryGalleryRepository()
        gt_repo = InMemoryGlobalTrackRepository()

        # Create two GTs with different identities but high similarity.
        gt_a = await gt_repo.merge_tracklets(tracklet_ids=["tl-a"], camera_ids=["cam-x"])
        gt_b = await gt_repo.merge_tracklets(tracklet_ids=["tl-b"], camera_ids=["cam-x"])
        await gt_repo.assign_identity(gt_a.global_track_id, "alice")
        await gt_repo.assign_identity(gt_b.global_track_id, "bob")

        # Both are in the same overlap group.
        from app.tracking.camera_adjacency import AdjacencyEdge as GraphAdjacencyEdge

        adjacency.add_edge(GraphAdjacencyEdge(from_camera="cam-x", to_camera="cam-x", overlap=True))

        service = GlobalTrackService(
            gallery=gallery,
            adjacency=adjacency,
            global_track_repo=gt_repo,
        )

        # Run consolidation: alice and bob should NOT merge.
        await service._consolidate_overlap_group_gts([gt_a, gt_b])

        # Both should still exist as separate GTs.
        fresh_a = await gt_repo.get(gt_a.global_track_id)
        fresh_b = await gt_repo.get(gt_b.global_track_id)
        assert fresh_a is not None
        assert fresh_b is not None
        assert fresh_a.global_track_id != fresh_b.global_track_id
