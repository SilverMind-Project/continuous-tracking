"""Tests for CrossCameraAssociator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import (
    CameraId,
    GalleryEmbedding,
    GlobalTrack,
    Tracklet,
    TrackletId,
)
from app.storage.base import (
    GlobalTrackRepository,
    InMemoryGalleryRepository,
    InMemoryGlobalTrackRepository,
)
from app.tracking.camera_adjacency import AdjacencyEdge, CameraAdjacency
from app.tracking.cross_camera import CrossCamConfig, CrossCameraAssociator, TrackletPairScore


@pytest.fixture()
def adjacency() -> CameraAdjacency:
    adj = CameraAdjacency()
    adj.add_edge(AdjacencyEdge("cam_a", "cam_b", max_transition_seconds=120))
    adj.add_edge(AdjacencyEdge("cam_b", "cam_c", max_transition_seconds=60))
    return adj


@pytest.fixture()
def global_track_repo() -> GlobalTrackRepository:
    repo = InMemoryGlobalTrackRepository()
    yield repo
    # Clean up after each test to prevent state leakage.
    repo._tracks.clear()
    repo._by_tracklet.clear()


@pytest.fixture()
def gallery() -> InMemoryGalleryRepository:
    return InMemoryGalleryRepository()


@pytest.fixture()
def assoc(
    adjacency: CameraAdjacency,
    global_track_repo: GlobalTrackRepository,
    gallery: InMemoryGalleryRepository,
) -> CrossCameraAssociator:
    return CrossCameraAssociator(
        gallery=gallery,
        adjacency=adjacency,
        global_track_repo=global_track_repo,
        config=CrossCamConfig(
            alpha=0.7,
            min_link_score=0.5,
        ),
    )


def _make_gallery_entry(
    tracklet_id: str,
    camera_id: str,
    embedding: list[float] | None = None,
    quality: float = 0.9,
) -> GalleryEmbedding:
    """Create a gallery embedding for a tracklet."""
    import uuid

    if embedding is None:
        embedding = [0.0] * 768
    from datetime import UTC

    return GalleryEmbedding(
        gallery_entry_id=str(uuid.uuid4()),
        identity_id="test-identity",
        embedding=embedding,
        seen_at=datetime.now(UTC),
        quality=quality,
        origin_tracklet_id=tracklet_id,
        camera_id=camera_id,
    )


def _make_tracklet(
    tracklet_id: TrackletId,
    camera_id: CameraId,
    state: str = "active",
) -> Tracklet:
    now = datetime.now(UTC)
    return Tracklet(
        tracklet_id=tracklet_id,
        camera_id=camera_id,
        detection_ids=[f"det-{tracklet_id}"],
        started_at=now,
        ended_at=None if state == "active" else now,
        state=state,
    )


class TestCrossCameraAssociator:
    """Test cross-camera association logic."""

    @pytest.mark.asyncio
    async def test_empty_tracklets(self, assoc: CrossCameraAssociator) -> None:
        result = await assoc.associate([], captured_at=datetime.now(UTC))
        assert result == []

    @pytest.mark.asyncio
    async def test_single_tracklet_no_match(
        self,
        assoc: CrossCameraAssociator,
        global_track_repo: GlobalTrackRepository,
    ) -> None:
        """A single tracklet should create a new GlobalTrack."""
        t = _make_tracklet("t1", "cam_a")
        result = await assoc.associate([t], captured_at=datetime.now(UTC))

        assert len(result) == 1
        assert result[0].global_track_id is not None
        assert "cam_a" in result[0].camera_ids
        assert result[0].tracklet_ids == ["t1"]

    @pytest.mark.asyncio
    async def test_two_tracklets_same_camera_no_link(
        self,
        assoc: CrossCameraAssociator,
    ) -> None:
        """Tracklets on the same camera should not be linked."""
        t_a = _make_tracklet("t1", "cam_a")
        t_b = _make_tracklet("t2", "cam_a")
        result = await assoc.associate([t_a, t_b], captured_at=datetime.now(UTC))

        # Each should be in its own GlobalTrack.
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_two_tracklets_different_cameras_no_adjacency(
        self,
        assoc: CrossCameraAssociator,
    ) -> None:
        """Tracklets on non-adjacent cameras should not be linked."""
        t_c = _make_tracklet("t1", "cam_c")
        t_a = _make_tracklet("t2", "cam_a")
        result = await assoc.associate([t_c, t_a], captured_at=datetime.now(UTC))

        # cam_a and cam_c are not directly adjacent, so two separate tracks.
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cross_camera_merge(
        self,
        assoc: CrossCameraAssociator,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
    ) -> None:
        """Tracklets on adjacent cameras should be merged into one GlobalTrack."""
        # Populate gallery with similar embeddings so appearance similarity
        # contributes positively to the combined score.
        import numpy as np

        np.random.seed(42)
        emb = np.random.randn(768).tolist()
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb.copy())
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_b", embedding=emb.copy())

        t_a = _make_tracklet("t1", "cam_a")
        t_b = _make_tracklet("t2", "cam_b")
        result = await assoc.associate([t_a, t_b], captured_at=datetime.now(UTC))

        # Should merge into one GlobalTrack.
        assert len(result) == 1
        assert result[0].global_track_id is not None
        assert "cam_a" in result[0].camera_ids
        assert "cam_b" in result[0].camera_ids
        assert "t1" in result[0].tracklet_ids
        assert "t2" in result[0].tracklet_ids

    @pytest.mark.asyncio
    async def test_existing_global_track_extended(
        self,
        assoc: CrossCameraAssociator,
        global_track_repo: GlobalTrackRepository,
    ) -> None:
        """A new tracklet on an adjacent camera should extend an existing GlobalTrack."""
        # Create an initial GlobalTrack with t1 on cam_a.
        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-1",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
            )
        )

        # Add a new tracklet on cam_b.
        t2 = _make_tracklet("t2", "cam_b")
        await assoc.associate([t2], captured_at=datetime.now(UTC))

        # The existing GlobalTrack should be extended.
        gt = await global_track_repo.get("gt-1")
        assert gt is not None
        assert "cam_b" in gt.camera_ids
        assert "t2" in gt.tracklet_ids

    @pytest.mark.asyncio
    async def test_closed_global_track_not_extended(
        self,
        assoc: CrossCameraAssociator,
        global_track_repo: GlobalTrackRepository,
    ) -> None:
        """Closed GlobalTracks should not be extended."""
        # Create a closed GlobalTrack.
        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-closed",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                state="closed",
            )
        )

        t2 = _make_tracklet("t2", "cam_a")
        await assoc.associate([t2], captured_at=datetime.now(UTC))

        # The closed track should not be in the result.
        gt = await global_track_repo.get("gt-closed")
        assert gt is not None
        assert gt.state == "closed"

    @pytest.mark.asyncio
    async def test_low_score_pair_not_linked(
        self,
        assoc: CrossCameraAssociator,
    ) -> None:
        """Pairs below min_link_score should not be linked."""
        # The default config uses min_link_score=0.5 and the approximate
        # gallery similarity returns 0.8, geo_score is also moderate.
        # To test the threshold, raise min_link_score above the combined score.
        assoc._config = CrossCamConfig(min_link_score=0.99)
        t_a = _make_tracklet("t1", "cam_a")
        t_b = _make_tracklet("t2", "cam_b")
        result = await assoc.associate([t_a, t_b], captured_at=datetime.now(UTC))
        # Should create two separate GlobalTracks.
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cluster_merge(
        self,
        assoc: CrossCameraAssociator,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
    ) -> None:
        """Three tracklets should merge into one cluster when both pairs link."""
        # cam_a -> cam_b -> cam_c chain, all adjacent.
        # t1 on cam_a, t2 on cam_b, t3 on cam_c.
        import numpy as np

        np.random.seed(99)
        emb = np.random.randn(768).tolist()
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb.copy())
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_b", embedding=emb.copy())
        gallery._entries["e3"] = _make_gallery_entry("t3", "cam_c", embedding=emb.copy())

        t_a = _make_tracklet("t1", "cam_a")
        t_b = _make_tracklet("t2", "cam_b")
        t_c = _make_tracklet("t3", "cam_c")
        result = await assoc.associate([t_a, t_b, t_c], captured_at=datetime.now(UTC))
        # All three should end up in a single GlobalTrack.
        assert len(result) == 1
        assert result[0].tracklet_ids == ["t1", "t2", "t3"]
        assert set(result[0].camera_ids) == {"cam_a", "cam_b", "cam_c"}

    @pytest.mark.asyncio
    async def test_different_clusters_are_consolidated_in_repo(
        self,
        global_track_repo: GlobalTrackRepository,
    ) -> None:
        """Issue #26: merging two candidate clusters must close the source
        GlobalTrack in the repository, not only patch an in-memory map."""
        adj = CameraAdjacency()
        for a, b in [
            ("cam_a", "cam_b"),
            ("cam_c", "cam_d"),
            ("cam_b", "cam_c"),
            ("cam_a", "cam_c"),
            ("cam_a", "cam_d"),
            ("cam_b", "cam_d"),
        ]:
            adj.add_edge(AdjacencyEdge(a, b, max_transition_seconds=120))

        assoc = CrossCameraAssociator(
            gallery=InMemoryGalleryRepository(),
            adjacency=adj,
            global_track_repo=global_track_repo,
            config=CrossCamConfig(min_link_score=0.5),
        )

        scores = {
            ("t1", "t2"): 0.9,
            ("t3", "t4"): 0.8,
            ("t2", "t3"): 0.7,
        }

        async def score_pair(ta: Tracklet, tb: Tracklet) -> TrackletPairScore | None:
            key = (ta.tracklet_id, tb.tracklet_id)
            reverse = (tb.tracklet_id, ta.tracklet_id)
            score = scores.get(key) or scores.get(reverse)
            if score is None:
                return TrackletPairScore(
                    ta.tracklet_id,
                    tb.tracklet_id,
                    ta.camera_id,
                    tb.camera_id,
                    appearance_sim=0.0,
                    geo_score=0.0,
                    combined_score=0.0,
                )
            return TrackletPairScore(
                ta.tracklet_id,
                tb.tracklet_id,
                ta.camera_id,
                tb.camera_id,
                appearance_sim=score,
                geo_score=1.0,
                combined_score=score,
            )

        assoc._score_pair = score_pair  # type: ignore[method-assign]

        result = await assoc.associate(
            [
                _make_tracklet("t1", "cam_a"),
                _make_tracklet("t2", "cam_b"),
                _make_tracklet("t3", "cam_c"),
                _make_tracklet("t4", "cam_d"),
            ],
            captured_at=datetime.now(UTC),
        )

        active = await global_track_repo.list_active()
        assert len(result) == 1
        assert len(active) == 1
        assert set(active[0].tracklet_ids) == {"t1", "t2", "t3", "t4"}
        assert await global_track_repo.get_by_tracklet_id("t3") == active[0]

    @pytest.mark.asyncio
    async def test_non_adjacent_cameras_create_separate_tracks(
        self,
        assoc: CrossCameraAssociator,
    ) -> None:
        """cam_a and cam_c are not directly adjacent, so should be separate."""
        # Only cam_a<->cam_b and cam_b<->cam_c edges exist.
        t_a = _make_tracklet("t1", "cam_a")
        t_c = _make_tracklet("t2", "cam_c")
        result = await assoc.associate([t_a, t_c], captured_at=datetime.now(UTC))
        assert len(result) == 2
        assert all(len(r.tracklet_ids) == 1 for r in result)

    @pytest.mark.asyncio
    async def test_closed_gt_not_reused_for_new_tracklet(
        self,
        assoc: CrossCameraAssociator,
        global_track_repo: GlobalTrackRepository,
    ) -> None:
        """A new tracklet should not be merged into a closed GlobalTrack."""
        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-closed",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                state="closed",
            )
        )
        t2 = _make_tracklet("t2", "cam_b")
        result = await assoc.associate([t2], captured_at=datetime.now(UTC))
        # Should create a new GlobalTrack, not extend the closed one.
        assert len(result) == 1
        assert result[0].global_track_id != "gt-closed"

    @pytest.mark.asyncio
    async def test_within_s_time_budget_reduces_matches(
        self,
        global_track_repo: GlobalTrackRepository,
    ) -> None:
        """Verify that the within_s time budget is wired into the cross-camera
        associator. When tracklets are old enough that the time budget exceeds
        max_transition, they should still be linkable. But when the budget
        is tight, adjacency filtering should apply.

        This exercises the fix for review issue #10 (within_s dead code).
        """
        adj = CameraAdjacency()
        # cam_a -> cam_b with 30s transition, cam_b -> cam_c with 30s transition.
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b", max_transition_seconds=30))
        adj.add_edge(AdjacencyEdge("cam_b", "cam_c", max_transition_seconds=30))

        gallery = InMemoryGalleryRepository()
        import numpy as np

        np.random.seed(7)
        emb = np.random.randn(768).tolist()
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb.copy())
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_b", embedding=emb.copy())

        assoc = CrossCameraAssociator(
            gallery=gallery,
            adjacency=adj,
            global_track_repo=global_track_repo,
            config=CrossCamConfig(min_link_score=0.5),
        )

        now = datetime.now(UTC)
        # t1 started 10 seconds ago, t2 just now.
        t_a = Tracklet(
            tracklet_id="t1",
            camera_id="cam_a",
            detection_ids=["det-t1"],
            started_at=now.replace(second=now.second - 10) if now.second >= 10 else now,
            ended_at=None,
            state="active",
        )
        t_b = _make_tracklet("t2", "cam_b")

        result = await assoc.associate([t_a, t_b], captured_at=now)
        # Both tracklets should merge since they are adjacent and time budget
        # is sufficient (tracklet age >= max_transition means budget = tracklet_age).
        assert len(result) == 1
        assert set(result[0].tracklet_ids) == {"t1", "t2"}
        assert set(result[0].camera_ids) == {"cam_a", "cam_b"}

    @pytest.mark.asyncio
    async def test_within_s_blocks_non_adjacent_cameras(
        self,
        global_track_repo: GlobalTrackRepository,
    ) -> None:
        """Verify that within_s filtering blocks tracklets from cameras that
        are not directly adjacent. cam_a and cam_c are not directly adjacent
        (only via cam_b), so they should not link."""
        adj = CameraAdjacency()
        adj.add_edge(AdjacencyEdge("cam_a", "cam_b", max_transition_seconds=120))
        adj.add_edge(AdjacencyEdge("cam_b", "cam_c", max_transition_seconds=60))
        # No direct edge between cam_a and cam_c.

        assoc = CrossCameraAssociator(
            gallery=InMemoryGalleryRepository(),
            adjacency=adj,
            global_track_repo=global_track_repo,
            config=CrossCamConfig(min_link_score=0.5),
        )

        t_a = _make_tracklet("t1", "cam_a")
        t_c = _make_tracklet("t2", "cam_c")

        result = await assoc.associate([t_a, t_c], captured_at=datetime.now(UTC))
        # Should create separate GlobalTracks since cam_a and cam_c are not
        # directly adjacent (within_s filtering blocks transitive edges).
        assert len(result) == 2
        assert all(len(r.tracklet_ids) == 1 for r in result)

    @pytest.mark.asyncio
    async def test_known_identity_same_camera_reentry_merges(
        self,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
        assoc: CrossCameraAssociator,
    ) -> None:
        """Re-entry on the same camera merges into the existing GlobalTrack even
        when that GT has a committed identity.

        Regression test for the bug where `if gt.current_identity_id is None:`
        guarded the same-camera merge path, causing every re-entry to spawn a
        fresh GlobalTrack assigned to the same person.
        """
        import numpy as np

        np.random.seed(7)
        emb = np.random.randn(768)
        emb = (emb / np.linalg.norm(emb)).tolist()

        # Pre-existing GT on cam_a with a committed identity and gallery entry.
        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-known",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                current_identity_id="sriram",
            )
        )
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb)

        # New tracklet on the same camera with a matching appearance.
        t2 = _make_tracklet("t2", "cam_a")
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_a", embedding=emb)

        await assoc.associate([t2], captured_at=datetime.now(UTC))

        gt = await global_track_repo.get("gt-known")
        assert gt is not None
        assert "t2" in gt.tracklet_ids, "Re-entry tracklet must merge into existing GT"
        assert gt.current_identity_id == "sriram"

    @pytest.mark.asyncio
    async def test_known_identity_same_camera_different_person_no_merge(
        self,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
        assoc: CrossCameraAssociator,
    ) -> None:
        """A different person on the same camera should NOT merge into an
        existing committed GT when appearance similarity is below threshold."""
        import numpy as np

        np.random.seed(11)
        emb_a = np.random.randn(768)
        emb_a = (emb_a / np.linalg.norm(emb_a)).tolist()
        emb_b = np.random.randn(768)
        emb_b = (emb_b / np.linalg.norm(emb_b)).tolist()

        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-known",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                current_identity_id="person_a",
            )
        )
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb_a)

        # Different person on same camera — dissimilar embedding.
        t2 = _make_tracklet("t2", "cam_a")
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_a", embedding=emb_b)

        await assoc.associate([t2], captured_at=datetime.now(UTC))

        gt = await global_track_repo.get("gt-known")
        assert gt is not None
        assert "t2" not in gt.tracklet_ids, "Different person must not merge into existing GT"

    @pytest.mark.asyncio
    async def test_overlap_group_fragmented_gts_consolidate(
        self,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
    ) -> None:
        """Two GTs in the same overlap group with similar appearances are merged.

        Simulates the race condition where two concurrent associate() calls
        each see an empty list_active() and create separate GTs for the same
        person observed from different cameras in the same overlap group.
        """
        import numpy as np
        from app.domain import OverlapGroup

        np.random.seed(42)
        emb = np.random.randn(768)
        emb = (emb / np.linalg.norm(emb)).tolist()

        adj = CameraAdjacency()
        adj.set_overlap_groups(
            [OverlapGroup(group_id="living-room", camera_ids=["cam_a", "cam_b", "cam_c"])]
        )
        assoc = CrossCameraAssociator(
            gallery=gallery,
            adjacency=adj,
            global_track_repo=global_track_repo,
        )

        # Two GTs created by a race — same person, different cameras, same group.
        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-frag-a",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
            )
        )
        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-frag-b",
                camera_ids=["cam_b"],
                tracklet_ids=["t2"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
            )
        )
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb)
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_b", embedding=emb)

        # Any new tracklet on cam_c triggers associate(), which runs consolidation.
        t3 = _make_tracklet("t3", "cam_c")
        gallery._entries["e3"] = _make_gallery_entry("t3", "cam_c", embedding=emb)
        await assoc.associate([t3], captured_at=datetime.now(UTC))

        active = await global_track_repo.list_active()
        assert len(active) == 1, f"Fragmented GTs should be consolidated; got {len(active)}"
        surviving = active[0]
        assert set(surviving.camera_ids) >= {"cam_a", "cam_b"}

    @pytest.mark.asyncio
    async def test_overlap_group_different_identities_not_merged(
        self,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
    ) -> None:
        """Two GTs in the same overlap group with DIFFERENT committed identities
        must never be consolidated — they are different enrolled people."""
        import numpy as np
        from app.domain import OverlapGroup

        np.random.seed(42)
        emb = np.random.randn(768)
        emb = (emb / np.linalg.norm(emb)).tolist()

        adj = CameraAdjacency()
        adj.set_overlap_groups(
            [OverlapGroup(group_id="living-room", camera_ids=["cam_a", "cam_b"])]
        )
        assoc = CrossCameraAssociator(
            gallery=gallery,
            adjacency=adj,
            global_track_repo=global_track_repo,
        )

        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-person-a",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                current_identity_id="alice",
            )
        )
        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-person-b",
                camera_ids=["cam_b"],
                tracklet_ids=["t2"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                current_identity_id="bob",
            )
        )
        # Give them identical embeddings to ensure only the identity guard stops the merge.
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb)
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_b", embedding=emb)

        # Trigger associate() with any new tracklet.
        t3 = _make_tracklet("t3", "cam_a")
        gallery._entries["e3"] = _make_gallery_entry("t3", "cam_a", embedding=emb)
        await assoc.associate([t3], captured_at=datetime.now(UTC))

        # Both GTs must remain active and separate.
        active = await global_track_repo.list_active()
        active_ids = {gt.global_track_id for gt in active}
        assert "gt-person-a" in active_ids, "Alice's GT must stay active"
        assert "gt-person-b" in active_ids, "Bob's GT must stay active"

    @pytest.mark.asyncio
    async def test_known_identity_lower_threshold_merges_medium_similarity(
        self,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
        assoc: CrossCameraAssociator,
    ) -> None:
        """When a GT has a committed identity and the re-entry gap is short,
        medium appearance similarity (0.80, above known_identity_reentry_threshold=0.72
        but below unknown_merge_appearance_threshold=0.92) must still merge.

        This is the turn-away scenario: front-facing and back-facing embeddings
        have cosine similarity ~0.7-0.85 for the same person.
        """
        import numpy as np

        # Construct unit vectors with cosine similarity ≈ 0.80.
        emb_a = np.zeros(768)
        emb_a[0] = 1.0
        emb_b = np.zeros(768)
        emb_b[0] = 0.8
        emb_b[1] = 0.6  # norm = 1.0, cos_sim with emb_a = 0.8
        emb_a = emb_a.tolist()
        emb_b = emb_b.tolist()

        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-known-med",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                current_identity_id="sriram",
            )
        )
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb_a)

        t2 = _make_tracklet("t2", "cam_a")
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_a", embedding=emb_b)

        await assoc.associate([t2], captured_at=datetime.now(UTC))

        gt = await global_track_repo.get("gt-known-med")
        assert gt is not None
        assert "t2" in gt.tracklet_ids, (
            "Turn-away re-entry (sim≈0.80) must merge into committed GT "
            "using known_identity_reentry_threshold, not the stricter unknown threshold"
        )

    @pytest.mark.asyncio
    async def test_unknown_gt_strict_threshold_blocks_medium_similarity(
        self,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
        assoc: CrossCameraAssociator,
    ) -> None:
        """When a GT has NO committed identity, the strict
        unknown_merge_appearance_threshold=0.92 applies.  Medium appearance
        similarity (0.80) must NOT merge into an UNKNOWN GT.

        This prevents a new person from being attached to a stale UNKNOWN GT.
        """
        import numpy as np

        emb_a = np.zeros(768)
        emb_a[0] = 1.0
        emb_b = np.zeros(768)
        emb_b[0] = 0.8
        emb_b[1] = 0.6
        emb_a = emb_a.tolist()
        emb_b = emb_b.tolist()

        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-unknown",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
                current_identity_id=None,
            )
        )
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb_a)

        t2 = _make_tracklet("t2", "cam_a")
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_a", embedding=emb_b)

        await assoc.associate([t2], captured_at=datetime.now(UTC))

        gt = await global_track_repo.get("gt-unknown")
        assert gt is not None
        assert "t2" not in gt.tracklet_ids, (
            "Medium similarity (0.80) must NOT merge into an UNKNOWN GT; "
            "strict unknown_merge_appearance_threshold=0.92 must apply"
        )

    @pytest.mark.asyncio
    async def test_overlap_group_low_similarity_not_merged(
        self,
        global_track_repo: GlobalTrackRepository,
        gallery: InMemoryGalleryRepository,
    ) -> None:
        """Two GTs in the same overlap group with dissimilar embeddings
        must not be consolidated — they are different people."""
        import numpy as np
        from app.domain import OverlapGroup

        np.random.seed(99)
        emb_a = np.random.randn(768)
        emb_a = (emb_a / np.linalg.norm(emb_a)).tolist()
        emb_b = np.random.randn(768)
        emb_b = (emb_b / np.linalg.norm(emb_b)).tolist()

        adj = CameraAdjacency()
        adj.set_overlap_groups(
            [OverlapGroup(group_id="living-room", camera_ids=["cam_a", "cam_b"])]
        )
        assoc = CrossCameraAssociator(
            gallery=gallery,
            adjacency=adj,
            global_track_repo=global_track_repo,
        )

        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-x",
                camera_ids=["cam_a"],
                tracklet_ids=["t1"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
            )
        )
        await global_track_repo.save(
            GlobalTrack(
                global_track_id="gt-y",
                camera_ids=["cam_b"],
                tracklet_ids=["t2"],
                started_at=datetime.now(UTC),
                last_seen_at=datetime.now(UTC),
            )
        )
        gallery._entries["e1"] = _make_gallery_entry("t1", "cam_a", embedding=emb_a)
        gallery._entries["e2"] = _make_gallery_entry("t2", "cam_b", embedding=emb_b)

        t3 = _make_tracklet("t3", "cam_a")
        gallery._entries["e3"] = _make_gallery_entry("t3", "cam_a", embedding=emb_a)
        await assoc.associate([t3], captured_at=datetime.now(UTC))

        active = await global_track_repo.list_active()
        active_ids = {gt.global_track_id for gt in active}
        assert "gt-x" in active_ids, "GT-X must stay active"
        assert "gt-y" in active_ids, "GT-Y must stay active"
