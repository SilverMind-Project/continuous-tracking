"""Tests for CrossCameraAssociator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import (
    CameraId,
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
from app.tracking.cross_camera import CrossCamConfig, CrossCameraAssociator


@pytest.fixture()
def adjacency() -> CameraAdjacency:
    adj = CameraAdjacency()
    adj.add_edge(AdjacencyEdge("cam_a", "cam_b", max_transition_seconds=120))
    adj.add_edge(AdjacencyEdge("cam_b", "cam_c", max_transition_seconds=60))
    return adj


@pytest.fixture()
def global_track_repo() -> GlobalTrackRepository:
    return InMemoryGlobalTrackRepository()


@pytest.fixture()
def assoc(
    adjacency: CameraAdjacency,
    global_track_repo: GlobalTrackRepository,
) -> CrossCameraAssociator:
    return CrossCameraAssociator(
        gallery=InMemoryGalleryRepository(),
        adjacency=adjacency,
        global_track_repo=global_track_repo,
        config=CrossCamConfig(
            alpha=0.7,
            min_link_score=0.5,
        ),
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
    ) -> None:
        """Tracklets on adjacent cameras should be merged into one GlobalTrack."""
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
    ) -> None:
        """Three tracklets should merge into one cluster when both pairs link."""
        # cam_a -> cam_b -> cam_c chain, all adjacent.
        # t1 on cam_a, t2 on cam_b, t3 on cam_c.
        t_a = _make_tracklet("t1", "cam_a")
        t_b = _make_tracklet("t2", "cam_b")
        t_c = _make_tracklet("t3", "cam_c")
        result = await assoc.associate([t_a, t_b, t_c], captured_at=datetime.now(UTC))
        # All three should end up in a single GlobalTrack.
        assert len(result) == 1
        assert result[0].tracklet_ids == ["t1", "t2", "t3"]
        assert set(result[0].camera_ids) == {"cam_a", "cam_b", "cam_c"}

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
