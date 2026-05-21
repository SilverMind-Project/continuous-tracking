"""TrackletUnmergeService: detaches a tracklet from its global track.

After detachment:
- A do_not_fuse hint is added so the cross-camera associator
  does not re-associate them.
- A new GlobalTrack is created for the detached tracklet.
- Historical trajectory rows are NOT rewritten — only future
  observations will use the new GlobalTrack.
"""

from __future__ import annotations

from dataclasses import dataclass

from structlog import get_logger

from ..storage.base import DoNotFuseRepository, GlobalTrackRepository

log = get_logger(__name__)


@dataclass(frozen=True)
class UnmergeResult:
    """Result of a successful tracklet unmerge."""

    tracklet_id: str
    original_global_track_id: str
    new_global_track_id: str


class TrackletUnmergeService:
    """Detaches a tracklet from its current global track."""

    def __init__(
        self,
        global_track_repo: GlobalTrackRepository,
        dnf_repo: DoNotFuseRepository,
    ) -> None:
        self._gt_repo = global_track_repo
        self._dnf_repo = dnf_repo

    async def unmerge(
        self,
        tracklet_id: str,
        requested_by: str,
    ) -> UnmergeResult:
        """Detach tracklet from its current global track and create a new one.

        Returns the UnmergeResult with both old and new global track IDs.
        Raises ValueError if the tracklet has no global track to unmerge from.
        """
        # Find the parent global track via the reverse index.
        parent_gt = await self._gt_repo.get_by_tracklet_id(tracklet_id)
        if parent_gt is None:
            raise ValueError(f"Tracklet {tracklet_id} is not associated with any global track")

        original_gt_id = parent_gt.global_track_id

        # 1. Add do_not_fuse hint first, before any other writes.
        await self._dnf_repo.add_hint(
            tracklet_id=tracklet_id,
            global_track_id=original_gt_id,
            created_by=requested_by,
        )

        # 2. Create a new GlobalTrack for this tracklet.
        # We need to infer camera_id from the parent GT.
        # The parent GT's camera_ids are ordered the same as tracklet_ids.
        tl_idx = parent_gt.tracklet_ids.index(tracklet_id)
        camera_id = (
            parent_gt.camera_ids[tl_idx]
            if tl_idx < len(parent_gt.camera_ids)
            else parent_gt.camera_ids[0]
        )
        new_gt = await self._gt_repo.merge_tracklets(
            tracklet_ids=[tracklet_id],
            camera_ids=[camera_id],
        )

        # 3. Remove the tracklet from the original global track.
        await self._gt_repo.remove_tracklet(
            global_track_id=original_gt_id,
            tracklet_id=tracklet_id,
        )

        log.info(
            "tracklet_unmerged",
            tracklet_id=tracklet_id,
            original_global_track_id=original_gt_id,
            new_global_track_id=new_gt.global_track_id,
            requested_by=requested_by,
        )

        return UnmergeResult(
            tracklet_id=tracklet_id,
            original_global_track_id=original_gt_id,
            new_global_track_id=new_gt.global_track_id,
        )
