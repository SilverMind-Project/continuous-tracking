"""Detection backfill stage: stamps tracklet_id and global_track_id on detections."""

from __future__ import annotations

from dataclasses import replace

from ..frame_context import FrameContext
from .base import FrameStage


class DetectionBackfillStage(FrameStage):
    name = "detection_backfill"

    # N0: was TrackletManager, deleted
    def __init__(self, tracklet_manager: object | None = None) -> None:
        self._tracklet_manager = tracklet_manager

    async def run(self, ctx: FrameContext) -> None:
        if (
            not ctx.domain_detections
            or self._tracklet_manager is None
            or not ctx.active_global_tracks
        ):
            return

        tracklet_to_gt: dict[str, str] = {}
        for gt in ctx.active_global_tracks:
            for tid in gt.tracklet_ids:
                tracklet_to_gt[tid] = gt.global_track_id

        updated = []
        for domain_det in ctx.domain_detections:
            tid = self._tracklet_manager.get_tracklet_id_for_detection(domain_det.detection_id)  # type: ignore[attr-defined]
            gt_id = tracklet_to_gt.get(tid, "")
            updated.append(replace(domain_det, tracklet_id=tid, global_track_id=gt_id))
        ctx.domain_detections = updated
