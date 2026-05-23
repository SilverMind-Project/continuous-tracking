"""Local tracking stage: updates per-camera tracker and tracklet manager."""

from __future__ import annotations

from ...domain import CameraConfig
from ...observability import metrics as _metrics
from ...tracking.tracker import PerCameraTrackers
from ...tracking.tracklet_manager import TrackletManager
from ..frame_context import FrameContext
from .base import FrameStage


class LocalTrackingStage(FrameStage):
    name = "local_tracking"

    def __init__(
        self,
        tracker: PerCameraTrackers,
        tracklet_manager: TrackletManager,
    ) -> None:
        self._tracker = tracker
        self._tracklet_manager = tracklet_manager

    async def run(self, ctx: FrameContext) -> None:
        local_tracks = self._tracker.update(
            camera_id=ctx.frame.camera_id,
            detections=ctx.domain_detections,
            embeddings=ctx.embeddings or None,
            frame_index=ctx.frame.frame_index,
        )
        dedup_dropped = self._tracker.get_dedup_dropped(ctx.frame.camera_id)
        if dedup_dropped > 0:
            _metrics.metrics.tracklets_dedup_dropped_total.labels(
                camera_id=ctx.frame.camera_id
            ).inc(dedup_dropped)

        camera_config = CameraConfig(camera_id=ctx.frame.camera_id)
        await self._tracklet_manager.step(
            camera=camera_config,
            local_tracks=local_tracks,
            detections=ctx.domain_detections,
            embeddings=ctx.embeddings,
            event_time=ctx.event_time,
            frame_index=ctx.frame.frame_index,
        )
