"""Posture and trails stage: classifies posture and maintains per-tracklet trails."""

from __future__ import annotations

from collections import deque

from ...trajectory.posture import classify_posture
from ...trajectory.posture_strategy import PostureStrategy
from ..frame_context import FrameContext
from .base import FrameStage


class PostureAndTrailsStage(FrameStage):
    name = "posture_and_trails"

    def __init__(
        self,
        posture_strategy: PostureStrategy | None = None,
        trail_by_tracklet: dict[str, deque[tuple[float, float]]] | None = None,
        trail_maxlen: int = 12,
    ) -> None:
        self._posture_strategy = posture_strategy
        # Shared mutable state (owned by pipeline).
        self._trail_by_tracklet: dict[str, deque[tuple[float, float]]] = (
            trail_by_tracklet if trail_by_tracklet is not None else {}
        )
        self._TRAIL_MAXLEN = trail_maxlen

    async def run(self, ctx: FrameContext) -> None:
        for domain_det in ctx.domain_detections:
            if domain_det.detection_id in ctx.det_posture:
                continue
            pose_result = ctx.det_pose_result.get(domain_det.detection_id)
            if self._posture_strategy is not None:
                image = ctx.require_image()
                posture = await self._posture_strategy.infer(image, domain_det, pose_result)
            elif pose_result is not None:
                posture = classify_posture(pose_result, domain_det.bbox)
            else:
                posture = "unknown"
            ctx.det_posture[domain_det.detection_id] = posture

        frame_w = float(ctx.effective_width) if ctx.effective_width else 1.0
        frame_h = float(ctx.effective_height) if ctx.effective_height else 1.0
        for domain_det in ctx.domain_detections:
            if not domain_det.tracklet_id:
                continue
            foot_x = (domain_det.bbox.x_min + domain_det.bbox.x_max) / 2.0 / frame_w
            foot_y = domain_det.bbox.y_max / frame_h
            trail_dq = self._trail_by_tracklet.get(domain_det.tracklet_id)
            if trail_dq is None:
                trail_dq = deque(maxlen=self._TRAIL_MAXLEN)
                self._trail_by_tracklet[domain_det.tracklet_id] = trail_dq
            trail_dq.append((float(foot_x), float(foot_y)))

        active_tids = {d.tracklet_id for d in ctx.domain_detections if d.tracklet_id}
        stale_tids = set(self._trail_by_tracklet) - active_tids
        for tid in stale_tids:
            del self._trail_by_tracklet[tid]

        ctx.trail_by_tracklet_snapshot = {
            tid: list(dq) for tid, dq in self._trail_by_tracklet.items()
        }
