"""Trail maintenance stage — maintains per-PH foot-position trails."""

from __future__ import annotations

from collections import deque

from ..frame_context import FrameContext
from .base import FrameStage


class TrailsStage(FrameStage):
    """Maintains per-PH foot-position trails and snapshots them into ctx."""

    name = "trails"

    def __init__(
        self,
        trail_by_tracklet: dict[str, deque[tuple[float, float]]] | None = None,
        trail_maxlen: int = 12,
    ) -> None:
        # Parameter kept for backward compat with frame_pipeline.py.
        self._trail_by_ph: dict[str, deque[tuple[float, float]]] = (
            trail_by_tracklet if trail_by_tracklet is not None else {}
        )
        self._TRAIL_MAXLEN = trail_maxlen

    async def run(self, ctx: FrameContext) -> None:
        frame_w = float(ctx.effective_width) if ctx.effective_width else 1.0
        frame_h = float(ctx.effective_height) if ctx.effective_height else 1.0
        for domain_det in ctx.domain_detections:
            ph_id = domain_det.ph_id
            if not ph_id:
                continue
            foot_x = (domain_det.bbox.x_min + domain_det.bbox.x_max) / 2.0 / frame_w
            foot_y = domain_det.bbox.y_max / frame_h
            trail_dq = self._trail_by_ph.get(ph_id)
            if trail_dq is None:
                trail_dq = deque(maxlen=self._TRAIL_MAXLEN)
                self._trail_by_ph[ph_id] = trail_dq
            trail_dq.append((float(foot_x), float(foot_y)))

        active_phs = ctx.active_ph_ids or {d.ph_id for d in ctx.domain_detections if d.ph_id}
        stale_phs = set(self._trail_by_ph) - active_phs
        for pid in stale_phs:
            del self._trail_by_ph[pid]

        ctx.trail_by_tracklet_snapshot = {pid: list(dq) for pid, dq in self._trail_by_ph.items()}
        # PH-native alias (trail_by_tracklet_snapshot kept for PublishStage compat).
        ctx.trail_by_ph_snapshot = ctx.trail_by_tracklet_snapshot
