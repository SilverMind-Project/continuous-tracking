"""Spatial projection stage: attaches calibrated floor points to detections.

Runs after privacy filtering and before ReID/pose so every domain detection
carries a ``floor_point`` before tracking consumes it.
"""

from __future__ import annotations

from ...domain import BoundingBox, FloorPoint
from ...tracking.spatial_projection import SpatialProjectionService
from ..frame_context import FrameContext
from .base import FrameStage


class SpatialProjectionStage(FrameStage):
    name = "spatial_projection"

    def __init__(self, projection_service: SpatialProjectionService) -> None:
        self._projection = projection_service

    async def run(self, ctx: FrameContext) -> None:
        ew = ctx.effective_width
        eh = ctx.effective_height
        floor_points: dict[int, FloorPoint] = {}
        for idx, det in enumerate(ctx.raw_detections):
            bbox = BoundingBox(
                x_min=int(det.x1 * ew),
                y_min=int(det.y1 * eh),
                x_max=int(det.x2 * ew),
                y_max=int(det.y2 * eh),
            )
            floor_points[idx] = self._projection.project_detection(ctx.frame.camera_id, bbox)
        ctx._floor_points_by_index = floor_points
