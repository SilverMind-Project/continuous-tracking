"""World tracking stage: single floor-plane Kalman tracker for all cameras.

Replaces LocalTrackingStage + GlobalTrackingStage (M1 refactor).
"""

from __future__ import annotations

from structlog import get_logger

from ...tracking.world.config import WorldTrackerConfig
from ...tracking.world.tracker import WorldTracker
from ..frame_context import FrameContext
from .base import FrameStage

logger = get_logger(__name__)


class WorldTrackingStage(FrameStage):
    """Pipeline stage that runs the world-coordinate person tracker.

    Consumes detections with calibrated floor points from
    SpatialProjectionStage and produces PersonHypothesis updates
    consumed by downstream stages.
    """

    name = "world_tracking"

    def __init__(
        self,
        tracker: WorldTracker,
        config: WorldTrackerConfig | None = None,
        room_polygons: dict[str, list[tuple[float, float]]] | None = None,
        camera_room_map: dict[str, str] | None = None,
        enabled: bool = True,
    ) -> None:
        self._tracker = tracker
        self._config = config or WorldTrackerConfig()
        self._room_polygons = room_polygons or {}
        self._camera_room_map = camera_room_map or {}
        self._enabled = enabled

    async def run(self, ctx: FrameContext) -> None:
        if not self._enabled:
            return

        from ...domain import WorldObservation

        # Build WorldObservation list from this frame's calibrated detections.
        observations: list[WorldObservation] = []
        for det in ctx.domain_detections:
            if not det.floor_point.calibrated:
                continue
            face_anchor = None
            for fa in ctx.face_anchors:
                if fa.tracklet_id == det.tracklet_id and fa.tracklet_id:
                    face_anchor = fa
                    break
            observations.append(
                WorldObservation(
                    camera_id=det.camera_id,
                    frame_index=ctx.frame.frame_index,
                    captured_at=det.capture_time if det.capture_time else ctx.event_time,
                    floor_point=det.floor_point,
                    bbox=det.bbox,
                    embedding=det.embedding,
                    detection_confidence=det.confidence,
                    height_estimate_m=None,
                    face_anchor=face_anchor,
                )
            )

        # Run the world tracker.
        result = await self._tracker.step(
            observations=observations,
            now=ctx.event_time,
            room_polygons=self._room_polygons,
            camera_room_map=self._camera_room_map,
        )

        # Populate frame context for downstream stages.
        ctx.active_global_tracks = []  # legacy field; downstream stages use snapshots
        ctx.outcome_decisions = []
        ctx.new_revisions = []
        ctx.committed_ids = {ph.ph_id: ph.current_identity_id for ph in result.updated_phs}

        # Store snapshots on the context for downstream stages.
        ctx._world_snapshots = list(result.snapshots)

        logger.debug(
            "world_tracking_frame",
            camera_id=ctx.frame.camera_id,
            frame_index=ctx.frame.frame_index,
            observations=len(observations),
            active_phs=len(result.updated_phs),
            snapshots=len(result.snapshots),
            continuations=len(result.continuations),
        )
