"""World tracking stage: single floor-plane Kalman tracker for all cameras.

Replaces LocalTrackingStage + GlobalTrackingStage (M1 refactor).
Wires TransitDetector and RoomTransitionPublisher (M2).
"""

from __future__ import annotations

from structlog import get_logger

from ...domain import GlobalTrack, PersonHypothesis
from ...tracking.world.config import WorldTrackerConfig
from ...tracking.world.tracker import WorldTracker
from ...tracking.world.transit_detector import TransitDetector, TransitZone
from ...transport.room_transition_publisher import RoomTransitionPublisher
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
        transit_detector: TransitDetector | None = None,
        transit_zones: list[TransitZone] | None = None,
        room_transition_publisher: RoomTransitionPublisher | None = None,
    ) -> None:
        self._tracker = tracker
        self._config = config or WorldTrackerConfig()
        self._room_polygons = room_polygons or {}
        self._camera_room_map = camera_room_map or {}
        self._enabled = enabled
        self._transit_detector = transit_detector
        self._transit_zones = transit_zones or []
        self._room_transition_publisher = room_transition_publisher

    async def run(self, ctx: FrameContext) -> None:
        if not self._enabled:
            return

        from ...domain import FaceAnchor, WorldObservation

        # Build WorldObservation list from this frame's calibrated detections.
        # Match face anchors by camera_id (per-camera tracklet_id is deprecated
        # in M1 since the per-camera tracker no longer runs).
        face_by_camera: dict[str, FaceAnchor] = {}
        for fa in ctx.face_anchors:
            key = fa.camera_id if fa.camera_id else fa.tracklet_id
            if key:
                face_by_camera[key] = fa
        observations: list[WorldObservation] = []
        for det in ctx.domain_detections:
            if not det.floor_point.calibrated:
                continue
            face_anchor = face_by_camera.get(det.camera_id)
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
            face_anchors=ctx.face_anchors if ctx.face_anchors else None,
        )

        # M2: detect transit zone crossings for each active PH.
        if (
            self._transit_detector is not None
            and self._transit_zones
            and self._room_transition_publisher is not None
        ):
            for ph in result.updated_phs:
                events = self._transit_detector.check(
                    ph_id=ph.ph_id,
                    floor_x_m=ph.state_mean[0],
                    floor_y_m=ph.state_mean[1],
                    zones=self._transit_zones,
                    now=ctx.event_time,
                )
                for event in events:
                    try:
                        from ...domain import RoomTransitionEvent as DomainTransition

                        domain_event = DomainTransition(
                            ph_id=event.ph_id,
                            transit_zone_id=event.transit_zone_id,
                            direction=event.direction,
                            inside_room_id=event.inside_room_id,
                            outside_room_id=event.outside_room_id,
                            floor_x_m=event.floor_x_m,
                            floor_y_m=event.floor_y_m,
                            event_time=event.event_time,
                        )
                        await self._room_transition_publisher.publish(domain_event)
                    except Exception:
                        logger.exception(
                            "room_transition_publish_error",
                            ph_id=ph.ph_id,
                        )
            # Clean up transit detector state for closed PHs.
            for ph in result.updated_phs:
                if ph.closed_at is not None:
                    self._transit_detector.remove_ph(ph.ph_id)

        # Populate frame context for downstream stages.
        ctx.active_global_tracks = _phs_to_global_tracks(result.updated_phs)
        ctx.outcome_decisions = list(result.identity_decisions)
        ctx.new_revisions = list(result.revisions)
        ctx.committed_ids = {ph.ph_id: ph.current_identity_id for ph in result.updated_phs}

        # Store snapshots on the context for downstream stages.
        ctx.world_snapshots = list(result.snapshots)

        logger.debug(
            "world_tracking_frame",
            camera_id=ctx.frame.camera_id,
            frame_index=ctx.frame.frame_index,
            observations=len(observations),
            active_phs=len(result.updated_phs),
            snapshots=len(result.snapshots),
            continuations=len(result.continuations),
        )


def _phs_to_global_tracks(phs: list[PersonHypothesis]) -> list[GlobalTrack]:
    """Build a transitional GlobalTrack view for legacy stages.

    Open PHs (closed_at is None) only — closed PHs must not appear in
    the active list, otherwise CloseTerminatedStage cannot detect them as
    terminated next frame.
    """
    out: list[GlobalTrack] = []
    for ph in phs:
        if ph.closed_at is not None:
            continue
        out.append(
            GlobalTrack(
                global_track_id=ph.ph_id,
                tracklet_ids=[],
                camera_ids=list(ph.active_cameras),
                started_at=ph.born_at,
                last_seen_at=ph.last_seen_at,
                current_identity_id=ph.current_identity_id,
                current_identity_committed_at=ph.current_identity_committed_at,
                state="active",
            )
        )
    return out
