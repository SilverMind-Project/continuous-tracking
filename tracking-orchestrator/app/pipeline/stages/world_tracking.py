"""World tracking stage: single floor-plane Kalman tracker for all cameras.

Replaces LocalTrackingStage + GlobalTrackingStage (M1 refactor).
Wires TransitDetector and RoomTransitionPublisher (M2).
"""

from __future__ import annotations

from structlog import get_logger

from ...domain import GlobalTrack, PersonHypothesis, TransitZone
from ...tracking.world.config import WorldTrackerConfig
from ...tracking.world.tracker import WorldTracker
from ...tracking.world.transit_detector import TransitDetector
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
        assertion_cache: object | None = None,
        anchor_match_window_s: float = 30.0,
        anchor_match_distance_m: float = 5.0,
        anchor_min_confidence: float = 0.5,
    ) -> None:
        self._tracker = tracker
        self._config = config or WorldTrackerConfig()
        self._room_polygons = room_polygons or {}
        self._camera_room_map = camera_room_map or {}
        self._enabled = enabled
        self._transit_detector = transit_detector
        self._transit_zones = transit_zones or []
        self._room_transition_publisher = room_transition_publisher
        self._assertion_cache = assertion_cache
        self._anchor_match_window_s = anchor_match_window_s
        self._anchor_match_distance_m = anchor_match_distance_m
        self._anchor_min_confidence = anchor_min_confidence

    async def run(self, ctx: FrameContext) -> None:
        if not self._enabled:
            return

        from ...domain import FaceAnchor, WorldObservation
        from ...tracking.world.assertion_matching import match_assertions_to_face_anchors

        # Build observations first (without face anchors), so we can use
        # their floor positions to match CC assertions.
        observations: list[WorldObservation] = []
        for det in ctx.domain_detections:
            if not det.floor_point.calibrated:
                continue
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
                    face_anchor=None,
                    detection_id=det.detection_id,
                )
            )

        # Match CC assertions to observations (spatial + temporal + confidence gate).
        cc_face_anchors: list[FaceAnchor] = []
        if self._assertion_cache is not None:
            try:
                recent_assertions = await self._assertion_cache.get_recent()  # type: ignore[attr-defined]
                cc_face_anchors = match_assertions_to_face_anchors(
                    assertions=recent_assertions,
                    observations=observations,
                    now=ctx.event_time,
                    anchor_match_window_s=self._anchor_match_window_s,
                    anchor_match_distance_m=self._anchor_match_distance_m,
                    anchor_min_confidence=self._anchor_min_confidence,
                )
                if cc_face_anchors:
                    logger.debug(
                        "cc_assertions_matched",
                        matched=len(cc_face_anchors),
                        assertions_checked=len(recent_assertions),
                    )
            except Exception:
                logger.exception("cc_assertion_matching_failed")

        # Merge direct face anchors (from FaceIdentityStage) with CC assertion anchors.
        all_face_anchors = list(ctx.face_anchors) + cc_face_anchors

        # Map face anchors to observations (detection_id primary, camera_id fallback).
        face_by_detection: dict[str, FaceAnchor] = {}
        face_by_camera: dict[str, FaceAnchor] = {}
        for fa in all_face_anchors:
            if fa.detection_id:
                face_by_detection[fa.detection_id] = fa
            key = fa.camera_id if fa.camera_id else fa.tracklet_id
            if key:
                face_by_camera[key] = fa

        # Rebuild observations with matched face anchors.
        observations_with_faces: list[WorldObservation] = []
        for obs in observations:
            face_anchor = face_by_detection.get(obs.detection_id)
            if face_anchor is None:
                face_anchor = face_by_camera.get(obs.camera_id)
            observations_with_faces.append(
                WorldObservation(
                    camera_id=obs.camera_id,
                    frame_index=obs.frame_index,
                    captured_at=obs.captured_at,
                    floor_point=obs.floor_point,
                    bbox=obs.bbox,
                    embedding=obs.embedding,
                    detection_confidence=obs.detection_confidence,
                    height_estimate_m=obs.height_estimate_m,
                    face_anchor=face_anchor,
                    detection_id=obs.detection_id,
                )
            )

        # Run the world tracker with combined face anchors.
        result = await self._tracker.step(
            observations=observations_with_faces,
            now=ctx.event_time,
            room_polygons=self._room_polygons,
            camera_room_map=self._camera_room_map,
            face_anchors=all_face_anchors,
        )

        # M2: detect transit zone crossings for each active PH (WTR5).
        if (
            self._transit_detector is not None
            and self._transit_zones
            and self._room_transition_publisher is not None
        ):
            # Build ph_id -> current_identity_id lookup for publishing.
            ph_identity: dict[str, str | None] = {
                ph.ph_id: ph.current_identity_id for ph in result.updated_phs
            }
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
                        identity_id = ph_identity.get(event.ph_id)
                        await self._room_transition_publisher.publish(
                            event, identity_id=identity_id
                        )
                    except Exception:
                        logger.exception(
                            "room_transition_publish_error",
                            ph_id=ph.ph_id,
                        )
            # Clean up transit detector state for closed PHs.
            for ph in result.updated_phs:
                if ph.closed_at is not None:
                    self._transit_detector.remove_ph(ph.ph_id)

        # Populate frame context for downstream stages (PH-native, WTR3).
        ctx.active_ph_ids = {ph.ph_id for ph in result.updated_phs if ph.closed_at is None}
        # Legacy bridge: build GlobalTrack views from open PHs for stages
        # not yet migrated (deprecated — remove in WTR9).
        ctx.active_global_tracks = _phs_to_global_tracks(result.updated_phs)
        ctx.outcome_decisions = list(result.identity_decisions)
        ctx.new_revisions = list(result.revisions)
        ctx.committed_ids = {ph.ph_id: ph.current_identity_id for ph in result.updated_phs}

        # Store snapshots on the context for downstream stages.
        ctx.world_snapshots = list(result.snapshots)
        ctx.det_to_ph = dict(result.det_to_ph)

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
