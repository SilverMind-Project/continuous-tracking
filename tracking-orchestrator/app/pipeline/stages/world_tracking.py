"""World tracking stage: single floor-plane Kalman tracker for all cameras.

Replaces LocalTrackingStage + GlobalTrackingStage.
Wires TransitDetector and RoomTransitionPublisher.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from structlog import get_logger

from ...domain import (
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    OrientationBin,
    WorldObservation,
)
from ...tracking.world.config import WorldTrackerConfig
from ...tracking.world.tracker import WorldTracker, WorldTrackerResult
from ...tracking.world.transit_detector import TransitDetector
from ...transport.room_transition_publisher import RoomTransitionPublisher
from ..frame_context import FrameContext
from ._room_maps import camera_room_names, room_polygon_snapshot
from .base import FrameStage

if TYPE_CHECKING:
    from ...services.camera_room_map import CameraRoomMap, RoomPolygonMap
    from ...services.transit_zone_map import TransitZoneMap

logger = get_logger(__name__)

# Virtual room dimensions for cameras without homography calibration.
# Detections are mapped from normalised image coordinates to a 4m x 4m
# virtual floor.  Each camera occupies a distinct 200m x 200m tile so that
# cross-camera dedup (threshold: ~0.6 m) never confuses separate cameras.
_VIRTUAL_ROOM_M: float = 4.0
_CAMERA_TILE_M: float = 200.0


def _stable_camera_hash(camera_id: str) -> int:
    """Return a stable non-negative integer for any camera_id string.

    Uses a simple polynomial hash (not Python's built-in hash which is
    randomised per process via PYTHONHASHSEED).
    """
    h = 5381
    for ch in camera_id:
        h = ((h << 5) + h) ^ ord(ch)
    return h & 0xFFFF  # 16-bit, 0-65535


def _synthetic_floor_point(
    bbox: BoundingBox,
    frame_w: int,
    frame_h: int,
    camera_id: str,
) -> FloorPoint:
    """Build a virtual floor point for an uncalibrated camera detection.

    Maps the detection's bbox centre into a 4m x 4m virtual room and
    offsets the entire room by a deterministic per-camera tile so that
    different cameras never share the same virtual floor region.

    The returned FloorPoint carries calibrated=False to preserve the domain
    invariant, but x_mm/y_mm are non-zero virtual coordinates that keep the
    WorldTracker's Kalman filter and dedup logic working correctly until real
    homography is configured.
    """
    cam_h = _stable_camera_hash(camera_id)
    tile_x = (cam_h % 256) * _CAMERA_TILE_M
    tile_y = (cam_h >> 8) * _CAMERA_TILE_M

    cx = (bbox.x_min + bbox.x_max) / 2.0
    cy = (bbox.y_min + bbox.y_max) / 2.0
    norm_x = cx / max(frame_w, 1)
    norm_y = cy / max(frame_h, 1)

    x_m = tile_x + norm_x * _VIRTUAL_ROOM_M
    y_m = tile_y + norm_y * _VIRTUAL_ROOM_M
    return FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=False)


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
        camera_room_map: CameraRoomMap,
        room_polygon_map: RoomPolygonMap,
        config: WorldTrackerConfig | None = None,
        enabled: bool = True,
        transit_detector: TransitDetector | None = None,
        transit_zone_map: TransitZoneMap | None = None,
        room_transition_publisher: RoomTransitionPublisher | None = None,
        assertion_cache: object | None = None,
        anchor_match_window_s: float = 30.0,
        anchor_match_distance_m: float = 5.0,
        anchor_min_confidence: float = 0.5,
    ) -> None:
        self._tracker = tracker
        self._config = config or WorldTrackerConfig()
        self._room_polygon_map = room_polygon_map
        self._camera_room_map = camera_room_map
        self._enabled = enabled
        self._transit_detector = transit_detector
        self._transit_zone_map = transit_zone_map
        self._room_transition_publisher = room_transition_publisher
        self._assertion_cache = assertion_cache
        self._anchor_match_window_s = anchor_match_window_s
        self._anchor_match_distance_m = anchor_match_distance_m
        self._anchor_min_confidence = anchor_min_confidence
        self._missing_room_binding_warnings: set[str] = set()

    async def run(self, ctx: FrameContext) -> None:
        await self.run_many([ctx])

    async def run_many(self, contexts: list[FrameContext]) -> None:
        if not self._enabled:
            return
        if not contexts:
            return

        observations: list[WorldObservation] = []
        for ctx in contexts:
            frame_observations, uncalibrated_count = self._build_observations(ctx)
            observations.extend(frame_observations)
            if uncalibrated_count:
                logger.debug(
                    "world_tracking_synthetic_floor_points",
                    camera_id=ctx.frame.camera_id,
                    uncalibrated_count=uncalibrated_count,
                )

        batch_time = max((ctx.event_time for ctx in contexts), default=contexts[0].event_time)
        cc_face_anchors = await self._match_cc_assertions(
            observations=observations,
            now=batch_time,
        )
        all_face_anchors = [
            face_anchor for ctx in contexts for face_anchor in ctx.face_anchors
        ] + cc_face_anchors
        observations_with_faces = self._attach_face_anchors(observations, all_face_anchors)
        face_evidence = [
            face_evidence for ctx in contexts for face_evidence in (ctx._face_evidence or [])
        ]
        camera_ids = {ctx.frame.camera_id for ctx in contexts} | {
            obs.camera_id for obs in observations
        }
        camera_room_map = await camera_room_names(self._camera_room_map, camera_ids)
        for camera_id in sorted(camera_ids - set(camera_room_map)):
            if camera_id not in self._missing_room_binding_warnings:
                self._missing_room_binding_warnings.add(camera_id)
                logger.warning("world_tracking_camera_room_binding_missing", camera_id=camera_id)
        room_polygons, room_names = await room_polygon_snapshot(self._room_polygon_map)

        result = await self._tracker.step(
            observations=observations_with_faces,
            now=batch_time,
            room_polygons=room_polygons,
            camera_room_map=camera_room_map,
            room_names=room_names,
            face_anchors=all_face_anchors,
            face_evidence=face_evidence or None,
        )

        await self._publish_transit_events(result, batch_time)

        primary_context = contexts[0]
        for ctx in contexts:
            self._populate_context(ctx, result, include_revisions=ctx is primary_context)
            logger.debug(
                "world_tracking_frame",
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
                observations=sum(1 for obs in observations if obs.camera_id == ctx.frame.camera_id),
                active_phs=len(result.updated_phs),
                snapshots=len(result.snapshots),
                continuations=len(result.continuations),
            )

    def _build_observations(self, ctx: FrameContext) -> tuple[list[WorldObservation], int]:
        # Build observations first (without face anchors), so we can use
        # their floor positions to match CC assertions.
        # For cameras without homography calibration, generate a synthetic
        # virtual floor point from the bbox centre so that the WorldTracker
        # can still create PersonHypotheses and commit face-based identities.
        observations: list[WorldObservation] = []
        uncalibrated_count = 0
        for det in ctx.domain_detections:
            fp = det.floor_point
            if not fp.calibrated:
                fp = _synthetic_floor_point(
                    det.bbox, ctx.effective_width, ctx.effective_height, ctx.frame.camera_id
                )
                uncalibrated_count += 1
            observations.append(
                WorldObservation(
                    camera_id=det.camera_id,
                    frame_index=ctx.frame.frame_index,
                    captured_at=det.capture_time if det.capture_time else ctx.event_time,
                    floor_point=fp,
                    bbox=det.bbox,
                    embedding=det.embedding,
                    detection_confidence=det.confidence,
                    height_estimate_m=None,
                    face_anchor=None,
                    detection_id=det.detection_id,
                    quality=det.crop_quality,
                    floor_residual_m=det.floor_residual_m if fp.calibrated else None,
                    orientation=ctx.orientation_by_detection.get(
                        det.detection_id, (OrientationBin.UNKNOWN, 0.0)
                    )[0],
                    orientation_confidence=ctx.orientation_by_detection.get(
                        det.detection_id, (OrientationBin.UNKNOWN, 0.0)
                    )[1],
                )
            )
        return observations, uncalibrated_count

    async def _match_cc_assertions(
        self,
        *,
        observations: list[WorldObservation],
        now: datetime,
    ) -> list[FaceAnchor]:
        from ...tracking.world.assertion_matching import match_assertions_to_face_anchors

        # Match CC assertions to observations (spatial + temporal + confidence gate).
        cc_face_anchors: list[FaceAnchor] = []
        if self._assertion_cache is not None:
            try:
                recent_assertions = await self._assertion_cache.get_recent()  # type: ignore[attr-defined]
                cc_face_anchors = match_assertions_to_face_anchors(
                    assertions=recent_assertions,
                    observations=observations,
                    now=now,
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
        return cc_face_anchors

    def _attach_face_anchors(
        self,
        observations: list[WorldObservation],
        all_face_anchors: list[FaceAnchor],
    ) -> list[WorldObservation]:
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
                    quality=obs.quality,
                    floor_residual_m=obs.floor_residual_m,
                    orientation=obs.orientation,
                    orientation_confidence=obs.orientation_confidence,
                )
            )
        return observations_with_faces

    async def _publish_transit_events(
        self, result: WorldTrackerResult, event_time: datetime
    ) -> None:
        # Detect transit zone crossings for each active PH.
        if (
            self._transit_detector is None
            or self._transit_zone_map is None
            or self._room_transition_publisher is None
        ):
            return

        zones = await self._transit_zone_map.snapshot()
        if zones:
            # Build ph_id -> current_identity_id lookup for publishing.
            ph_identity: dict[str, str | None] = {
                ph.ph_id: ph.current_identity_id for ph in result.updated_phs
            }
            for ph in result.updated_phs:
                events = self._transit_detector.check(
                    ph_id=ph.ph_id,
                    floor_x_m=ph.state_mean[0],
                    floor_y_m=ph.state_mean[1],
                    zones=zones,
                    now=event_time,
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

        # Clean up transit detector state for closed PHs. Zone edits are read
        # live from TransitZoneMap; per-PH last-position state may span an edit,
        # which can at worst affect the first post-edit movement sample.
        for ph in result.updated_phs:
            if ph.closed_at is not None:
                self._transit_detector.remove_ph(ph.ph_id)

    @staticmethod
    def _populate_context(
        ctx: FrameContext, result: WorldTrackerResult, *, include_revisions: bool
    ) -> None:
        # Populate frame context for downstream stages (PH-native).
        ctx.active_ph_ids = {ph.ph_id for ph in result.updated_phs if ph.closed_at is None}
        ctx.outcome_decisions = list(result.identity_decisions)
        ctx.new_revisions = list(result.revisions) if include_revisions else []
        ctx.committed_ids = {ph.ph_id: ph.current_identity_id for ph in result.updated_phs}
        # born_at per PH so RevisionsStage can set applies_from to track start.
        ctx.ph_born_at_by_id = {ph.ph_id: ph.born_at for ph in result.updated_phs}
        # revived PH ids so ClosePHStage and TrajectoryStage can reconcile.
        ctx.revived_ph_ids = result.revived_ph_ids

        # Store snapshots on the context for downstream stages.
        ctx.world_snapshots = list(result.snapshots)
        ctx.det_to_ph = dict(result.det_to_ph)
