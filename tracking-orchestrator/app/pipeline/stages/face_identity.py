"""Face identity stage: identifies faces via person-identification-service."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ...domain import (
    BoundingBox,
    Detection,
    FaceAnchor,
    FaceGateOutcome,
    FaceObservationStat,
    Identity,
)
from ...inference.evidence import FaceEvidence
from ...inference.face_id_client import (
    SINGLE_FRAME_PROFILE,
    CaptureContext,
    CropQueryResult,
    FaceIdentificationClient,
    FaceResult,
)
from ...observability import metrics as _metrics
from ...storage.base import GalleryRepository
from ...storage.face_stats import FaceObservationStatsRepository
from ..crops import FaceCropPadding, crop_detection_from_bbox, face_crop_box
from ..face_attribution import FaceAttribution, HeadRegionPolicy, assign_face_to_person
from ..frame_context import FrameContext
from ..types import FaceIdCameraConfig
from .base import FrameStage

logger = get_logger(__name__)

# IoU between a current detection and a previous-frame track required to
# inherit the track's throttle key.  Below this the detection is treated as
# a new person (fresh face-id call immediately).
_IOU_TRACK_THRESHOLD: float = 0.8

# Mutual IoU between two detections in the *same* frame that signals a
# potential crossing.  Both detections bypass the cooldown so that
# identity consistency is preserved through the overlap.
_IOU_CROSSING_THRESHOLD: float = 0.25


@dataclass
class _TrackEntry:
    """Lightweight cross-frame record used only for IoU-based throttle matching."""

    track_id: str
    bbox: BoundingBox


def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x_min, b.x_min)
    iy1 = max(a.y_min, b.y_min)
    ix2 = min(a.x_max, b.x_max)
    iy2 = min(a.y_max, b.y_max)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, (a.x_max - a.x_min) * (a.y_max - a.y_min))
    area_b = max(0, (b.x_max - b.x_min) * (b.y_max - b.y_min))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_to_tracks(
    detections: list[Detection],
    prev_tracks: list[_TrackEntry],
) -> list[str]:
    """Assign a stable throttle key to each detection via greedy IoU matching.

    Detections that overlap a previous-frame track at IoU >= _IOU_TRACK_THRESHOLD
    inherit that track's ID so the cooldown fires correctly.  Unmatched
    detections (new people, re-entries) get a fresh UUID so face-id runs
    immediately.

    Greedy assignment: each previous track is claimed at most once, preventing
    two detections from sharing a key after a brief occlusion.
    """
    used: set[str] = set()
    result: list[str] = []
    for det in detections:
        best_iou = _IOU_TRACK_THRESHOLD  # strict lower bound
        best_tid: str | None = None
        for track in prev_tracks:
            if track.track_id in used:
                continue
            iou = _bbox_iou(det.bbox, track.bbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = track.track_id
        if best_tid is not None:
            used.add(best_tid)
            result.append(best_tid)
        else:
            result.append(str(uuid.uuid4()))
    return result


def _crossing_indices(detections: list[Detection]) -> frozenset[int]:
    """Return indices of detections whose bboxes mutually overlap > threshold.

    When two people cross, their bounding boxes partially overlap.  Forcing
    face-id for both ensures their identities stay consistent through the
    transition — one cannot be silently throttled under the wrong label.
    """
    flagged: set[int] = set()
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if _bbox_iou(detections[i].bbox, detections[j].bbox) > _IOU_CROSSING_THRESHOLD:
                flagged.add(i)
                flagged.add(j)
    return frozenset(flagged)


class FaceIdentityStage(FrameStage):
    name = "face_identity"

    def __init__(
        self,
        face_stats_repo: FaceObservationStatsRepository,
        face_id_client: FaceIdentificationClient | None = None,
        gallery_repo: GalleryRepository | None = None,
        face_id_cooldown_s: float = 5.0,
        face_id_min_raw_similarity: float = 0.5,
        face_id_camera_configs: dict[str, FaceIdCameraConfig] | None = None,
        last_face_id_by_tracklet: dict[str, datetime] | None = None,
        face_service_version: str = "",
        expected_arcface_model_version: str = "",
        expected_preprocessing_version: str = "",
        expected_quality_version: str = "heuristic-v1",
        head_region_policy: HeadRegionPolicy | None = None,
        crop_padding: FaceCropPadding | None = None,
    ) -> None:
        self._face_id_client = face_id_client
        # Production values come from settings.yaml face_id.* via app/main.py; the defaults
        # exist for tests that do not exercise attribution or padding.
        self._head_region_policy = head_region_policy or HeadRegionPolicy()
        self._crop_padding = crop_padding or FaceCropPadding()
        self._face_stats_repo = face_stats_repo
        self._gallery_repo = gallery_repo
        self._face_id_cooldown_s = face_id_cooldown_s
        self._face_id_min_raw_similarity = face_id_min_raw_similarity
        self._face_id_camera_configs = face_id_camera_configs or {}
        self._face_service_version = face_service_version
        self._expected_arcface_model_version = expected_arcface_model_version
        self._expected_preprocessing_version = expected_preprocessing_version
        self._expected_quality_version = expected_quality_version
        # Cooldown dict keyed by stable track ID (not ephemeral detection_id).
        # Shared with the pipeline so the pruning interval in frame_pipeline.py
        # can also clear stale entries.
        self._last_face_id_by_tracklet: dict[str, datetime] = (
            last_face_id_by_tracklet if last_face_id_by_tracklet is not None else {}
        )
        # Per-camera previous-frame track state used for IoU matching.
        # Rebuilt every frame; only the most recent frame's bbox + track_id
        # are kept so stale entries cannot persist across camera restarts.
        self._camera_tracks: dict[str, list[_TrackEntry]] = {}

    def _get_face_id_config(self, camera_id: str) -> FaceIdCameraConfig:
        return self._face_id_camera_configs.get(camera_id, FaceIdCameraConfig())

    def _resolve_calibrated_confidence(self, face: FaceResult) -> float | None:
        """Return calibrated_confidence only when the service reports ready and versions match.

        Any degraded status, version mismatch, a calibration profile other than single-frame, or a
        face-quality scorer other than the one the artifact was fitted on yields None, so the
        authority gate fails closed.
        """
        if face.calibration_status != "ready":
            return None
        if face.calibration_profile != SINGLE_FRAME_PROFILE:
            return None
        if face.quality is None or face.quality.version != self._expected_quality_version:
            return None
        if self._expected_arcface_model_version and (
            face.arcface_model_version != self._expected_arcface_model_version
        ):
            return None
        if self._expected_preprocessing_version and (
            face.preprocessing_version != self._expected_preprocessing_version
        ):
            return None
        return face.calibrated_confidence

    async def run(self, ctx: FrameContext) -> None:
        camera_id = ctx.frame.camera_id
        now = datetime.now(UTC)

        cam_cfg = self._get_face_id_config(camera_id)
        if not cam_cfg.enabled:
            return
        if self._face_id_client is None:
            return

        prev_tracks = self._camera_tracks.get(camera_id, [])

        if not ctx.domain_detections or not ctx.crops:
            # No detections this frame: prune cooldown entries for all previous
            # tracks and clear camera state so the next appearance is treated fresh.
            for old_track in prev_tracks:
                self._last_face_id_by_tracklet.pop(old_track.track_id, None)
            self._camera_tracks.pop(camera_id, None)
            return

        # ------------------------------------------------------------------
        # 1. Build stable per-detection throttle keys via IoU matching.
        #    Detections that overlap a previous-frame track at >= 0.8 IoU
        #    inherit that track's ID; new persons get a fresh UUID.
        # ------------------------------------------------------------------
        track_ids = _match_to_tracks(ctx.domain_detections, prev_tracks)

        # ------------------------------------------------------------------
        # 2. Detect potential crossings — force face-id for both detections.
        # ------------------------------------------------------------------
        crossing = _crossing_indices(ctx.domain_detections)
        if crossing:
            logger.debug(
                "face_id_crossing_detected",
                camera_id=camera_id,
                crossing_count=len(crossing),
            )

        # ------------------------------------------------------------------
        # 3. Apply per-track cooldown; collect eligible detections.
        # ------------------------------------------------------------------
        eligible_crops: list[npt.NDArray[np.uint8]] = []
        eligible_detections: list[Detection] = []
        eligible_face_boxes: list[BoundingBox] = []
        sent_ids: set[str] = set()
        image = ctx.require_image()
        image_h, image_w = image.shape[:2]

        for idx, det in enumerate(ctx.domain_detections):
            track_id = track_ids[idx]

            if idx not in crossing:
                last_call = self._last_face_id_by_tracklet.get(track_id)
                if last_call is not None:
                    elapsed = (now - last_call).total_seconds()
                    if elapsed < self._face_id_cooldown_s:
                        _metrics.metrics.face_id_cooldown_skips_total.inc()
                        logger.debug(
                            "face_id_cooldown_skip",
                            track_id=track_id,
                            elapsed_s=round(elapsed, 1),
                            cooldown_s=self._face_id_cooldown_s,
                        )
                        continue

            # The face path crops its own padded box from the privacy-filtered frame; ReID and
            # pose keep the tight crops in ctx.crops.
            face_box = face_crop_box(det.bbox, image_w, image_h, self._crop_padding)
            eligible_crops.append(crop_detection_from_bbox(image, face_box))
            eligible_detections.append(det)
            eligible_face_boxes.append(face_box)
            sent_ids.add(track_id)

        # ------------------------------------------------------------------
        # 4. Update camera track state for next frame and prune disappeared
        #    tracks from the cooldown dict (prevents unbounded growth).
        # ------------------------------------------------------------------
        active_track_ids = set(track_ids)
        for old_track in prev_tracks:
            if old_track.track_id not in active_track_ids:
                self._last_face_id_by_tracklet.pop(old_track.track_id, None)

        self._camera_tracks[camera_id] = [
            _TrackEntry(track_id=tid, bbox=det.bbox)
            for tid, det in zip(track_ids, ctx.domain_detections, strict=True)
        ]

        if not eligible_crops:
            return

        ctx.face_anchors = await self._identify_faces_from_crops(
            crops=eligible_crops,
            crop_detections=eligible_detections,
            face_boxes=eligible_face_boxes,
            frame_width=ctx.effective_width,
            frame_height=ctx.effective_height,
            camera_id=camera_id,
            captured_at=ctx.capture_time,
            sent_ids=sent_ids,
        )
        if ctx.face_anchors:
            self._build_face_evidence(ctx)

        if ctx.face_anchors and self._gallery_repo is not None:
            seen: set[str] = set()
            for fa in ctx.face_anchors:
                if fa.person_id and fa.person_id != "unknown" and fa.person_id not in seen:
                    seen.add(fa.person_id)
                    await self._gallery_repo.upsert_identity(
                        Identity(
                            identity_id=fa.person_id,
                            display_name=fa.person_id,
                            enrolled_at=now,
                        )
                    )

    async def _identify_faces_from_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_detections: list[Detection],
        face_boxes: list[BoundingBox],
        frame_width: int,
        frame_height: int,
        camera_id: str,
        captured_at: datetime,
        sent_ids: set[str] | None = None,
    ) -> list[FaceAnchor]:
        if self._face_id_client is None or not crops:
            return []

        cam_cfg = self._get_face_id_config(camera_id)
        if not cam_cfg.enabled:
            return []

        # Normalized coordinates of the padded crops actually sent, so face boxes map back to
        # frame space correctly.
        crop_bboxes_norm = [
            (
                box.x_min / frame_width,
                box.y_min / frame_height,
                box.x_max / frame_width,
                box.y_max / frame_height,
            )
            for box in face_boxes
        ]

        now = datetime.now(UTC)
        try:
            crop_face_results = await self._face_id_client.identify_crops(
                crops,
                crop_bboxes_norm,
                CaptureContext(
                    camera_id=camera_id,
                    captured_at=captured_at,
                    crop_pad_top_frac=self._crop_padding.top_frac,
                    crop_pad_side_frac=self._crop_padding.side_frac,
                ),
            )
        except Exception:  # noqa: BLE001
            if sent_ids:
                for key in sent_ids:
                    self._last_face_id_by_tracklet[key] = now
            logger.warning(
                "face_id_service_error",
                camera_id=camera_id,
                crop_count=len(crops),
            )
            return []

        if sent_ids:
            for key in sent_ids:
                self._last_face_id_by_tracklet[key] = now

        for crop_result in crop_face_results:
            _metrics.metrics.cts_face_queries_total.labels(
                camera_id=camera_id, result=crop_result.status.value
            ).inc()

        min_raw_similarity = (
            cam_cfg.min_raw_similarity
            if cam_cfg.min_raw_similarity is not None
            else self._face_id_min_raw_similarity
        )

        face_anchors: list[FaceAnchor] = []
        stats: list[FaceObservationStat] = []
        recognized_count = 0
        candidate_count = 0
        unrecognized_count = 0

        for crop_result in crop_face_results:
            det = crop_detections[crop_result.crop_index]
            detection_id = det.detection_id

            if not detection_id:
                logger.debug(
                    "face_anchor_dropped_no_detection_id",
                    camera_id=camera_id,
                )
                stats.extend(
                    _face_stat(
                        face,
                        crop_result,
                        camera_id,
                        "",
                        captured_at,
                        FaceGateOutcome.DROPPED_NO_DETECTION,
                        self._resolve_calibrated_confidence(face),
                    )
                    for face in crop_result.faces
                    if face.recognition_state in _RECOGNITION_STATES
                )
                continue

            attributions = self._attribute_faces(
                crop_result,
                det.bbox,
                face_boxes[crop_result.crop_index],
                frame_width,
                frame_height,
                camera_id,
            )
            for face, attribution in zip(crop_result.faces, attributions, strict=True):
                state = face.recognition_state
                outcome = _gate_outcome(face, min_raw_similarity, attribution)
                if outcome is None:
                    continue
                stats.append(
                    _face_stat(
                        face,
                        crop_result,
                        camera_id,
                        detection_id,
                        captured_at,
                        outcome,
                        self._resolve_calibrated_confidence(face),
                    )
                )
                if outcome is not FaceGateOutcome.EMITTED:
                    continue

                # Recorded as measured; whether it weighs the evidence is the resolver's
                # face-quality mode, never decided here.
                face_quality = face.quality.score if face.quality is not None else None
                face_quality_version = face.quality.version if face.quality is not None else ""

                if state == "recognized":
                    face_anchors.append(
                        FaceAnchor(
                            person_id=face.person_id,
                            match_score=face.raw_similarity,
                            tracklet_id="",
                            detection_id=detection_id,
                            camera_id=camera_id,
                            captured_at=now,
                            recognition_state="recognized",
                            similarity=face.raw_similarity,
                            yaw_deg=face.yaw_deg,
                            det_score=face.det_score,
                            face_quality=face_quality,
                            face_quality_version=face_quality_version,
                            calibrated_confidence=self._resolve_calibrated_confidence(face),
                        )
                    )
                    recognized_count += 1

                elif state == "candidate":
                    # Weak positive for the best candidate (grey zone).
                    # The raw-similarity floor does not apply; the similarity is intentionally below
                    # the recognition threshold but at or above unknown_threshold.
                    assert face.best_candidate_id is not None  # guaranteed by _gate_outcome
                    face_anchors.append(
                        FaceAnchor(
                            person_id=face.best_candidate_id,
                            match_score=face.raw_similarity,
                            tracklet_id="",
                            detection_id=detection_id,
                            camera_id=camera_id,
                            captured_at=now,
                            recognition_state="candidate",
                            similarity=face.raw_similarity,
                            yaw_deg=face.yaw_deg,
                            det_score=face.det_score,
                            face_quality=face_quality,
                            face_quality_version=face_quality_version,
                            # candidates never carry calibrated authority
                            calibrated_confidence=None,
                        )
                    )
                    candidate_count += 1

                elif state == "unrecognized":
                    # Marker: a face region was detected but the embedding is too far
                    # from any enrolled identity.  Carries det_score so the resolver
                    # knows a real face existed.
                    face_anchors.append(
                        FaceAnchor(
                            person_id="unknown",
                            match_score=face.det_score,  # detection score, not recognition
                            tracklet_id="",
                            detection_id=detection_id,
                            camera_id=camera_id,
                            captured_at=now,
                            recognition_state="unrecognized",
                            similarity=face.raw_similarity,
                            yaw_deg=face.yaw_deg,
                            det_score=face.det_score,
                            face_quality=face_quality,
                            face_quality_version=face_quality_version,
                            calibrated_confidence=None,
                        )
                    )
                    unrecognized_count += 1

        await self._record_face_stats(camera_id, stats)

        if face_anchors:
            logger.debug(
                "face_anchors_created",
                camera_id=camera_id,
                anchor_count=len(face_anchors),
                recognized=recognized_count,
                candidate=candidate_count,
                unrecognized=unrecognized_count,
                identities=[fa.person_id for fa in face_anchors],
                mode="ph",
            )
        # Emit per-state metric so operators can see the recognition-state mix.
        if recognized_count:
            _metrics.metrics.cts_face_anchors_total.labels(recognition_state="recognized").inc(
                recognized_count
            )
        if candidate_count:
            _metrics.metrics.cts_face_anchors_total.labels(recognition_state="candidate").inc(
                candidate_count
            )
        if unrecognized_count:
            _metrics.metrics.cts_face_anchors_total.labels(recognition_state="unrecognized").inc(
                unrecognized_count
            )
        return face_anchors

    def _attribute_faces(
        self,
        crop_result: CropQueryResult,
        person_box_px: BoundingBox,
        face_box_px: BoundingBox,
        frame_width: int,
        frame_height: int,
        camera_id: str,
    ) -> list[FaceAttribution]:
        """Per-face attribution for one crop: ASSIGNED for at most one face, else why not.

        A face outside the head region is another person's; every face of an ambiguous crop is
        ambiguous.
        """
        if not crop_result.faces:
            return []
        face_boxes_px = [
            (
                face.bbox_normalized[0] * frame_width,
                face.bbox_normalized[1] * frame_height,
                face.bbox_normalized[2] * frame_width,
                face.bbox_normalized[3] * frame_height,
            )
            for face in crop_result.faces
        ]
        result, assigned_index = assign_face_to_person(
            person_box_px,
            face_boxes_px,
            self._head_region_policy,
            pad_top_px=person_box_px.y_min - face_box_px.y_min,
        )
        _metrics.metrics.cts_face_attribution_total.labels(
            camera_id=camera_id, result=result.value
        ).inc()
        if result is FaceAttribution.AMBIGUOUS:
            return [FaceAttribution.AMBIGUOUS] * len(crop_result.faces)
        return [
            FaceAttribution.ASSIGNED
            if index == assigned_index
            else FaceAttribution.NO_FACE_IN_HEAD_REGION
            for index in range(len(crop_result.faces))
        ]

    async def _record_face_stats(self, camera_id: str, stats: list[FaceObservationStat]) -> None:
        """Persist measurement rows and count outcomes; a failure never affects face evidence."""
        for stat in stats:
            _metrics.metrics.cts_face_gate_outcomes_total.labels(
                camera_id=camera_id, outcome=stat.gate_outcome.value
            ).inc()
        if not stats:
            return
        try:
            await self._face_stats_repo.append_many(stats)
        except Exception:  # noqa: BLE001
            _metrics.metrics.cts_face_stats_write_failures_total.inc()
            logger.warning("face_stats_append_failed", camera_id=camera_id, rows=len(stats))

    def _build_face_evidence(self, ctx: FrameContext) -> None:
        """Populate ``ctx._face_evidence`` from ``ctx.face_anchors``.

        FaceEvidence carries ``source="direct"`` to distinguish real ArcFace
        matches from synthetic propagated anchors in the identity resolver.
        In PH mode, detection_id is the primary key for matching evidence
        to observations.

        ``recognition_state``, ``similarity``, and ``yaw_deg`` are forwarded
        so the resolver can weight evidence by recognition state and frontality.
        """
        evidence: list[FaceEvidence] = []
        for fa in ctx.face_anchors:
            evidence.append(
                FaceEvidence(
                    person_id=fa.person_id,
                    match_score=fa.match_score,
                    tracklet_id=fa.tracklet_id,
                    detection_id=fa.detection_id,
                    camera_id=fa.camera_id,
                    frame_index=ctx.frame.frame_index,
                    source="direct",
                    quality=fa.quality,
                    model_version=self._face_service_version,
                    captured_at=fa.captured_at,
                    recognition_state=fa.recognition_state,
                    similarity=fa.similarity,
                    yaw_deg=fa.yaw_deg,
                    calibrated_confidence=fa.calibrated_confidence,
                )
            )
        ctx._face_evidence = evidence


_RECOGNITION_STATES = frozenset({"recognized", "candidate", "unrecognized"})


def _gate_outcome(
    face: FaceResult, min_raw_similarity: float, attribution: FaceAttribution
) -> FaceGateOutcome | None:
    """Decide what the stage does with one face; None for a state the service never defines.

    Only the one face attributed to the person can become evidence: a face outside the person's
    head region belongs to someone else, and every face of an ambiguous crop is dropped.
    Recognized faces below the raw-similarity floor are dropped; candidates without a nearest
    enrolled identity cannot name anyone and are dropped; everything else becomes an anchor.
    """
    if face.recognition_state not in _RECOGNITION_STATES:
        return None
    if attribution is FaceAttribution.AMBIGUOUS:
        return FaceGateOutcome.DROPPED_AMBIGUOUS
    if attribution is FaceAttribution.NO_FACE_IN_HEAD_REGION:
        return FaceGateOutcome.DROPPED_NOT_IN_HEAD_REGION
    if face.recognition_state == "recognized" and face.raw_similarity < min_raw_similarity:
        return FaceGateOutcome.DROPPED_MIN_SIMILARITY
    if face.recognition_state == "candidate" and face.best_candidate_id is None:
        return FaceGateOutcome.DROPPED_NO_CANDIDATE
    return FaceGateOutcome.EMITTED


def _face_stat(
    face: FaceResult,
    crop_result: CropQueryResult,
    camera_id: str,
    detection_id: str,
    captured_at: datetime,
    outcome: FaceGateOutcome,
    calibrated_confidence: float | None,
) -> FaceObservationStat:
    return FaceObservationStat(
        observed_at=captured_at,
        camera_id=camera_id,
        detection_id=detection_id,
        crop_width_px=crop_result.crop_width_px,
        crop_height_px=crop_result.crop_height_px,
        face_width_px=face.face_width_px,
        face_height_px=face.face_height_px,
        interocular_px=face.interocular_px,
        det_score=face.det_score,
        yaw_deg=face.yaw_deg,
        pitch_deg=face.pitch_deg,
        raw_similarity=face.raw_similarity,
        recognition_state=face.recognition_state,
        best_candidate_id=face.best_candidate_id,
        gate_outcome=outcome,
        quality_score=face.quality.score if face.quality is not None else None,
        quality_version=face.quality.version if face.quality is not None else None,
        calibrated_confidence=calibrated_confidence,
    )
