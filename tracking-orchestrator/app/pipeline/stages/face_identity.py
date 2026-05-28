"""Face identity stage: identifies faces via person-identification-service."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ...domain import Detection, FaceAnchor, Identity
from ...inference.evidence import FaceEvidence
from ...inference.face_id_client import FaceIdentificationClient
from ...observability import metrics as _metrics
from ...storage.base import GalleryRepository
from ..frame_context import FrameContext
from ..types import FaceIdCameraConfig
from .base import FrameStage

logger = get_logger(__name__)


class FaceIdentityStage(FrameStage):
    name = "face_identity"

    def __init__(
        self,
        face_id_client: FaceIdentificationClient | None = None,
        tracklet_manager: object | None = None,  # N0: was TrackletManager, deleted
        gallery_repo: GalleryRepository | None = None,
        face_id_cooldown_s: float = 5.0,
        face_id_min_confidence: float = 0.5,
        face_id_camera_configs: dict[str, FaceIdCameraConfig] | None = None,
        last_face_id_by_tracklet: dict[str, datetime] | None = None,
        face_service_version: str = "",
    ) -> None:
        self._face_id_client = face_id_client
        self._tracklet_manager = tracklet_manager
        self._gallery_repo = gallery_repo
        self._face_id_cooldown_s = face_id_cooldown_s
        self._face_id_min_confidence = face_id_min_confidence
        self._face_id_camera_configs = face_id_camera_configs or {}
        self._face_service_version = face_service_version
        # Shared mutable state (owned by pipeline, read/written here and pruned
        # by GlobalTrackingStage).
        self._last_face_id_by_tracklet: dict[str, datetime] = (
            last_face_id_by_tracklet if last_face_id_by_tracklet is not None else {}
        )

    def _get_face_id_config(self, camera_id: str) -> FaceIdCameraConfig:
        return self._face_id_camera_configs.get(camera_id, FaceIdCameraConfig())

    async def run(self, ctx: FrameContext) -> None:
        if not ctx.domain_detections or not ctx.crops:
            return
        camera_id = ctx.frame.camera_id
        now = datetime.now(UTC)

        cam_cfg = self._get_face_id_config(camera_id)
        if not cam_cfg.enabled:
            return
        if self._face_id_client is None:
            return

        eligible_indices: list[int] = []
        eligible_crops: list[npt.NDArray[np.uint8]] = []
        eligible_detections = []
        sent_ids: set[str] = set()

        for idx, det in enumerate(ctx.domain_detections):
            # Key cooldown by detection_id (PH mode) or tracklet_id (legacy).
            if self._tracklet_manager is not None:
                key = self._tracklet_manager.get_tracklet_id_for_detection(det.detection_id)  # type: ignore[attr-defined]
                if not key:
                    continue
            else:
                key = det.detection_id
                if not key:
                    continue

            last_call = self._last_face_id_by_tracklet.get(key)
            if last_call is not None:
                elapsed = (now - last_call).total_seconds()
                if elapsed < self._face_id_cooldown_s:
                    _metrics.metrics.face_id_cooldown_skips_total.inc()
                    logger.debug(
                        "face_id_cooldown_skip",
                        key=key,
                        elapsed_s=round(elapsed, 1),
                        cooldown_s=self._face_id_cooldown_s,
                    )
                    continue
            eligible_indices.append(idx)
            eligible_crops.append(ctx.crops[idx])
            eligible_detections.append(det)
            sent_ids.add(key)

        if not eligible_crops:
            return

        ctx.face_anchors = await self._identify_faces_from_crops(
            crops=eligible_crops,
            crop_detections=eligible_detections,
            frame_width=ctx.effective_width,
            frame_height=ctx.effective_height,
            camera_id=camera_id,
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
        frame_width: int,
        frame_height: int,
        camera_id: str,
        sent_ids: set[str] | None = None,
    ) -> list[FaceAnchor]:
        if self._face_id_client is None or not crops:
            return []

        cam_cfg = self._get_face_id_config(camera_id)
        if not cam_cfg.enabled:
            return []

        crop_bboxes_norm: list[tuple[float, float, float, float]] = []
        for det in crop_detections:
            crop_bboxes_norm.append(
                (
                    det.bbox.x_min / frame_width,
                    det.bbox.y_min / frame_height,
                    det.bbox.x_max / frame_width,
                    det.bbox.y_max / frame_height,
                )
            )

        now = datetime.now(UTC)
        try:
            crop_face_results = await self._face_id_client.identify_crops(crops, crop_bboxes_norm)
        except Exception:
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

        if not crop_face_results:
            return []

        min_conf = (
            cam_cfg.min_confidence
            if cam_cfg.min_confidence is not None
            else self._face_id_min_confidence
        )

        face_anchors: list[FaceAnchor] = []
        for crop_idx, face_results in crop_face_results:
            det = crop_detections[crop_idx]

            for face in face_results:
                if face.person_id == "unknown":
                    continue
                if face.confidence < min_conf:
                    continue

                # Build the detection key for tracker anchoring.
                tracklet_id = ""
                detection_id = det.detection_id
                if self._tracklet_manager is not None:
                    tracklet_id = self._tracklet_manager.get_tracklet_id_for_detection(  # type: ignore[attr-defined]
                        det.detection_id
                    )
                    if not tracklet_id:
                        logger.debug(
                            "face_anchor_dropped_no_tracklet",
                            person_id=face.person_id,
                            detection_id=det.detection_id,
                            camera_id=camera_id,
                        )
                        continue

                # PH mode: anchor by detection_id when no tracklet_manager.
                if self._tracklet_manager is None and not detection_id:
                    logger.debug(
                        "face_anchor_dropped_no_detection_id",
                        person_id=face.person_id,
                        camera_id=camera_id,
                    )
                    continue

                face_anchors.append(
                    FaceAnchor(
                        person_id=face.person_id,
                        confidence=face.confidence,
                        tracklet_id=tracklet_id,
                        detection_id=detection_id,
                        camera_id=camera_id,
                        captured_at=now,
                    )
                )

        if face_anchors:
            logger.debug(
                "face_anchors_created",
                camera_id=camera_id,
                anchor_count=len(face_anchors),
                identities=[fa.person_id for fa in face_anchors],
                mode="ph" if self._tracklet_manager is None else "tracklet",
            )
        return face_anchors

    def _build_face_evidence(self, ctx: FrameContext) -> None:
        """Populate ``ctx._face_evidence`` from ``ctx.face_anchors``.

        FaceEvidence carries ``source="direct"`` to distinguish real ArcFace
        matches from synthetic propagated anchors in the identity resolver.
        In PH mode, detection_id is the primary key for matching evidence
        to observations.
        """
        evidence: list[FaceEvidence] = []
        for fa in ctx.face_anchors:
            evidence.append(
                FaceEvidence(
                    person_id=fa.person_id,
                    confidence=fa.confidence,
                    tracklet_id=fa.tracklet_id,
                    detection_id=fa.detection_id,
                    camera_id=fa.camera_id,
                    frame_index=ctx.frame.frame_index,
                    source="direct",
                    quality=fa.quality,
                    model_version=self._face_service_version,
                    captured_at=fa.captured_at,
                )
            )
        ctx._face_evidence = evidence
