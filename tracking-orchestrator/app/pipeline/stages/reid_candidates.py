"""ReID candidate creation stage (M04): governed candidate creation.

The only pipeline write path into ``reid_gallery``. Runs after
``ProvenancePersistStage`` and before ``PublishStage`` on all three execution
routes -- creation is durability-adjacent (audit-visible governance data,
same argument as provenance persistence) so it must not sit behind
``PublishStage``'s per-camera publish throttle.

See ``app/tracking/identity/candidate_eligibility.py`` for the pure gate this
stage evaluates, and the cts-identity-governance skill for the authority
rules the gate encodes.
"""

from __future__ import annotations

import hashlib
import uuid

import cv2
import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ...domain import FaceAnchor, NewReviewCandidate, OrientationBin
from ...observability import metrics as _metrics
from ...storage.gallery import GalleryRepository
from ...tracking.identity.candidate_eligibility import CandidatePolicy, evaluate_candidate
from ..frame_context import FrameContext
from ..types import CropStorageProtocol
from .base import FrameStage

logger = get_logger(__name__)

_JPEG_QUALITY = 85


def _encode_jpeg(crop: npt.NDArray[np.uint8]) -> bytes:
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed for ReID candidate crop")
    return buf.tobytes()


class ReIDCandidateStage(FrameStage):
    """Creates governed ``pending_review`` candidates from committed, face-matched detections."""

    name = "reid_candidates"

    def __init__(
        self,
        gallery_repo: GalleryRepository | None = None,
        crop_storage: CropStorageProtocol | None = None,
        policy: CandidatePolicy = CandidatePolicy(),
    ) -> None:
        self._gallery_repo = gallery_repo
        self._crop_storage = crop_storage
        self._policy = policy
        self._warned_no_storage = False

    async def run(self, ctx: FrameContext) -> None:
        if not self._policy.enabled or not ctx.det_to_ph:
            return
        if self._gallery_repo is None:
            return
        if self._crop_storage is None:
            if not self._warned_no_storage:
                self._warned_no_storage = True
                logger.warning("reid_candidate_storage_unconfigured")
            return

        face_by_det: dict[str, FaceAnchor] = {
            fa.detection_id: fa for fa in ctx.face_anchors if fa.detection_id
        }
        det_idx_by_id = {det.detection_id: i for i, det in enumerate(ctx.domain_detections)}

        for det_id, ph_id in ctx.det_to_ph.items():
            await self._maybe_create_candidate(ctx, det_id, ph_id, face_by_det, det_idx_by_id)

    async def _maybe_create_candidate(
        self,
        ctx: FrameContext,
        det_id: str,
        ph_id: str,
        face_by_det: dict[str, FaceAnchor],
        det_idx_by_id: dict[str, int],
    ) -> None:
        assert self._gallery_repo is not None  # checked by caller (run())
        det_idx = det_idx_by_id.get(det_id)
        if det_idx is None:
            return
        det = ctx.domain_detections[det_idx]
        face_anchor = face_by_det.get(det_id)
        committed_identity_id = ctx.committed_ids.get(ph_id)
        orientation, orientation_confidence = ctx.orientation_by_detection.get(
            det_id, (OrientationBin.UNKNOWN, 0.0)
        )

        eligibility = evaluate_candidate(
            committed_identity_id=committed_identity_id,
            face_anchor=face_anchor,
            embedding=det.embedding,
            quality=det.crop_quality,
            orientation=orientation,
            orientation_confidence=orientation_confidence,
            cfg=self._policy,
        )
        if not eligibility.eligible:
            _metrics.metrics.reid_candidate_rejected_total.labels(reason=eligibility.reason).inc()
            return

        # evaluate_candidate's no_identity/no_face_anchor gates guarantee both below.
        assert committed_identity_id is not None
        assert face_anchor is not None

        existing = await self._gallery_repo.count_gallery_entries(
            identity_id=committed_identity_id,
            orientation=int(orientation),
        )
        if existing >= self._policy.max_per_identity_orientation:
            _metrics.metrics.reid_candidate_rejected_total.labels(reason="cap_reached").inc()
            return

        if det_idx >= len(ctx.crops):
            return
        crop = ctx.crops[det_idx]

        candidate_id = str(uuid.uuid4())
        crop_bytes = _encode_jpeg(crop)
        crop_hash = hashlib.sha256(crop_bytes).hexdigest()
        crop_key = f"reid-candidates/{self._policy.model_version}/{candidate_id}.jpg"

        assert self._crop_storage is not None  # checked by caller
        try:
            await self._crop_storage.put_bytes(crop_key, crop_bytes)
        except Exception:  # noqa: BLE001 -- best-effort auxiliary governance data,
            # a storage hiccup on one detection must never fail the tracking frame.
            logger.warning(
                "reid_candidate_crop_upload_failed",
                candidate_id=candidate_id,
                crop_key=crop_key,
                exc_info=True,
            )
            return

        ew, eh = ctx.effective_width, ctx.effective_height
        edge_truncated = (
            det.bbox.x_min <= 0
            or det.bbox.y_min <= 0
            or det.bbox.x_max >= ew
            or det.bbox.y_max >= eh
        )
        observation_id = ctx.det_to_observation_id.get(det_id, "")
        confidence = (
            face_anchor.calibrated_confidence
            if face_anchor.calibrated_confidence is not None
            else face_anchor.confidence
        )

        candidate = NewReviewCandidate(
            candidate_id=candidate_id,
            identity_id=committed_identity_id,
            embedding=list(det.embedding),
            quality=det.crop_quality,
            orientation=int(orientation),
            camera_id=det.camera_id,
            capture_time=det.capture_time,
            ph_id=ph_id,
            observation_id=observation_id,
            # origin_tracklet_id must be the repository-assigned observation id
            # (not detection_id): the resolver's list_gallery_entries_for_tracklets
            # query matches against PersonHypothesis.observation_ids, which are
            # these same ids (see WorldTrackerResult.det_to_observation_id).
            origin_tracklet_id=observation_id,
            keyframe_id=None,
            crop_key=crop_key,
            source_frame_key=None,
            crop_hash=crop_hash,
            frame_hash=None,
            dimensions=(crop.shape[1], crop.shape[0]),
            is_truncated=edge_truncated,
            is_occluded=False,  # no occlusion detector exists yet (TD-008)
            candidate_reason="face_derived",
            # Episode = one PH's continuous track: gives the resolver's
            # per-episode vote cap a real group key (see cts-identity-governance).
            source_episode_id=ph_id,
            created_actor="pipeline",
            model_version=self._policy.model_version,
            preprocessing_version=self._policy.preprocessing_version,
            confidence=confidence,
            state=eligibility.mint_state,
        )

        try:
            await self._gallery_repo.create_review_candidate(candidate)
        except Exception:  # noqa: BLE001 -- see justification above; clean up the
            # orphaned crop object on DB failure so a retry with the same
            # candidate_id (idempotent on create_review_candidate) does not leak.
            logger.warning(
                "reid_candidate_create_failed",
                candidate_id=candidate_id,
                exc_info=True,
            )
            try:
                await self._crop_storage.delete_object(crop_key)
            except Exception:  # noqa: BLE001 -- reconciliation job cleans up orphans
                logger.warning(
                    "reid_candidate_orphan_crop_delete_failed",
                    crop_key=crop_key,
                    exc_info=True,
                )
            return

        _metrics.metrics.reid_candidate_created_total.labels(state=eligibility.mint_state).inc()
        logger.debug(
            "reid_candidate_created",
            candidate_id=candidate_id,
            identity_id=committed_identity_id,
            ph_id=ph_id,
            orientation=int(orientation),
            state=eligibility.mint_state,
        )
