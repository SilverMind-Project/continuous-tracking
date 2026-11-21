"""ReID candidate creation stage: governed candidate creation.

The only pipeline write path into ``reid_gallery``. Runs after
``ProvenancePersistStage`` and before ``PublishStage`` on all three execution
routes -- creation is durability-adjacent (audit-visible governance data,
same argument as provenance persistence) so it must not sit behind
``PublishStage``'s per-camera publish throttle.

A detection seeds a row from a recognized face first; when no face seed qualifies and its PH
carries attested wearable evidence this round, it may seed from the wearable instead
(``reid_candidates.wearable_seeding_enabled``). See
``app/tracking/identity/candidate_eligibility.py`` for the pure gates and the
cts-identity-governance skill for the authority rules they encode.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from datetime import datetime, time, timedelta

import cv2
import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ...domain import (
    Detection,
    FaceAnchor,
    GallerySeedSource,
    NewReviewCandidate,
    OrientationBin,
    WearableEvidence,
)
from ...observability import metrics as _metrics
from ...observability.metrics import WearableSeedOutcome
from ...storage.gallery import PENDING_AND_VERIFIED, GalleryRepository
from ...storage.wearable_teaching_holds import WearableTeachingHoldRepository
from ...tracking.identity.candidate_eligibility import (
    CandidateEligibility,
    CandidatePolicy,
    FaceSeed,
    WearableSeed,
    evaluate_candidate,
)
from ..frame_context import FrameContext
from ..types import CropStorageProtocol
from .base import FrameStage

logger = get_logger(__name__)

_JPEG_QUALITY = 85
_WEARABLE_SEED_SOURCES = frozenset(
    {GallerySeedSource.WEARABLE_ATTESTED, GallerySeedSource.WEARABLE_ROOM_ATTESTED}
)


def _encode_jpeg(crop: npt.NDArray[np.uint8]) -> bytes:
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed for ReID candidate crop")
    return buf.tobytes()


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def _utc_day_start(at: datetime) -> datetime:
    return datetime.combine(at.date(), time.min, tzinfo=at.tzinfo)


class ReIDCandidateStage(FrameStage):
    """Creates governed gallery candidates from committed, face- or wearable-seeded detections."""

    name = "reid_candidates"

    def __init__(
        self,
        gallery_repo: GalleryRepository | None = None,
        crop_storage: CropStorageProtocol | None = None,
        policy: CandidatePolicy = CandidatePolicy(),
        teaching_holds: WearableTeachingHoldRepository | None = None,
    ) -> None:
        self._gallery_repo = gallery_repo
        self._crop_storage = crop_storage
        self._policy = policy
        self._teaching_holds = teaching_holds
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
        faces_by_det: dict[str, list[FaceAnchor]] = {}
        for fa in (*ctx.face_anchors, *ctx.shadow_face_anchors):
            if fa.detection_id:
                faces_by_det.setdefault(fa.detection_id, []).append(fa)
        det_idx_by_id = {det.detection_id: i for i, det in enumerate(ctx.domain_detections)}

        for det_id, ph_id in ctx.det_to_ph.items():
            await self._maybe_create_candidate(
                ctx, det_id, ph_id, face_by_det, faces_by_det, det_idx_by_id
            )

    async def _maybe_create_candidate(
        self,
        ctx: FrameContext,
        det_id: str,
        ph_id: str,
        face_by_det: dict[str, FaceAnchor],
        faces_by_det: dict[str, list[FaceAnchor]],
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
            seed=FaceSeed(face_anchor) if face_anchor is not None else None,
            embedding=det.embedding,
            quality=det.crop_quality,
            orientation=orientation,
            orientation_confidence=orientation_confidence,
            cfg=self._policy,
        )
        if eligibility.eligible:
            # evaluate_candidate's no_identity/no_face_anchor gates guarantee both below.
            assert committed_identity_id is not None
            assert face_anchor is not None
            if await self._at_orientation_cap(
                identity_id=committed_identity_id,
                orientation=orientation,
                capture_time=det.capture_time,
            ):
                _metrics.metrics.reid_candidate_rejected_total.labels(reason="cap_reached").inc()
                return
            confidence = (
                face_anchor.calibrated_confidence
                if face_anchor.calibrated_confidence is not None
                else face_anchor.match_score
            )
            await self._persist(
                ctx,
                det,
                det_idx,
                ph_id=ph_id,
                identity_id=committed_identity_id,
                orientation=orientation,
                eligibility=eligibility,
                confidence=confidence,
                seed_association_id=None,
            )
            return

        _metrics.metrics.reid_candidate_rejected_total.labels(reason=eligibility.reason).inc()
        evidence = ctx.wearable_evidence_by_ph.get(ph_id)
        if self._policy.wearable_seeding_enabled and evidence is not None and evidence.attested:
            await self._maybe_create_wearable_candidate(
                ctx,
                det,
                det_idx,
                ph_id=ph_id,
                committed_identity_id=committed_identity_id,
                evidence=evidence,
                detection_faces=tuple(faces_by_det.get(det_id, ())),
                orientation=orientation,
                orientation_confidence=orientation_confidence,
            )

    async def _maybe_create_wearable_candidate(
        self,
        ctx: FrameContext,
        det: Detection,
        det_idx: int,
        *,
        ph_id: str,
        committed_identity_id: str | None,
        evidence: WearableEvidence,
        detection_faces: tuple[FaceAnchor, ...],
        orientation: OrientationBin,
        orientation_confidence: float,
    ) -> None:
        """Seed from an attested wearable association when no face seed qualified.

        The repository is the only state: the seed interval and the diversity check read the
        identity's own recent rows, so a restart neither resets the spacing nor needs an
        in-memory map keyed by identity.
        """
        assert self._gallery_repo is not None
        seed_outcomes = _metrics.metrics.cts_wearable_seed_outcomes_total
        if self._teaching_holds is not None and await self._teaching_holds.is_held(evidence.tag_id):
            seed_outcomes.labels(outcome=WearableSeedOutcome.TEACHING_HELD).inc()
            return
        eligibility = evaluate_candidate(
            committed_identity_id=committed_identity_id,
            seed=WearableSeed(evidence=evidence, detection_faces=detection_faces),
            embedding=det.embedding,
            quality=det.crop_quality,
            orientation=orientation,
            orientation_confidence=orientation_confidence,
            cfg=self._policy,
        )
        if not eligibility.eligible:
            seed_outcomes.labels(outcome=WearableSeedOutcome(eligibility.reason)).inc()
            return
        seed_identity_id = committed_identity_id or evidence.identity_id
        assert det.embedding is not None

        capture_time = det.capture_time
        if eligibility.seed_source == GallerySeedSource.WEARABLE_ROOM_ATTESTED:
            minted_today = await self._gallery_repo.count_seeded_entries(
                identity_id=seed_identity_id,
                seed_source=GallerySeedSource.WEARABLE_ROOM_ATTESTED,
                seen_since=_utc_day_start(capture_time),
            )
            if minted_today >= self._policy.wearable_room_grade_max_per_day:
                seed_outcomes.labels(outcome=WearableSeedOutcome.ROOM_DAILY_CAP).inc()
                return

        if await self._at_orientation_cap(
            identity_id=seed_identity_id,
            orientation=orientation,
            capture_time=capture_time,
        ):
            seed_outcomes.labels(outcome=WearableSeedOutcome.CAP_REACHED).inc()
            return

        rejection = await self._recent_rows_rejection(
            identity_id=seed_identity_id,
            orientation=orientation,
            embedding=list(det.embedding),
            capture_time=capture_time,
        )
        if rejection is not None:
            seed_outcomes.labels(outcome=rejection).inc()
            return

        created = await self._persist(
            ctx,
            det,
            det_idx,
            ph_id=ph_id,
            identity_id=seed_identity_id,
            orientation=orientation,
            eligibility=eligibility,
            confidence=None,
            seed_association_id=evidence.segment_id,
        )
        seed_outcomes.labels(
            outcome=(
                (
                    WearableSeedOutcome.MINTED_AUTO_VERIFIED
                    if eligibility.mint_state == "auto_verified"
                    else WearableSeedOutcome.MINTED_PENDING_REVIEW
                )
                if created
                else WearableSeedOutcome.CREATE_FAILED
            )
        ).inc()

    async def _at_orientation_cap(
        self,
        *,
        identity_id: str,
        orientation: OrientationBin,
        capture_time: datetime,
    ) -> bool:
        """Whether *(identity_id, orientation)* has reached its working-set cap.

        Both seed paths share this check: only rows inside the active vote window count,
        so yesterday's outfit never prevents today's samples from minting.
        """
        assert self._gallery_repo is not None
        max_age_s = self._policy.gallery_vote_max_age_s
        seen_since = capture_time - timedelta(seconds=max_age_s) if max_age_s is not None else None
        existing = await self._gallery_repo.count_gallery_entries(
            identity_id=identity_id,
            orientation=int(orientation),
            seen_since=seen_since,
        )
        return existing >= self._policy.max_per_identity_orientation

    async def _recent_rows_rejection(
        self,
        *,
        identity_id: str,
        orientation: OrientationBin,
        embedding: list[float],
        capture_time: datetime,
    ) -> WearableSeedOutcome | None:
        """Seed spacing and diversity against the identity's own recent rows of this orientation."""
        assert self._gallery_repo is not None
        interval = timedelta(seconds=self._policy.wearable_min_seed_interval_s)
        day_start = _utc_day_start(capture_time)
        window_start = min(day_start, capture_time - interval)
        rows = [
            row
            for row in await self._gallery_repo.list_gallery_entries(
                identity_id=identity_id,
                # The identity is committed on a live PH, so its activity flag adds nothing.
                active_only=False,
                # Pending rows count: an unreviewed seed still occupies its slot and its pose.
                states=PENDING_AND_VERIFIED,
            )
            if row.orientation == int(orientation) and row.seen_at >= window_start
        ]
        if any(
            row.seed_source in _WEARABLE_SEED_SOURCES and row.seen_at > capture_time - interval
            for row in rows
        ):
            return WearableSeedOutcome.SEED_INTERVAL
        bar = self._policy.wearable_max_similarity_to_recent
        if any(
            row.seen_at >= day_start and row.embedding and _cosine(row.embedding, embedding) > bar
            for row in rows
        ):
            return WearableSeedOutcome.TOO_SIMILAR_TO_RECENT
        return None

    async def _persist(
        self,
        ctx: FrameContext,
        det: Detection,
        det_idx: int,
        *,
        ph_id: str,
        identity_id: str,
        orientation: OrientationBin,
        eligibility: CandidateEligibility,
        confidence: float | None,
        seed_association_id: str | None,
    ) -> bool:
        """Upload the crop and create the row; returns whether the row now exists."""
        assert self._gallery_repo is not None
        assert det.embedding is not None
        if det_idx >= len(ctx.crops):
            return False
        crop = ctx.crops[det_idx]

        candidate_id = str(uuid.uuid4())
        crop_bytes = _encode_jpeg(crop)
        crop_hash = hashlib.sha256(crop_bytes).hexdigest()
        crop_key = f"reid-candidates/{self._policy.model_version}/{candidate_id}.jpg"

        assert self._crop_storage is not None  # checked by caller
        try:
            await self._crop_storage.put_bytes(crop_key, crop_bytes)
        except Exception:
            # a storage hiccup on one detection must never fail the tracking frame.
            logger.warning(
                "reid_candidate_crop_upload_failed",
                candidate_id=candidate_id,
                crop_key=crop_key,
                exc_info=True,
            )
            return False

        ew, eh = ctx.effective_width, ctx.effective_height
        edge_truncated = (
            det.bbox.x_min <= 0
            or det.bbox.y_min <= 0
            or det.bbox.x_max >= ew
            or det.bbox.y_max >= eh
        )
        observation_id = ctx.det_to_observation_id.get(det.detection_id, "")
        seed_source = eligibility.seed_source

        candidate = NewReviewCandidate(
            candidate_id=candidate_id,
            identity_id=identity_id,
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
            candidate_reason=(
                "face_derived" if seed_source == GallerySeedSource.FACE else str(seed_source)
            ),
            # Episode = one PH's continuous track: gives the resolver's
            # per-episode vote cap a real group key (see cts-identity-governance).
            source_episode_id=ph_id,
            created_actor="pipeline",
            model_version=self._policy.model_version,
            preprocessing_version=self._policy.preprocessing_version,
            confidence=confidence,
            state=eligibility.mint_state,
            seed_source=seed_source,
            seed_association_id=seed_association_id,
        )

        try:
            await self._gallery_repo.create_review_candidate(candidate)
        except Exception:
            # orphaned crop object on DB failure so a retry with the same
            # candidate_id (idempotent on create_review_candidate) does not leak.
            logger.warning(
                "reid_candidate_create_failed",
                candidate_id=candidate_id,
                exc_info=True,
            )
            try:
                await self._crop_storage.delete_object(crop_key)
            except Exception:
                logger.warning(
                    "reid_candidate_orphan_crop_delete_failed",
                    crop_key=crop_key,
                    exc_info=True,
                )
            return False

        _metrics.metrics.reid_candidate_created_total.labels(state=eligibility.mint_state).inc()
        logger.debug(
            "reid_candidate_created",
            candidate_id=candidate_id,
            identity_id=identity_id,
            ph_id=ph_id,
            orientation=int(orientation),
            state=eligibility.mint_state,
            seed_source=str(seed_source),
        )
        return True
