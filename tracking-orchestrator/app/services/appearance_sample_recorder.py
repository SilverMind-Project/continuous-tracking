"""Records high-trust body-appearance samples for offline household classifier training.

Samples are captured only from:
- calibrated direct-face authority decisions (never posterior-only or temporal-prior labels);
- trajectory-grade wearable-attested intervals (room-grade wearable supervision enters
  training only after explicit operator approval, never automatically here);
- verified open-set negatives (unrecognized direct face on an uncommitted PH, or a PH
  explicitly marked as a named guest), capped per local calendar day.

Enforces crop quality, finite non-zero L2-normalized embeddings, wearable teaching holds,
face-conflict vetoes, and a per-(PH, orientation) capture-time deduplication window.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from structlog import get_logger

from app.domain import (
    AppearanceLabelSource,
    AppearanceTrainingSample,
    Detection,
    FaceAnchor,
    IdentityDecision,
    OrientationBin,
    WearableAttestationGrade,
    WearableEvidence,
)
from app.observability import metrics as _metrics
from app.observability.metrics import AppearanceSampleDropReason
from app.storage.appearance_samples import AppearanceSampleRepository
from app.storage.wearable_teaching_holds import WearableTeachingHoldRepository
from app.tracking.identity.types import IdentityAuthority

logger = get_logger(__name__)

_NORM_EPS = 1e-8
_DEDUP_PRUNE_MULT = 6.0


@dataclass(frozen=True)
class AppearanceSampleRecorderConfig:
    """Configuration governing multi-day body-appearance sample capture."""

    enabled: bool = False
    feature_version: str = "reid-solider"
    tz_name: str = "America/New_York"
    min_quality: float = 0.35
    dedup_window_s: float = 10.0
    max_negatives_per_day: int = 500


def _normalize_embedding(values: Sequence[float]) -> tuple[float, ...] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        return None
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= _NORM_EPS:
        return None
    normed = arr / norm
    return tuple(float(v) for v in normed)


class AppearanceSampleRecorder:
    """Evaluates per-frame detections after world tracking and persists eligible samples."""

    def __init__(
        self,
        *,
        repo: AppearanceSampleRepository,
        config: AppearanceSampleRecorderConfig,
        teaching_holds: WearableTeachingHoldRepository | None = None,
        guest_identity_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._repo = repo
        self._config = config
        self._teaching_holds = teaching_holds
        self._guest_identity_ids = guest_identity_ids
        self._tz = ZoneInfo(config.tz_name)
        # Capture-time deduplication table keyed by (ph_id, orientation), never by identity_id.
        self._last_recorded: dict[tuple[str, OrientationBin], datetime] = {}
        # Lazily hydrated daily negative count keyed by local calendar date.
        self._negatives_today: dict[date, int] = {}

    async def record_frame(
        self,
        *,
        detections: Sequence[Detection],
        det_to_ph: Mapping[str, str],
        orientation_by_detection: Mapping[str, tuple[OrientationBin, float]],
        decisions: Sequence[IdentityDecision],
        committed_ids: Mapping[str, str | None],
        wearable_evidence_by_ph: Mapping[str, WearableEvidence],
        face_anchors: Sequence[FaceAnchor],
        shadow_face_anchors: Sequence[FaceAnchor] = (),
        guest_identity_ids: frozenset[str] | None = None,
    ) -> list[AppearanceTrainingSample]:
        """Inspect one frame's detections and persist all eligible training samples."""
        if not self._config.enabled or not det_to_ph or not detections:
            return []

        guests = self._guest_identity_ids if guest_identity_ids is None else guest_identity_ids
        decisions_by_ph = {d.ph_id: d for d in decisions}
        faces_by_det: dict[str, list[FaceAnchor]] = {}
        for fa in (*face_anchors, *shadow_face_anchors):
            if fa.detection_id:
                faces_by_det.setdefault(fa.detection_id, []).append(fa)

        to_persist: list[AppearanceTrainingSample] = []
        pending_dedup: dict[tuple[str, OrientationBin], datetime] = {}
        pending_negatives: dict[date, int] = {}
        newest_time: datetime | None = None

        for det in detections:
            ph_id = det_to_ph.get(det.detection_id)
            if not ph_id:
                continue

            capture_time = det.capture_time
            if newest_time is None or capture_time > newest_time:
                newest_time = capture_time

            if det.crop_quality < self._config.min_quality:
                self._count_drop(AppearanceSampleDropReason.LOW_QUALITY)
                continue

            if not det.embedding:
                self._count_drop(AppearanceSampleDropReason.NO_EMBEDDING)
                continue

            normed = _normalize_embedding(det.embedding)
            if normed is None:
                self._count_drop(AppearanceSampleDropReason.NON_FINITE_EMBEDDING)
                continue

            orientation, _orient_conf = orientation_by_detection.get(
                det.detection_id, (OrientationBin.UNKNOWN, 0.0)
            )
            det_faces = tuple(faces_by_det.get(det.detection_id, ()))
            decision = decisions_by_ph.get(ph_id)
            committed_id = committed_ids.get(ph_id)
            wearable_ev = wearable_evidence_by_ph.get(ph_id)

            label_result = await self._classify_label(
                decision=decision,
                committed_id=committed_id,
                wearable_ev=wearable_ev,
                det_faces=det_faces,
                guests=guests,
            )
            if label_result is None:
                continue
            label_source, identity_id, seed_assoc_id = label_result

            # Check capture-time deduplication window per (ph_id, orientation).
            dedup_key = (ph_id, orientation)
            prev_at = pending_dedup.get(dedup_key) or self._last_recorded.get(dedup_key)
            if (
                prev_at is not None
                and (capture_time - prev_at).total_seconds() < self._config.dedup_window_s
            ):
                self._count_drop(AppearanceSampleDropReason.DEDUP_WINDOW)
                continue

            local_day = capture_time.astimezone(self._tz).date()
            if (
                label_source == AppearanceLabelSource.NEGATIVE
                and not await self._can_record_negative(
                    local_day, pending_negatives.get(local_day, 0)
                )
            ):
                self._count_drop(AppearanceSampleDropReason.NEGATIVE_DAILY_CAP)
                continue

            sample = AppearanceTrainingSample(
                sample_id=str(uuid.uuid4()),
                observed_at=capture_time,
                day=local_day,
                identity_id=identity_id,
                label_source=label_source,
                camera_id=det.camera_id,
                orientation=orientation,
                quality=float(det.crop_quality),
                feature_version=self._config.feature_version,
                embedding=normed,
                seed_association_id=seed_assoc_id,
            )
            to_persist.append(sample)
            pending_dedup[dedup_key] = capture_time
            if label_source == AppearanceLabelSource.NEGATIVE:
                pending_negatives[local_day] = pending_negatives.get(local_day, 0) + 1

        if newest_time is not None:
            self._prune_dedup_state(newest_time)

        if not to_persist:
            return []

        try:
            await self._repo.append_many(to_persist)
        except Exception:
            logger.warning(
                "appearance_training_samples_persist_failed",
                sample_count=len(to_persist),
                exc_info=True,
            )
            return []

        self._last_recorded.update(pending_dedup)
        for day_key, delta in pending_negatives.items():
            self._negatives_today[day_key] = self._negatives_today.get(day_key, 0) + delta
            if len(self._negatives_today) > 4:
                oldest = min(self._negatives_today)
                if oldest != day_key:
                    del self._negatives_today[oldest]

        for sample in to_persist:
            _metrics.metrics.cts_appearance_training_samples_recorded_total.labels(
                source=str(sample.label_source)
            ).inc()
        return to_persist

    async def _classify_label(
        self,
        *,
        decision: IdentityDecision | None,
        committed_id: str | None,
        wearable_ev: WearableEvidence | None,
        det_faces: Sequence[FaceAnchor],
        guests: frozenset[str],
    ) -> tuple[AppearanceLabelSource, str | None, str | None] | None:
        """Determine label source, identity_id, and optional association segment id."""
        # 1. Calibrated direct-face authority for an enrolled household resident.
        if (
            decision is not None
            and decision.authority == str(IdentityAuthority.DIRECT_FACE)
            and decision.identity_id is not None
            and not decision.conflict
            and decision.identity_id not in guests
        ):
            return (AppearanceLabelSource.DIRECT_FACE, decision.identity_id, None)

        # 2. Trajectory-attested wearable interval (room-grade enters only via operator approval).
        if wearable_ev is not None and wearable_ev.attested:
            if wearable_ev.attestation_grade != WearableAttestationGrade.TRAJECTORY:
                self._count_drop(AppearanceSampleDropReason.ROOM_GRADE_EXCLUDED)
                return None
            if wearable_ev.identity_id in guests:
                self._count_drop(AppearanceSampleDropReason.NOT_ELIGIBLE)
                return None
            if self._teaching_holds is not None and await self._teaching_holds.is_held(
                wearable_ev.tag_id
            ):
                self._count_drop(AppearanceSampleDropReason.TEACHING_HELD)
                return None
            if any(
                fa.recognition_state == "recognized"
                and fa.person_id
                and fa.person_id != wearable_ev.identity_id
                for fa in det_faces
            ):
                self._count_drop(AppearanceSampleDropReason.FACE_CONFLICT)
                return None
            return (
                AppearanceLabelSource.WEARABLE_ATTESTED,
                wearable_ev.identity_id,
                wearable_ev.segment_id,
            )

        # 3. Verified open-set negative: named guest, or uncommitted PH with an unrecognized face.
        has_resident_face = any(
            fa.recognition_state in ("recognized", "candidate")
            and fa.person_id
            and fa.person_id != "unknown"
            and fa.person_id not in guests
            for fa in det_faces
        )
        if not has_resident_face:
            if committed_id is not None and committed_id in guests:
                return (AppearanceLabelSource.NEGATIVE, None, None)
            has_unrecognized_face = any(fa.recognition_state == "unrecognized" for fa in det_faces)
            if committed_id is None and has_unrecognized_face:
                return (AppearanceLabelSource.NEGATIVE, None, None)

        self._count_drop(AppearanceSampleDropReason.NOT_ELIGIBLE)
        return None

    async def _can_record_negative(self, local_day: date, pending_for_day: int) -> bool:
        cap = self._config.max_negatives_per_day
        if cap <= 0:
            return False
        current = self._negatives_today.get(local_day)
        if current is None:
            current = await self._repo.count_negatives_for_day(
                local_day, feature_version=self._config.feature_version
            )
            self._negatives_today[local_day] = current
        return (current + pending_for_day) < cap

    def _prune_dedup_state(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=max(60.0, self._config.dedup_window_s * _DEDUP_PRUNE_MULT))
        stale = [key for key, ts in self._last_recorded.items() if ts < cutoff]
        for key in stale:
            del self._last_recorded[key]

    @staticmethod
    def _count_drop(reason: AppearanceSampleDropReason) -> None:
        _metrics.metrics.cts_appearance_training_samples_dropped_total.labels(
            reason=str(reason)
        ).inc()
