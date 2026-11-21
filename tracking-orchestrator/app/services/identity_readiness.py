"""Read-only snapshot of the identity settings and evidence stores the CC readiness page needs.

The operator sees, in one place, which link of the identity chain is broken. CTS owns four of
those facts: the face and resolver policies the pipeline was built with, the governed ReID gallery
by lifecycle state, which enabled cameras have a floor homography at runtime, and how many overlap
groups are in use. This service reports them and nothing else: no identity ids, no positions, no
settings re-read from YAML (the policy snapshot is built from the same config objects the pipeline
was constructed with).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from app.calibration.state import CalibrationState
from app.domain import GallerySeedSource
from app.services.camera_room_map import CameraRoomMap
from app.services.image_only_evaluator import summarize_attribution_difference
from app.storage.gallery import GalleryRepository
from app.storage.image_only_evaluation import ImageOnlyEvaluationRepository
from app.storage.wearable_associations import WearableAssociationRepository
from app.storage.wearable_teaching_holds import WearableTeachingHoldRepository


@dataclass(frozen=True)
class IdentityPolicySnapshot:
    """The identity policies in force, copied from the built pipeline configuration."""

    face_evidence_policy: str
    face_templates_mode: str
    face_quality_mode: str
    cc_assertion_mode: str
    # None when the pipeline accepts any ArcFace model version.
    expected_arcface_model_version: str | None
    expected_quality_version: str
    wearable_mode: str = "off"
    wearable_seeding_enabled: bool = False
    wearable_face_attestation_mode: str = "off"
    wearable_backfill_enabled: bool = False
    wearable_auto_on_swap: bool = True
    wearable_evaluation_sample_rate: float = 0.0


@dataclass(frozen=True)
class CameraCalibrationReadiness:
    camera_id: str
    calibrated: bool


@dataclass(frozen=True)
class GalleryStateCounts:
    pending_review: int
    auto_verified: int
    operator_verified: int
    rejected: int
    expired: int = 0


@dataclass(frozen=True)
class WearableLearnersReadiness:
    """Read-only summary of the wearable-supervision learners, evaluation, and holds."""

    wearable_mode: str
    wearable_seeding_enabled: bool
    face_attestation_mode: str
    wearable_backfill_enabled: bool
    auto_on_swap: bool
    evaluation_sample_rate: float
    evaluation_days_14d: int
    attribution_difference_14d: float | None
    open_teaching_holds: int
    revoked_segments_14d: int


@dataclass(frozen=True)
class IdentityReadinessSnapshot:
    policy: IdentityPolicySnapshot
    # None when this process runs without a gallery repository (no database).
    reid_gallery_counts: GalleryStateCounts | None
    # Enabled cameras from the last CC config sync, each with whether a homography is in use.
    cameras: tuple[CameraCalibrationReadiness, ...]
    # None when the world tracker is not built.
    overlap_group_count: int | None
    # None when this process runs without a gallery repository.
    identities_known: int | None
    # Count of currently open wearable-to-PH association segments; None when not wired.
    wearable_active_associations: int | None = None
    # Gallery rows a wearable taught (trajectory- or room-attested seeds), by state; None without
    # a gallery repository.
    wearable_gallery_counts: GalleryStateCounts | None = None
    # Wearable-supervision learner flags, 14-day paired evaluation summary, and revocation holds.
    wearable_learners: WearableLearnersReadiness | None = None


class OverlapGroupSource(Protocol):
    def overlap_group_count(self) -> int | None: ...


class WearableAssociationCountSource(Protocol):
    def active_wearable_association_count(self) -> int | None: ...


class IdentityReadinessService:
    def __init__(
        self,
        *,
        policy: IdentityPolicySnapshot,
        calibration_state: CalibrationState,
        camera_room_map: CameraRoomMap,
        overlap_groups: OverlapGroupSource,
        gallery_repo: GalleryRepository | None = None,
        wearable_associations: WearableAssociationCountSource | None = None,
        image_only_eval_repo: ImageOnlyEvaluationRepository | None = None,
        wearable_assoc_repo: WearableAssociationRepository | None = None,
        teaching_holds_repo: WearableTeachingHoldRepository | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._policy = policy
        self._calibration_state = calibration_state
        self._camera_room_map = camera_room_map
        self._overlap_groups = overlap_groups
        self._gallery_repo = gallery_repo
        self._wearable_associations = wearable_associations
        self._image_only_eval_repo = image_only_eval_repo
        self._wearable_assoc_repo = wearable_assoc_repo
        self._teaching_holds_repo = teaching_holds_repo
        self._clock = clock

    async def snapshot(self) -> IdentityReadinessSnapshot:
        calibrated_ids = self._calibration_state.calibrated_camera_ids()
        cameras = tuple(
            CameraCalibrationReadiness(
                camera_id=binding.camera_id, calibrated=binding.camera_id in calibrated_ids
            )
            for binding in await self._camera_room_map.snapshot()
        )
        counts: GalleryStateCounts | None = None
        wearable_counts: GalleryStateCounts | None = None
        identities_known: int | None = None
        if self._gallery_repo is not None:
            counts = _state_counts(await self._gallery_repo.count_review_queue())
            wearable_counts = _state_counts(
                await self._gallery_repo.count_review_queue(seed_sources=_WEARABLE_SEED_SOURCES)
            )
            identities_known = len(await self._gallery_repo.list_identities(active_only=True))
        wearable_active = (
            self._wearable_associations.active_wearable_association_count()
            if self._wearable_associations is not None
            else None
        )
        now = self._clock()
        since_day = (now - timedelta(days=13)).date()
        until_day = now.date()
        eval_days = 0
        attr_diff: float | None = None
        if self._image_only_eval_repo is not None:
            totals = await self._image_only_eval_repo.list_totals(since=since_day, until=until_day)
            eval_days, attr_diff = summarize_attribution_difference(totals)
        open_holds = (
            len(await self._teaching_holds_repo.list_open_holds())
            if self._teaching_holds_repo is not None
            else 0
        )
        revoked_14d = (
            await self._wearable_assoc_repo.count_revoked_since(now - timedelta(days=14))
            if self._wearable_assoc_repo is not None
            else 0
        )
        learners = WearableLearnersReadiness(
            wearable_mode=self._policy.wearable_mode,
            wearable_seeding_enabled=self._policy.wearable_seeding_enabled,
            face_attestation_mode=self._policy.wearable_face_attestation_mode,
            wearable_backfill_enabled=self._policy.wearable_backfill_enabled,
            auto_on_swap=self._policy.wearable_auto_on_swap,
            evaluation_sample_rate=self._policy.wearable_evaluation_sample_rate,
            evaluation_days_14d=eval_days,
            attribution_difference_14d=attr_diff,
            open_teaching_holds=open_holds,
            revoked_segments_14d=revoked_14d,
        )
        return IdentityReadinessSnapshot(
            policy=self._policy,
            reid_gallery_counts=counts,
            cameras=cameras,
            overlap_group_count=self._overlap_groups.overlap_group_count(),
            identities_known=identities_known,
            wearable_active_associations=wearable_active,
            wearable_gallery_counts=wearable_counts,
            wearable_learners=learners,
        )


_WEARABLE_SEED_SOURCES: frozenset[str] = frozenset(
    {str(GallerySeedSource.WEARABLE_ATTESTED), str(GallerySeedSource.WEARABLE_ROOM_ATTESTED)}
)


def _state_counts(by_state: dict[str, int]) -> GalleryStateCounts:
    return GalleryStateCounts(
        pending_review=by_state.get("pending_review", 0),
        auto_verified=by_state.get("auto_verified", 0),
        operator_verified=by_state.get("operator_verified", 0),
        rejected=by_state.get("rejected", 0),
        expired=by_state.get("expired", 0),
    )
