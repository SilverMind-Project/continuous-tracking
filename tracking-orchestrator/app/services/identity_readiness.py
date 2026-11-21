"""Read-only snapshot of the identity settings and evidence stores the CC readiness page needs.

The operator sees, in one place, which link of the identity chain is broken. CTS owns four of
those facts: the face and resolver policies the pipeline was built with, the governed ReID gallery
by lifecycle state, which enabled cameras have a floor homography at runtime, and how many overlap
groups are in use. This service reports them and nothing else: no identity ids, no positions, no
settings re-read from YAML (the policy snapshot is built from the same config objects the pipeline
was constructed with).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.calibration.state import CalibrationState
from app.domain import GallerySeedSource
from app.services.camera_room_map import CameraRoomMap
from app.storage.gallery import GalleryRepository


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
    ) -> None:
        self._policy = policy
        self._calibration_state = calibration_state
        self._camera_room_map = camera_room_map
        self._overlap_groups = overlap_groups
        self._gallery_repo = gallery_repo
        self._wearable_associations = wearable_associations

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
        return IdentityReadinessSnapshot(
            policy=self._policy,
            reid_gallery_counts=counts,
            cameras=cameras,
            overlap_group_count=self._overlap_groups.overlap_group_count(),
            identities_known=identities_known,
            wearable_active_associations=wearable_active,
            wearable_gallery_counts=wearable_counts,
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
    )
