"""Storage protocols — Protocol + InMemory implementations."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from structlog import get_logger

from ..domain import (
    IdentityRevision,
    Keyframe,
    PersonHypothesis,
    WorldObservation,
)
from .annotations import BboxAnnotationRepository, InMemoryBboxAnnotationRepository
from .gallery import GalleryRepository, InMemoryGalleryRepository
from .misc import (
    ActivityRepository,
    AssignmentRepository,
    CorrectionRepository,
    InMemoryActivityRepository,
    InMemoryAssignmentRepository,
    InMemoryCorrectionRepository,
    InMemoryPrivacyRepository,
    InMemorySettingsRepository,
    PrivacyRepository,
    SettingsRepository,
)
from .signals import (
    BehaviorBaselineRepository,
    DementiaSignalRepository,
    HourlyActivitySummary,
    InMemoryBehaviorBaselineRepository,
    InMemoryDementiaSignalRepository,
    StillnessEpisode,
)
from .trajectory import (
    InMemoryKeyframeRepository,
    InMemoryTrajectoryRepository,
    KeyframeRepository,
    TrajectoryRepository,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PHRepositoryProtocol
# ---------------------------------------------------------------------------


class PHRepositoryProtocol(Protocol):
    """Persist Person Hypotheses and their observations (structural interface)."""

    async def save(self, ph: PersonHypothesis) -> None: ...
    async def get(self, ph_id: str) -> PersonHypothesis | None: ...
    async def get_by_id(self, ph_id: str) -> PersonHypothesis | None: ...
    async def list_open(self) -> list[PersonHypothesis]: ...
    async def list_closed_since(
        self, since: datetime, limit: int = 100
    ) -> list[PersonHypothesis]: ...
    async def update_identity(
        self, ph_id: str, identity_id: str | None, committed_at: datetime
    ) -> None: ...

    async def list_active(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        room_id: str | None = None,
        identity_id: str | None = None,
        state: str | None = None,
        include_transient: bool = False,
        min_duration_s: float | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]: ...

    async def list_history(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]: ...

    async def list_observations(
        self, ph_id: str, *, limit: int = 200
    ) -> list[WorldObservation]: ...
    async def list_observations_by_ph(
        self, ph_id: str, *, limit: int = 200
    ) -> list[WorldObservation]: ...
    async def get_observations(
        self, ph_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[WorldObservation], int]: ...
    async def get_trail(
        self, ph_id: str, *, since: datetime | None = None
    ) -> list[dict[str, Any]]: ...
    async def get_co_present(
        self, ph_id: str, *, at: datetime | None = None, radius_m: float = 5.0
    ) -> list[PersonHypothesis]: ...
    async def get_keyframes(
        self, ph_id: str, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[Keyframe], int]: ...

    async def correct_identity(
        self,
        ph_id: str,
        *,
        new_identity_id: str | None,
        reason: str,
        actor: str,
        idempotency_key: str | None = None,
    ) -> IdentityRevision: ...
    async def merge(
        self,
        *,
        source_ph_id: str,
        target_ph_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> IdentityRevision: ...
    async def batch_merge(
        self,
        *,
        source_ph_ids: list[str],
        target_ph_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> list[IdentityRevision]: ...
    async def split(
        self,
        ph_id: str,
        *,
        at_observation_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> tuple[str, str]: ...
    async def batch_correct(
        self,
        ph_ids: list[str],
        new_identity_ids: list[str | None],
        actor: str,
        reasons: list[str],
        idempotency_key: str | None = None,
    ) -> list[IdentityRevision]: ...
    async def delete_many(self, ph_ids: list[str], *, actor: str, reason: str) -> int: ...
    async def purge_unknown_older_than(self, cutoff: datetime, *, limit: int = 1000) -> int: ...
    async def list_revisions(
        self,
        *,
        ph_id: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        before_id: str | None = None,
    ) -> tuple[list[IdentityRevision], bool]: ...


class WorldObservationRepositoryProtocol(Protocol):
    """Persist individual world observations linked to a PH."""

    async def save(self, observation: WorldObservation, ph_id: str) -> str: ...
    async def list_by_ph(self, ph_id: str, limit: int = 50) -> list[WorldObservation]: ...


# ---------------------------------------------------------------------------
# InMemoryPHRepository
# ---------------------------------------------------------------------------


class InMemoryPHRepository:
    """In-memory store for Person Hypotheses — uses dict/list only, zero I/O."""

    def __init__(self) -> None:
        import asyncio

        self._phs: dict[str, PersonHypothesis] = {}
        self._observations: dict[str, list[WorldObservation]] = {}
        self._revisions: list[IdentityRevision] = []
        self._merges: dict[str, str] = {}
        self._idempotency: dict[str, IdentityRevision] = {}
        self._batch_idempotency: dict[str, list[IdentityRevision]] = {}
        self._lock = asyncio.Lock()

    # -- save / get / list_open / list_closed_since / update_identity --

    async def save(self, ph: PersonHypothesis) -> None:
        async with self._lock:
            self._phs[ph.ph_id] = ph

    async def get(self, ph_id: str) -> PersonHypothesis | None:
        return self._phs.get(ph_id)

    async def get_by_id(self, ph_id: str) -> PersonHypothesis | None:
        return self._phs.get(ph_id)

    async def list_open(self) -> list[PersonHypothesis]:
        return [ph for ph in self._phs.values() if ph.closed_at is None]

    async def list_closed_since(self, since: datetime, limit: int = 100) -> list[PersonHypothesis]:
        closed = [
            ph for ph in self._phs.values() if ph.closed_at is not None and ph.closed_at >= since
        ]
        closed.sort(key=lambda ph: ph.closed_at, reverse=True)  # type: ignore[arg-type,return-value]
        return closed[:limit]

    async def update_identity(
        self, ph_id: str, identity_id: str | None, committed_at: datetime
    ) -> None:
        async with self._lock:
            ph = self._phs.get(ph_id)
            if ph is not None:
                self._phs[ph_id] = PersonHypothesis(
                    ph_id=ph.ph_id,
                    state_mean=ph.state_mean,
                    state_cov=ph.state_cov,
                    born_at=ph.born_at,
                    last_seen_at=ph.last_seen_at,
                    last_seen_camera=ph.last_seen_camera,
                    observation_count=ph.observation_count,
                    current_identity_id=identity_id,
                    current_identity_committed_at=committed_at,
                    gallery_mean=ph.gallery_mean,
                    height_estimate_m=ph.height_estimate_m,
                    active_cameras=ph.active_cameras,
                    closed_at=ph.closed_at,
                    last_floor_speed_m_s=ph.last_floor_speed_m_s,
                    last_posture=ph.last_posture,
                    metadata=ph.metadata,
                    mean_quality=ph.mean_quality,
                )

    # -- list_active --

    async def list_active(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        room_id: str | None = None,
        identity_id: str | None = None,
        state: str | None = None,
        include_transient: bool = False,
        min_duration_s: float | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]:
        if state == "ended":
            results = [
                ph
                for ph in self._phs.values()
                if ph.closed_at is not None
                and (since is None or ph.last_seen_at >= since)
                and (until is None or ph.born_at <= until)
                and (identity_id is None or ph.current_identity_id == identity_id)
                and (
                    search is None
                    or search.lower() in str(ph.metadata.get("display_name", "")).lower()
                )
            ]
        else:
            results = [
                ph
                for ph in self._phs.values()
                if ph.closed_at is None
                and (since is None or ph.last_seen_at >= since)
                and (until is None or ph.born_at <= until)
                and (identity_id is None or ph.current_identity_id == identity_id)
                and (
                    search is None
                    or search.lower() in str(ph.metadata.get("display_name", "")).lower()
                )
            ]
        if state == "active":
            results = [ph for ph in results if ph.closed_at is None]
        if not include_transient:
            now = datetime.now(UTC)
            results = [
                ph for ph in results if ((ph.closed_at or now) - ph.born_at).total_seconds() >= 2.0
            ]
        if min_duration_s is not None:
            now = datetime.now(UTC)
            results = [
                ph
                for ph in results
                if ((ph.closed_at or now) - ph.born_at).total_seconds() >= min_duration_s
            ]
        results.sort(key=lambda ph: ph.last_seen_at, reverse=True)
        total = len(results)
        return results[offset : offset + limit], total

    # -- list_history --

    async def list_history(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        identity_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PersonHypothesis], int]:
        results = [
            ph
            for ph in self._phs.values()
            if (since is None or ph.last_seen_at >= since)
            and (until is None or ph.last_seen_at <= until)
            and (identity_id is None or ph.current_identity_id == identity_id)
        ]
        results.sort(key=lambda ph: ph.last_seen_at, reverse=True)
        total = len(results)
        return results[offset : offset + limit], total

    # -- observations --

    async def list_observations(self, ph_id: str, *, limit: int = 200) -> list[WorldObservation]:
        obs_list = self._observations.get(ph_id, [])
        return obs_list[-limit:]

    async def list_observations_by_ph(
        self, ph_id: str, *, limit: int = 200
    ) -> list[WorldObservation]:
        return await self.list_observations(ph_id, limit=limit)

    async def get_observations(
        self, ph_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[WorldObservation], int]:
        obs_list = self._observations.get(ph_id, [])
        total = len(obs_list)
        page = obs_list[-limit - offset : -offset] if offset else obs_list[-limit:]
        return page, total

    async def get_trail(self, ph_id: str, *, since: datetime | None = None) -> list[dict[str, Any]]:
        obs_list = self._observations.get(ph_id, [])
        return [
            {
                "captured_at": obs.captured_at.isoformat(),
                "floor_x_m": obs.floor_point.x_mm / 1000.0,
                "floor_y_m": obs.floor_point.y_mm / 1000.0,
                "camera_id": obs.camera_id,
            }
            for obs in obs_list
            if since is None or obs.captured_at >= since
        ]

    async def get_co_present(
        self, ph_id: str, *, at: datetime | None = None, radius_m: float = 5.0
    ) -> list[PersonHypothesis]:
        ph = self._phs.get(ph_id)
        if ph is None:
            return []
        ref_time = at if at is not None else ph.last_seen_at
        ref_x_m, ref_y_m = ph.state_mean[0], ph.state_mean[1]
        return [
            p
            for p in self._phs.values()
            if p.ph_id != ph_id
            and p.closed_at is None
            and abs((p.last_seen_at - ref_time).total_seconds()) <= 30
            and math.hypot(p.state_mean[0] - ref_x_m, p.state_mean[1] - ref_y_m) <= radius_m
        ]

    async def get_keyframes(
        self, ph_id: str, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[Keyframe], int]:
        return [], 0

    # -- corrections --

    async def correct_identity(
        self,
        ph_id: str,
        *,
        new_identity_id: str | None,
        reason: str,
        actor: str,
        idempotency_key: str | None = None,
    ) -> IdentityRevision:
        if idempotency_key and idempotency_key in self._idempotency:
            return self._idempotency[idempotency_key]
        async with self._lock:
            ph = self._phs.get(ph_id)
            if ph is None:
                raise ValueError(f"PH not found: {ph_id}")
            # Cannot correct a closed PH.
            if ph.closed_at is not None:
                raise ValueError(
                    f"Cannot correct closed PH {ph_id} (closed at {ph.closed_at.isoformat()})"
                )
            previous = ph.current_identity_id
            now = datetime.now(UTC)
            self._phs[ph_id] = PersonHypothesis(
                ph_id=ph.ph_id,
                state_mean=ph.state_mean,
                state_cov=ph.state_cov,
                born_at=ph.born_at,
                last_seen_at=ph.last_seen_at,
                last_seen_camera=ph.last_seen_camera,
                observation_count=ph.observation_count,
                current_identity_id=new_identity_id,
                current_identity_committed_at=now,
                gallery_mean=ph.gallery_mean,
                height_estimate_m=ph.height_estimate_m,
                active_cameras=ph.active_cameras,
                closed_at=ph.closed_at,
                last_floor_speed_m_s=ph.last_floor_speed_m_s,
                last_posture=ph.last_posture,
                metadata=ph.metadata,
                mean_quality=ph.mean_quality,
            )
        revision = IdentityRevision(
            revision_id=str(uuid.uuid4()),
            ph_id=ph_id,
            previous_identity_id=previous,
            new_identity_id=new_identity_id,
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=1,
            evidence=None,
        )
        self._revisions.append(revision)
        if idempotency_key:
            self._idempotency[idempotency_key] = revision
        return revision

    async def merge(
        self,
        *,
        source_ph_id: str,
        target_ph_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> IdentityRevision:
        if idempotency_key and idempotency_key in self._idempotency:
            return self._idempotency[idempotency_key]
        async with self._lock:
            source = self._phs.get(source_ph_id)
            target = self._phs.get(target_ph_id)
            if source is None or target is None:
                raise ValueError("Source or target PH not found")
            # Cannot merge PHs with overlapping same-camera observations.
            if source.active_cameras & target.active_cameras:
                overlap_cams = source.active_cameras & target.active_cameras
                raise ValueError(
                    f"Cannot merge PHs with overlapping camera observations: {overlap_cams}"
                )
            src_obs = self._observations.pop(source_ph_id, [])
            self._observations.setdefault(target_ph_id, []).extend(src_obs)
            now = datetime.now(UTC)
            self._phs[source_ph_id] = PersonHypothesis(
                ph_id=source.ph_id,
                state_mean=source.state_mean,
                state_cov=source.state_cov,
                born_at=source.born_at,
                last_seen_at=source.last_seen_at,
                last_seen_camera=source.last_seen_camera,
                observation_count=source.observation_count,
                current_identity_id=source.current_identity_id,
                current_identity_committed_at=source.current_identity_committed_at,
                gallery_mean=source.gallery_mean,
                height_estimate_m=source.height_estimate_m,
                active_cameras=source.active_cameras,
                closed_at=now,
                last_floor_speed_m_s=source.last_floor_speed_m_s,
                last_posture=source.last_posture,
                metadata={**source.metadata, "merged_into_ph_id": target_ph_id},
                mean_quality=source.mean_quality,
            )
            self._merges[source_ph_id] = target_ph_id
        revision = IdentityRevision(
            revision_id=str(uuid.uuid4()),
            ph_id=source_ph_id,
            previous_identity_id=source.current_identity_id,
            new_identity_id=target.current_identity_id,
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=len(src_obs),
            evidence=None,
        )
        self._revisions.append(revision)
        if idempotency_key:
            self._idempotency[idempotency_key] = revision
        return revision

    async def batch_merge(
        self,
        *,
        source_ph_ids: list[str],
        target_ph_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> list[IdentityRevision]:
        if idempotency_key and idempotency_key in self._batch_idempotency:
            return self._batch_idempotency[idempotency_key]
        if target_ph_id in source_ph_ids:
            raise ValueError("Target PH cannot also be a merge source")
        if len(set(source_ph_ids)) != len(source_ph_ids):
            raise ValueError("Duplicate source PH IDs are not allowed")
        missing = [ph_id for ph_id in [target_ph_id, *source_ph_ids] if ph_id not in self._phs]
        if missing:
            raise ValueError(f"PH not found: {', '.join(missing)}")

        revisions: list[IdentityRevision] = []
        for source_ph_id in source_ph_ids:
            revisions.append(
                await self.merge(
                    source_ph_id=source_ph_id,
                    target_ph_id=target_ph_id,
                    actor=actor,
                    reason=reason,
                    idempotency_key=None,
                )
            )
        if idempotency_key:
            self._batch_idempotency[idempotency_key] = revisions
        return revisions

    async def split(
        self,
        ph_id: str,
        *,
        at_observation_id: str,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> tuple[str, str]:
        async with self._lock:
            ph = self._phs.get(ph_id)
            if ph is None:
                raise ValueError(f"PH not found: {ph_id}")
            obs_list = self._observations.get(ph_id, [])
            split_idx = None
            for i, obs in enumerate(obs_list):
                if getattr(obs, "observation_id", None) == at_observation_id:
                    split_idx = i
                    break
            if split_idx is None:
                raise ValueError(f"Observation {at_observation_id} not found")
            if split_idx == 0:
                raise ValueError("Cannot split at first observation")
            earlier_obs = obs_list[:split_idx]
            later_obs = obs_list[split_idx:]
            new_ph_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            self._phs[ph_id] = PersonHypothesis(
                ph_id=ph.ph_id,
                state_mean=ph.state_mean,
                state_cov=ph.state_cov,
                born_at=ph.born_at,
                last_seen_at=earlier_obs[-1].captured_at if earlier_obs else ph.last_seen_at,
                last_seen_camera=ph.last_seen_camera,
                observation_count=len(earlier_obs),
                current_identity_id=ph.current_identity_id,
                current_identity_committed_at=ph.current_identity_committed_at,
                gallery_mean=ph.gallery_mean,
                height_estimate_m=ph.height_estimate_m,
                active_cameras=ph.active_cameras,
                closed_at=now,
                last_floor_speed_m_s=ph.last_floor_speed_m_s,
                last_posture=ph.last_posture,
                metadata=ph.metadata,
                mean_quality=ph.mean_quality,
            )
            later_ph = PersonHypothesis(
                ph_id=new_ph_id,
                state_mean=ph.state_mean,
                state_cov=ph.state_cov,
                born_at=later_obs[0].captured_at,
                last_seen_at=later_obs[-1].captured_at,
                last_seen_camera=ph.last_seen_camera,
                observation_count=len(later_obs),
                current_identity_id=ph.current_identity_id,
                current_identity_committed_at=ph.current_identity_committed_at,
                gallery_mean=ph.gallery_mean,
                height_estimate_m=ph.height_estimate_m,
                active_cameras=ph.active_cameras,
                closed_at=None,
                last_floor_speed_m_s=ph.last_floor_speed_m_s,
                last_posture=ph.last_posture,
                metadata=ph.metadata,
                mean_quality=ph.mean_quality,
            )
            self._phs[new_ph_id] = later_ph
            self._observations[ph_id] = earlier_obs
            self._observations[new_ph_id] = later_obs
        revision = IdentityRevision(
            revision_id=str(uuid.uuid4()),
            ph_id=ph_id,
            previous_identity_id=ph.current_identity_id,
            new_identity_id=ph.current_identity_id,
            actor=actor,
            reason=reason,
            applied_at=now,
            rewritten_rows=len(later_obs),
            evidence=None,
        )
        self._revisions.append(revision)
        return ph_id, new_ph_id

    async def batch_correct(
        self,
        ph_ids: list[str],
        new_identity_ids: list[str | None],
        actor: str,
        reasons: list[str],
        idempotency_key: str | None = None,
    ) -> list[IdentityRevision]:
        now = datetime.now(UTC)
        revisions: list[IdentityRevision] = []
        async with self._lock:
            for ph_id, new_identity_id, reason in zip(
                ph_ids, new_identity_ids, reasons, strict=True
            ):
                ph = self._phs.get(ph_id)
                if ph is None:
                    raise ValueError(f"PH not found: {ph_id}")
                previous = ph.current_identity_id
                self._phs[ph_id] = PersonHypothesis(
                    ph_id=ph.ph_id,
                    state_mean=ph.state_mean,
                    state_cov=ph.state_cov,
                    born_at=ph.born_at,
                    last_seen_at=ph.last_seen_at,
                    last_seen_camera=ph.last_seen_camera,
                    observation_count=ph.observation_count,
                    current_identity_id=new_identity_id,
                    current_identity_committed_at=now,
                    gallery_mean=ph.gallery_mean,
                    height_estimate_m=ph.height_estimate_m,
                    active_cameras=ph.active_cameras,
                    closed_at=ph.closed_at,
                    last_floor_speed_m_s=ph.last_floor_speed_m_s,
                    last_posture=ph.last_posture,
                    metadata=ph.metadata,
                    mean_quality=ph.mean_quality,
                )
                revision = IdentityRevision(
                    revision_id=str(uuid.uuid4()),
                    ph_id=ph_id,
                    previous_identity_id=previous,
                    new_identity_id=new_identity_id,
                    actor=actor,
                    reason=reason,
                    applied_at=now,
                    rewritten_rows=1,
                    evidence=None,
                )
                self._revisions.append(revision)
                revisions.append(revision)
        return revisions

    async def delete_many(self, ph_ids: list[str], *, actor: str, reason: str) -> int:
        ph_id_set = set(ph_ids)
        async with self._lock:
            deleted = 0
            for ph_id in list(ph_id_set):
                if ph_id in self._phs:
                    deleted += 1
                    del self._phs[ph_id]
                    self._observations.pop(ph_id, None)
            self._revisions = [rev for rev in self._revisions if rev.ph_id not in ph_id_set]
            for source_ph_id, target_ph_id in list(self._merges.items()):
                if source_ph_id in ph_id_set or target_ph_id in ph_id_set:
                    del self._merges[source_ph_id]
            return deleted

    async def purge_unknown_older_than(self, cutoff: datetime, *, limit: int = 1000) -> int:
        candidates = [
            ph.ph_id
            for ph in sorted(self._phs.values(), key=lambda p: p.last_seen_at)
            if ph.current_identity_id is None
            and ph.closed_at is not None
            and ph.last_seen_at < cutoff
        ][:limit]
        return await self.delete_many(candidates, actor="system", reason="unknown_purge")

    async def list_revisions(
        self,
        *,
        ph_id: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        before_id: str | None = None,
    ) -> tuple[list[IdentityRevision], bool]:
        results = self._revisions
        if ph_id is not None:
            results = [r for r in results if r.ph_id == ph_id]
        results.sort(key=lambda r: r.applied_at, reverse=True)
        total = len(results)
        page = results[:limit]
        has_more = total > limit
        return page, has_more


class InMemoryWorldObservationRepository:
    """In-memory store for world observations — uses dict/list only, zero I/O."""

    def __init__(self) -> None:
        self._observations: dict[str, list[WorldObservation]] = {}

    async def save(self, observation: WorldObservation, ph_id: str) -> str:
        import uuid as _uuid

        oid = str(_uuid.uuid4())
        stored = WorldObservation(
            observation_id=oid,
            camera_id=observation.camera_id,
            frame_index=observation.frame_index,
            captured_at=observation.captured_at,
            floor_point=observation.floor_point,
            bbox=observation.bbox,
            embedding=observation.embedding,
            detection_confidence=observation.detection_confidence,
            height_estimate_m=observation.height_estimate_m,
            face_anchor=observation.face_anchor,
            detection_id=observation.detection_id,
            quality=observation.quality,
            floor_residual_m=observation.floor_residual_m,
        )
        self._observations.setdefault(ph_id, []).append(stored)
        return oid

    async def list_by_ph(self, ph_id: str, limit: int = 50) -> list[WorldObservation]:
        obs_list = self._observations.get(ph_id, [])
        return obs_list[-limit:]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "ActivityRepository",
    "AssignmentRepository",
    "BboxAnnotationRepository",
    "BehaviorBaselineRepository",
    "CorrectionRepository",
    "DementiaSignalRepository",
    "GalleryRepository",
    "HourlyActivitySummary",
    "InMemoryActivityRepository",
    "InMemoryAssignmentRepository",
    "InMemoryBboxAnnotationRepository",
    "InMemoryBehaviorBaselineRepository",
    "InMemoryCorrectionRepository",
    "InMemoryDementiaSignalRepository",
    "InMemoryGalleryRepository",
    "InMemoryKeyframeRepository",
    "InMemoryPHRepository",
    "InMemoryPrivacyRepository",
    "InMemorySettingsRepository",
    "InMemoryTrajectoryRepository",
    "InMemoryWorldObservationRepository",
    "KeyframeRepository",
    "PHRepositoryProtocol",
    "PrivacyRepository",
    "SettingsRepository",
    "StillnessEpisode",
    "TrajectoryRepository",
    "WorldObservationRepositoryProtocol",
]
