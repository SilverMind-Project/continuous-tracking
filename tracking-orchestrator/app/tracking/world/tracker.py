"""WorldTracker: top-level orchestrator for world-coordinate person tracking.

Depends on Protocols (PHRepositoryProtocol, WorldObservationRepositoryProtocol);
no direct I/O. Called by WorldTrackingStage once per frame.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import numpy as np
from structlog import get_logger

if TYPE_CHECKING:
    from app.inference.evidence import FaceEvidence
    from app.storage.base import GalleryRepository
    from app.tracking.identity_resolver import IdentityResolver

from app.domain import (
    AppearanceFeature,
    AppearanceFeatureExtractor,
    BoundingBox,
    CameraTopologyEdge,
    CoPresenceLink,
    FaceAnchor,
    IdentityDecision,
    IdentityResolvableEntity,
    IdentityRevision,
    ImageOnlyEvaluation,
    OpenPHOccupancy,
    OrientationBin,
    OverlapGroup,
    PersonHypothesis,
    PHContinuationCandidate,
    ViewPrototype,
    WearableAssociationTransition,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
    WearableFix,
    WorldFrameSnapshot,
    WorldObservation,
    tuple_to_cov2x2,
)
from app.observability import metrics as _metrics
from app.observability.metrics import WearableSpatialFusionOutcome
from app.storage.base import (
    VOTING_STATES,
    CameraTopologyRepository,
    CoPresenceRepository,
    PHRepositoryProtocol,
    WorldObservationRepositoryProtocol,
)
from app.tracking.camera_pose import update_ph_height_estimate
from app.tracking.identity.closed_world_channel import ClosedWorldConfig, canonical_ph_pair
from app.tracking.identity.evidence_grades import grade_of
from app.tracking.orientation import update_view_prototypes
from app.tracking.world.appearance_policy import AppearanceDecision, evaluate_appearance_update
from app.tracking.world.association import associate
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.dedup import DedupMergeStats, _finalize_singleton, dedup_observations
from app.tracking.world.helpers import (
    is_in_any_room_polygon,
    position_sigma_m,
    resolve_room,
    speed_m_s,
    update_gallery_mean,
)
from app.tracking.world.kalman import (
    KalmanState,
    initialize,
    isotropic_cov,
    mahalanobis2_position,
    predict,
    update,
    zero_velocity_update,
)
from app.tracking.world.ph_floor_history import PHFloorHistory
from app.tracking.world.revival import select_revival_candidate
from app.tracking.world.topology import record_handoff
from app.tracking.world.wearable_association import WearableAssociationTracker
from app.tracking.world.wearable_config import WearableConfig
from app.tracking.world.wearable_spatial_fusion import (
    MotionModel,
    fuse_position_fix,
    select_fusion_fix,
)

logger = get_logger(__name__)


def _metadata_with_room(
    metadata: dict[str, object],
    room_id: str,
    room_name: str,
) -> dict[str, object]:
    merged = dict(metadata)
    if room_id:
        merged["last_room_id"] = room_id
    if room_name:
        merged["last_room_name"] = room_name
    return merged


def _best_primary_camera(
    source_detection_ids: tuple[str, ...],
    obs_by_det_id: dict[str, WorldObservation],
    fallback: WorldObservation,
) -> str:
    candidates = [obs_by_det_id[did] for did in source_detection_ids if did in obs_by_det_id]
    if not candidates:
        candidates = [fallback]
    best = max(candidates, key=lambda obs: (obs.primary_score, obs.camera_id, obs.detection_id))
    return best.camera_id


class ContinuationPublisher(Protocol):
    """Publishes PHContinuationCandidate events to tracking.continuations."""

    async def publish(self, candidate: PHContinuationCandidate) -> object: ...


@dataclass
class _PHResolvable:
    """Thin adapter to supply real observation IDs to the identity resolver.

    PersonHypothesis is frozen, so observation_ids cannot be mutated onto it
    after construction.  This adapter wraps a PH and a pre-fetched list of
    observation IDs to satisfy the IdentityResolvableEntity protocol.
    """

    _ph: PersonHypothesis
    _obs_ids: list[str] = field(default_factory=list)

    @property
    def entity_id(self) -> str:
        return self._ph.ph_id

    @property
    def observation_ids(self) -> list[str]:
        return self._obs_ids

    @property
    def camera_ids(self) -> list[str]:
        return list(self._ph.active_cameras)

    @property
    def current_identity_id(self) -> str | None:
        return self._ph.current_identity_id

    @property
    def current_identity_committed_at(self) -> datetime | None:
        return self._ph.current_identity_committed_at

    @property
    def last_independent_identity_evidence_at(self) -> datetime | None:
        return self._ph.last_independent_identity_evidence_at

    @property
    def last_seen_at(self) -> datetime:
        return self._ph.last_seen_at

    @property
    def started_at(self) -> datetime:
        return self._ph.born_at

    @property
    def view_prototypes(self) -> tuple[ViewPrototype, ...]:
        return self._ph.view_prototypes


@dataclass(frozen=True)
class WorldTrackerResult:
    """Output of one WorldTracker.step() call."""

    updated_phs: list[PersonHypothesis]
    snapshots: list[WorldFrameSnapshot]
    continuations: list[PHContinuationCandidate]
    identity_decisions: list[IdentityDecision] = field(default_factory=list)
    revisions: list[IdentityRevision] = field(default_factory=list)
    det_to_ph: dict[str, str] = field(default_factory=dict)
    # Real repository-assigned WorldObservation IDs, keyed by source detection ID.
    # ReIDCandidateStage uses these for origin_tracklet_id lookups.
    det_to_observation_id: dict[str, str] = field(default_factory=dict)
    revived_ph_ids: frozenset[str] = frozenset()
    coasting_ph_ids: frozenset[str] = frozenset()
    wearable_association_transitions: tuple[WearableAssociationTransition, ...] = ()
    wearable_evidence_by_ph: Mapping[str, WearableEvidence] = field(default_factory=dict)
    image_only_evaluations: tuple[ImageOnlyEvaluation, ...] = ()


@dataclass(frozen=True)
class _ObsVectors:
    """Parallel arrays unpacked from a deduped observation list for association."""

    floor_points: list[tuple[float, float]]
    embeddings: list[list[float] | None]
    face_person_ids: list[str | None]
    face_confidences: list[float]
    heights: list[float | None]
    calibrated: list[bool]
    covs: list[tuple[float, float, float, float] | None]


@dataclass(frozen=True)
class _PHVectors:
    """Parallel arrays unpacked from the active-PH list for association."""

    gallery_means: list[list[float] | None]
    identity_ids: list[str | None]
    heights: list[float | None]
    view_prototypes: list[tuple[ViewPrototype, ...]]


def _sanitize_identity_id(raw: str | None) -> str | None:
    """Convert sentinel identity values (``""``, ``"unknown"``) to ``None``.

    These sentinels originate from the face-ID client and face-identity stage
    when no enrolled identity matches the detected face.  They must not be
    treated as valid identity references because ``person_trajectories`` and
    ``room_dwells`` enforce a FK to ``identities``.
    """
    if raw is not None and raw.lower() in ("", "unknown"):
        return None
    return raw


def _record_appearance_rejection(
    m: _metrics.Metrics,
    ph_id: str,
    obs: WorldObservation,
    decision: AppearanceDecision,
) -> None:
    """Emit the diagnostic for a barred PH-local appearance update (M03 task 9).

    Records the typed reason without labelling the embedding as the PH identity.
    """
    reason = str(decision.reason) if decision.reason is not None else "unknown"
    m.worldtracker_appearance_updates_rejected_total.labels(reason=reason).inc()
    logger.debug(
        "ph_appearance_update_rejected",
        ph_id=ph_id,
        camera_id=obs.camera_id,
        orientation=int(obs.orientation),
        reason=reason,
    )


def _unpack_observations(observations: list[WorldObservation]) -> _ObsVectors:
    """Unpack a deduped observation list into parallel arrays for association.

    Only recognized face anchors whose evidence grade allows association
    (``grade_of(fa).enters_association``) are forwarded to the identity-conflict
    gate; candidate, unrecognized, template, and external anchors never drive PH
    association or hard-conflict splits.
    """
    floor_points = [
        (obs.floor_point.x_mm / 1000.0, obs.floor_point.y_mm / 1000.0) for obs in observations
    ]
    embeddings: list[list[float] | None] = [obs.embedding or None for obs in observations]

    def _association_recognized(obs: WorldObservation) -> bool:
        fa = obs.face_anchor
        return (
            fa is not None
            and fa.recognition_state == "recognized"
            and grade_of(fa).enters_association
        )

    face_person_ids: list[str | None] = [
        obs.face_anchor.person_id if _association_recognized(obs) and obs.face_anchor else None
        for obs in observations
    ]
    face_confidences: list[float] = [
        obs.face_anchor.match_score if _association_recognized(obs) and obs.face_anchor else 0.0
        for obs in observations
    ]
    return _ObsVectors(
        floor_points=floor_points,
        embeddings=embeddings,
        face_person_ids=face_person_ids,
        face_confidences=face_confidences,
        heights=[obs.height_estimate_m for obs in observations],
        calibrated=[obs.floor_point.calibrated for obs in observations],
        covs=[obs.floor_cov_random for obs in observations],
    )


def _unpack_ph_vectors(phs: list[PersonHypothesis]) -> _PHVectors:
    """Unpack the active-PH list into parallel arrays for association."""
    return _PHVectors(
        gallery_means=[ph.gallery_mean for ph in phs],
        identity_ids=[ph.current_identity_id for ph in phs],
        heights=[ph.height_estimate_m for ph in phs],
        view_prototypes=[ph.view_prototypes for ph in phs],
    )


class WorldTracker:
    """Single floor-plane Kalman tracker for multi-camera person tracking.

    One instance per process. Called once per frame with all calibrated
    observations from all cameras. No per-camera tracker, no cross-camera
    merge pass, no healing pass.
    """

    def __init__(
        self,
        ph_repo: PHRepositoryProtocol,
        obs_repo: WorldObservationRepositoryProtocol,
        config: WorldTrackerConfig | None = None,
        continuation_publisher: ContinuationPublisher | None = None,
        identity_resolver: IdentityResolver | None = None,
        gallery_repo: GalleryRepository | None = None,
        topology_repo: CameraTopologyRepository | None = None,
        copresence_repo: CoPresenceRepository | None = None,
        overlap_groups: list[OverlapGroup] | None = None,
        wearable_config: WearableConfig | None = None,
        closed_world_config: ClosedWorldConfig | None = None,
        appearance_feature_extractor: AppearanceFeatureExtractor | None = None,
    ) -> None:
        self._ph_repo = ph_repo
        self._obs_repo = obs_repo
        self._config = config or WorldTrackerConfig()
        self._continuation_publisher = continuation_publisher
        self._identity_resolver = identity_resolver
        self._gallery_repo = gallery_repo
        self._topology_repo = topology_repo
        self._copresence_repo = copresence_repo
        self._overlap_groups: list[OverlapGroup] = list(overlap_groups or [])
        self._wearable_config = wearable_config or WearableConfig()
        self._closed_world_config = closed_world_config or ClosedWorldConfig()
        self._appearance_feature_extractor = appearance_feature_extractor
        self._ph_floor_history = PHFloorHistory(
            max_span_s=self._config.ph_close_grace_s + self._config.revive_max_age_s,
        )
        self._wearable_tracker: WearableAssociationTracker | None = (
            WearableAssociationTracker(self._wearable_config)
            if self._wearable_config.mode != "off"
            else None
        )
        # In-process cache of each PH's last positive identity confidence.
        # On coasting / unresolved frames the resolver produces no decision for a
        # PH, so its per-frame posterior is absent (0.0). Replaying the last
        # committed confidence here keeps the published top_probability meaningful
        # for a held identity instead of emitting a sentinel 0.0 that the UI shows
        # as null. Lost on restart (self-heals on next resolution); pruned on close.
        self._last_identity_confidence: dict[str, float] = {}
        # Tracks PHs whose current identity was supported solely by room-class
        # wearable evidence (excluded from closed-world joint elimination), plus
        # each PH's latest non-room-wearable evidence timestamp.
        self._room_wearable_only_phs: set[str] = set()
        self._last_elim_eligible_evidence_at: dict[str, datetime] = {}
        # Snapshot of open PHs from the most recent step call. Exposed via
        # last_open_phs so ReidNeedPolicy (InferenceStage) can evaluate proximity
        # without issuing an async DB query. Empty until the first step.
        self._last_open_phs: list[PersonHypothesis] = []
        self._still_counter: dict[str, int] = {}
        # Spatial fusion of attested wearable fixes: the last camera observation of each PH
        # (a fused fix may advance last_seen_at past it) and the newest fix already fused.
        self._camera_seen_at: dict[str, datetime] = {}
        self._fused_through: dict[str, datetime] = {}
        self._primary_camera: dict[str, str] = {}
        self._primary_challenger: dict[str, tuple[str, int]] = {}

    def purge_identity(self, identity_id: str) -> int:
        """Purge in-memory per-identity tracker state (wearable presence timestamps)."""
        if self._wearable_tracker is None:
            return 0
        return self._wearable_tracker.purge_identity(identity_id)

    @property
    def last_open_phs(self) -> list[PersonHypothesis]:
        """Open PHs from the most recent step() call.

        Synchronous snapshot for ReidNeedPolicy proximity checks.
        Empty until the first step() completes.
        """
        return self._last_open_phs

    @property
    def ph_floor_history(self) -> PHFloorHistory:
        """Calibrated floor-plane history maintained for wearable trajectory association."""
        return self._ph_floor_history

    def active_wearable_association_count(self) -> int:
        """Return the number of currently active wearable-to-PH association segments."""
        if self._wearable_tracker is None:
            return 0
        return self._wearable_tracker.active_segment_count

    def reset_ph_wearable_state(
        self, ph_id: str, *, now: datetime, reason: str = "ph_split_or_merge"
    ) -> list[WearableAssociationTransition]:
        """Clear calibrated floor history and close any open wearable segment on *ph_id*."""
        self._ph_floor_history.reset_ph(ph_id)
        self._camera_seen_at.pop(ph_id, None)
        self._fused_through.pop(ph_id, None)
        if self._wearable_tracker is None:
            return []
        return self._wearable_tracker.reset_ph(ph_id, now=now, reason=reason)

    async def _resolve_verified_reid_identities(
        self, observations: list[WorldObservation], as_of: datetime
    ) -> list[str | None]:
        """Resolve each observation's verified-ReID identity from the gallery.

        ``operator_verified`` and ``auto_verified`` gallery entries vote
        (``VOTING_STATES``): the probe sees the same corpus the resolver votes
        with. A capture-time vote window ``(as_of - reid_disagreement_max_age_s, as_of]``
        is applied at the repository boundary so stale clothing matches cannot
        disagree-cost a fresh observation and replays reproduce live behavior.
        Returns a list aligned to *observations*; an entry is ``None`` when there
        is no embedding, no gallery repository, or no voting-state match clears
        ``reid_disagreement_min_similarity``. The caller only invokes this when
        ``enable_reid_disagreement_cost`` is true, so it adds no per-frame gallery
        queries while the flag is off.
        """
        if self._gallery_repo is None:
            return [None] * len(observations)
        cfg = self._config
        max_age = cfg.reid_disagreement_max_age_s
        seen_after = as_of - timedelta(seconds=max_age) if max_age is not None else None
        resolved: list[str | None] = []
        for obs in observations:
            identity: str | None = None
            if obs.embedding:
                hits = await self._gallery_repo.search_similar(
                    obs.embedding,
                    limit=1,
                    seen_after=seen_after,
                    seen_before=as_of,
                    states=VOTING_STATES,
                )
                if hits:
                    entry, similarity = hits[0]
                    if (
                        entry.state in VOTING_STATES
                        and entry.identity_id
                        and similarity >= cfg.reid_disagreement_min_similarity
                    ):
                        identity = entry.identity_id
            resolved.append(identity)
        return resolved

    def set_overlap_groups(self, overlap_groups: Sequence[OverlapGroup]) -> None:
        """Replace the declared overlap groups; the next round's dedup and co-presence use them."""
        self._overlap_groups = list(overlap_groups)

    def overlap_group_count(self) -> int:
        """How many declared overlap groups dedup and co-presence use."""
        return len(self._overlap_groups)

    async def step(
        self,
        observations: list[WorldObservation],
        now: datetime,
        face_anchors_by_detection: dict[str, FaceAnchor] | None = None,
        room_polygons: dict[str, list[tuple[float, float]]] | None = None,
        camera_room_map: dict[str, str] | None = None,
        room_names: dict[str, str] | None = None,
        face_anchors: list[FaceAnchor] | None = None,
        face_evidence: list[FaceEvidence] | None = None,
        low_band_observations: list[WorldObservation] | None = None,
        wearable_fixes: Sequence[WearableFix] | None = None,
        camera_room_ids: dict[str, str] | None = None,
    ) -> WorldTrackerResult:
        """Run one frame of the world tracker.

        Args:
            observations: calibrated WorldObservations from all cameras.
            now: current wall-clock time (pipeline event_time).
            face_anchors_by_detection: face anchors keyed by detection_id.
            room_polygons: room_id → list of (x_m, y_m) vertices.
            room_names: room_id → display name for polygon-derived rooms.
            camera_room_map: camera_id → room_name fallback.
            camera_room_ids: camera_id → room id, the namespace of room polygons and
                wearable fix rooms; wearable association resolves PH rooms with it.
            face_anchors: flat list of FaceAnchors for identity resolution only. It may contain
                ``shadow_only`` anchors, which the resolver keeps out of every live decision;
                nothing else in the tracker reads this list.

        Returns:
            WorldTrackerResult with updated PHs, snapshots, and continuations.
        """
        cfg = self._config
        room_polygons = room_polygons or {}
        camera_room_map = camera_room_map or {}
        camera_room_ids = camera_room_ids or {}
        room_names = room_names or {}
        wcfg = self._wearable_config
        fusion_on = self._wearable_tracker is not None and wcfg.spatial_fusion_mode != "off"
        # Only when fusion applies does PH floor history record the raw camera measurements
        # of a round instead of the posterior: a posterior that absorbed a fused fix would let
        # trajectory association score the wearable against its own earlier fixes. Shadow
        # never moves PH state, so its history (and every association) stays as with off.
        fusion_applies = fusion_on and wcfg.spatial_fusion_mode == "enabled"
        camera_measurements: dict[str, list[tuple[datetime, float, float, np.ndarray]]] = {}
        continuations: list[PHContinuationCandidate] = []
        identity_decisions: list[IdentityDecision] = []
        det_to_ph: dict[str, str] = {}
        det_to_observation_id: dict[str, str] = {}

        # 1. Load active PHs and predict forward.
        active_phs = await self._ph_repo.list_open()
        # Cache for ReidNeedPolicy (InferenceStage proximity check next frame).
        self._last_open_phs = list(active_phs)
        stored_states: list[KalmanState] = []
        predicted_states: list[KalmanState] = []
        predicted_by_ph: dict[str, KalmanState] = {}
        for ph in active_phs:
            ks = KalmanState(
                mean=np.array(ph.state_mean, dtype=np.float64),
                covariance=np.array(ph.state_cov, dtype=np.float64).reshape(4, 4),
                updated_at=ph.last_seen_at,
            )
            stored_states.append(ks)
            pred_ks = predict(
                ks,
                now,
                cfg.process_noise_accel_m_s2,
                cfg.velocity_decay_s,
                damped=cfg.damped_velocity_prediction,
            )
            predicted_states.append(pred_ks)
            predicted_by_ph[ph.ph_id] = pred_ks

        # 2. Build observation lists; run cross-camera dedup before association.
        raw_observations = observations  # keep original for cluster expansion
        dedup_stats = DedupMergeStats()
        observations, cluster_map = dedup_observations(
            raw_observations, cfg, overlap_groups=self._overlap_groups or None, stats=dedup_stats
        )

        # Emit dedup metrics (primary pass only; the low-band pass is not an agreement sample).
        m = _metrics.metrics
        for distance_m in dedup_stats.floor_pair_distances_m:
            m.worldtracker_dedup_pair_distance_m.observe(distance_m)
        if dedup_stats.group_appearance_pairs:
            m.worldtracker_group_appearance_dedup_total.inc(dedup_stats.group_appearance_pairs)
        for _rep_id, src_ids in cluster_map.items():
            if len(src_ids) > 1:
                m.worldtracker_observations_deduped_total.inc(len(src_ids) - 1)
                m.worldtracker_dedup_clusters_total.inc()
        # Count missing-floor-point observations excluded from quality weighting.
        for obs in raw_observations:
            if not obs.floor_point.calibrated:
                m.worldtracker_observation_missing_floorpoint_total.inc()

        obs_vecs = _unpack_observations(observations)
        ph_vecs = _unpack_ph_vectors(active_phs)

        # resolve each observation's verified-ReID identity from the
        # governed (operator_verified) gallery so association can charge a
        # disagreement cost (cost path). Gated on the flag: while it is
        # off this adds zero per-frame gallery queries and the input stays
        # None, leaving association behaviour unchanged.
        obs_verified_reid_ids: list[str | None] | None = None
        if cfg.enable_reid_disagreement_cost:
            obs_verified_reid_ids = await self._resolve_verified_reid_identities(observations, now)

        # 3. Associate.
        assignment = associate(
            ph_states=predicted_states,
            ph_gallery_means=ph_vecs.gallery_means,
            ph_identity_ids=ph_vecs.identity_ids,
            ph_heights=ph_vecs.heights,
            obs_floor_points=obs_vecs.floor_points,
            obs_embeddings=obs_vecs.embeddings,
            obs_face_person_ids=obs_vecs.face_person_ids,
            obs_face_confidences=obs_vecs.face_confidences,
            obs_height_estimates=obs_vecs.heights,
            cfg=cfg,
            obs_calibrated=obs_vecs.calibrated,
            ph_view_prototypes=ph_vecs.view_prototypes,
            obs_covs=obs_vecs.covs,
            obs_verified_reid_identity_ids=obs_verified_reid_ids,
        )

        # record association integrity diagnostics from the PRIMARY pass
        # only. The shadow and low-band passes below must not touch these
        # counters or they would double/triple-count the same frame.
        for _reason, _count in assignment.rejection_reasons.items():
            m.worldtracker_association_rejections_total.labels(reason=_reason).inc(_count)
        m.worldtracker_association_outcome_total.labels(outcome="matched").inc(
            len(assignment.matched)
        )
        m.worldtracker_association_outcome_total.labels(outcome="unmatched_obs").inc(
            len(assignment.unmatched_obs)
        )
        m.worldtracker_association_outcome_total.labels(outcome="unmatched_ph").inc(
            len(assignment.unmatched_phs)
        )
        # Per-camera batch-skew histogram over the raw (pre-dedup) observations.
        # Diagnostic only — it does not change ordering or the single-batch now.
        for _raw in raw_observations:
            _skew_ms = (now - _raw.captured_at).total_seconds() * 1000.0
            if _skew_ms >= 0.0:
                m.worldtracker_batch_skew_ms.labels(camera_id=_raw.camera_id).observe(_skew_ms)

        # Shadow association under relaxed uncalibrated gate.
        if not cfg.enable_uncalibrated_gate_relax and any(not c for c in obs_vecs.calibrated):
            relaxed_cfg = replace(cfg, enable_uncalibrated_gate_relax=True)
            shadow_assignment = associate(
                ph_states=predicted_states,
                ph_gallery_means=ph_vecs.gallery_means,
                ph_identity_ids=ph_vecs.identity_ids,
                ph_heights=ph_vecs.heights,
                obs_floor_points=obs_vecs.floor_points,
                obs_embeddings=obs_vecs.embeddings,
                obs_face_person_ids=obs_vecs.face_person_ids,
                obs_face_confidences=obs_vecs.face_confidences,
                obs_height_estimates=obs_vecs.heights,
                cfg=relaxed_cfg,
                obs_calibrated=obs_vecs.calibrated,
                ph_view_prototypes=ph_vecs.view_prototypes,
                obs_covs=obs_vecs.covs,
            )
            if set(assignment.matched) != set(shadow_assignment.matched) or set(
                assignment.unmatched_obs
            ) != set(shadow_assignment.unmatched_obs):
                _metrics.metrics.world_tracker_shadow_assoc_mismatch_total.inc()

        updated_phs: list[PersonHypothesis] = []
        # Track per-PH observation metadata for snapshot building.
        ph_obs_meta: dict[
            str, tuple[int, BoundingBox | None, float]
        ] = {}  # ph_id -> (frame_index, bbox, detection_confidence)
        ph_obs_appearance: dict[
            str, tuple[OrientationBin, list[float], float]
        ] = {}  # ph_id -> (orientation, embedding, quality)
        ph_floor_calibrated: dict[str, bool] = {}  # ph_id -> obs.floor_point.calibrated this frame
        ph_confidence_meta: dict[
            str, tuple[int, bool]
        ] = {}  # ph_id -> (contributing_camera_count, footpoint_reliable)

        # Build a lookup from detection_id to the original (pre-dedup) observation.
        obs_by_det_id = {obs.detection_id: obs for obs in raw_observations if obs.detection_id}

        # 4. Update matched PHs.
        for ph_idx, obs_idx in assignment.matched:
            ph = active_phs[ph_idx]
            obs = observations[obs_idx]  # deduped representative
            src_ids = cluster_map.get(obs.detection_id, (obs.detection_id,))
            cluster_source_obs = [obs_by_det_id[did] for did in src_ids if did in obs_by_det_id]
            cluster_source_obs.sort(key=lambda o: (o.captured_at, o.camera_id, o.detection_id))
            distinct_capture_times = {o.captured_at for o in cluster_source_obs}

            cur_state = stored_states[ph_idx]
            cur_last_seen_at = ph.last_seen_at
            if len(cluster_source_obs) > 1 and len(distinct_capture_times) > 1:
                # Multi-camera cluster with distinct capture times: predict and update
                # sequentially at each observation's own capture time.
                for src_obs in cluster_source_obs:
                    finalized_src = _finalize_singleton(src_obs, cfg.k_cal)
                    if finalized_src.captured_at < cur_last_seen_at:
                        m.worldtracker_out_of_order_observations_total.labels(
                            camera_id=finalized_src.camera_id
                        ).inc()
                    pred_at_obs = predict(
                        cur_state,
                        finalized_src.captured_at,
                        cfg.process_noise_accel_m_s2,
                        cfg.velocity_decay_s,
                        damped=cfg.damped_velocity_prediction,
                    )
                    src_r = (
                        tuple_to_cov2x2(finalized_src.floor_cov_random)
                        if finalized_src.floor_cov_random is not None
                        and finalized_src.floor_point.calibrated
                        else isotropic_cov(cfg.observation_noise_m)
                    )
                    fx_src = finalized_src.floor_point.x_mm / 1000.0
                    fy_src = finalized_src.floor_point.y_mm / 1000.0
                    if fusion_applies and finalized_src.floor_point.calibrated:
                        camera_measurements.setdefault(ph.ph_id, []).append(
                            (finalized_src.captured_at, fx_src, fy_src, src_r)
                        )
                    upd_state = update(pred_at_obs, fx_src, fy_src, src_r)
                    cur_state = self._maybe_apply_zero_velocity_update(
                        ph_id=ph.ph_id,
                        predicted_state=pred_at_obs,
                        updated_state=upd_state,
                        observation_x_m=fx_src,
                        observation_y_m=fy_src,
                        observation_cov_m2=src_r,
                    )
                    if finalized_src.captured_at > cur_last_seen_at:
                        cur_last_seen_at = finalized_src.captured_at
                new_state = cur_state
            else:
                # Singleton or simultaneous multi-camera cluster: predict from the PH's
                # stored state at last_seen_at to obs.captured_at, then apply one update.
                for src_obs in cluster_source_obs or [obs]:
                    if src_obs.captured_at < cur_last_seen_at:
                        m.worldtracker_out_of_order_observations_total.labels(
                            camera_id=src_obs.camera_id
                        ).inc()
                pred_at_obs = predict(
                    cur_state,
                    obs.captured_at,
                    cfg.process_noise_accel_m_s2,
                    cfg.velocity_decay_s,
                    damped=cfg.damped_velocity_prediction,
                )
                obs_r = (
                    tuple_to_cov2x2(obs.floor_cov_random)
                    if obs.floor_cov_random is not None and obs.floor_point.calibrated
                    else isotropic_cov(cfg.observation_noise_m)
                )
                fx_obs = obs.floor_point.x_mm / 1000.0
                fy_obs = obs.floor_point.y_mm / 1000.0
                if fusion_applies and obs.floor_point.calibrated:
                    camera_measurements.setdefault(ph.ph_id, []).append(
                        (obs.captured_at, fx_obs, fy_obs, obs_r)
                    )
                new_state = update(pred_at_obs, fx_obs, fy_obs, obs_r)
                new_state = self._maybe_apply_zero_velocity_update(
                    ph_id=ph.ph_id,
                    predicted_state=pred_at_obs,
                    updated_state=new_state,
                    observation_x_m=fx_obs,
                    observation_y_m=fy_obs,
                    observation_cov_m2=obs_r,
                )
                if obs.captured_at > cur_last_seen_at:
                    cur_last_seen_at = obs.captured_at
            # contamination guard: a geometrically valid match may still be
            # an appearance outlier. When it is, the Kalman state and
            # observation_count still advance (the person was there) but
            # gallery_mean / view prototypes / mean_quality are left untouched,
            # and the embedding is never labelled with the PH identity.
            appearance_decision = evaluate_appearance_update(
                embedding=obs.embedding,
                orientation=obs.orientation,
                orientation_confidence=obs.orientation_confidence,
                quality=obs.quality,
                existing_prototypes=ph.view_prototypes,
                cfg=cfg,
            )
            if appearance_decision.accept:
                new_gallery_mean = update_gallery_mean(
                    ph.gallery_mean, obs.embedding, ph.observation_count
                )
                new_prototypes = update_view_prototypes(
                    ph.view_prototypes,
                    obs.orientation,
                    obs.embedding,
                    obs.orientation_confidence,
                )
                new_mean_quality = 0.1 * obs.quality + 0.9 * ph.mean_quality
            else:
                new_gallery_mean = ph.gallery_mean
                new_prototypes = ph.view_prototypes
                new_mean_quality = ph.mean_quality
                _record_appearance_rejection(m, ph.ph_id, obs, appearance_decision)
            new_height, new_height_sigma, _height_accepted = update_ph_height_estimate(
                ph.height_estimate_m,
                ph.height_sigma_m,
                obs.height_estimate_m,
                obs.height_sigma_m,
            )
            room_id, room_name = resolve_room(
                obs.floor_point.x_mm / 1000.0,
                obs.floor_point.y_mm / 1000.0,
                obs.camera_id,
                room_polygons,
                camera_room_map,
                room_names,
            )

            # Expand active_cameras to include all cameras from the dedup cluster.
            cluster_cameras = frozenset(
                obs_by_det_id[did].camera_id for did in src_ids if did in obs_by_det_id
            )
            self._update_primary_camera(ph.ph_id, _best_primary_camera(src_ids, obs_by_det_id, obs))

            updated = replace(
                ph,
                state_mean=(
                    float(new_state.mean[0]),
                    float(new_state.mean[1]),
                    float(new_state.mean[2]),
                    float(new_state.mean[3]),
                ),
                state_cov=tuple(float(v) for v in new_state.covariance.flatten()),
                last_seen_at=cur_last_seen_at,
                last_seen_camera=obs.camera_id,
                observation_count=ph.observation_count + 1,
                gallery_mean=new_gallery_mean,
                height_estimate_m=new_height,
                height_sigma_m=new_height_sigma,
                active_cameras=ph.active_cameras | cluster_cameras,
                last_floor_speed_m_s=speed_m_s(
                    (
                        float(new_state.mean[0]),
                        float(new_state.mean[1]),
                        float(new_state.mean[2]),
                        float(new_state.mean[3]),
                    )
                ),
                mean_quality=new_mean_quality,
                view_prototypes=new_prototypes,
                metadata=_metadata_with_room(ph.metadata, room_id, room_name),
            )
            updated_phs.append(updated)
            ph_obs_meta[ph.ph_id] = (
                obs.frame_index,
                obs.bbox,
                obs.detection_confidence,
            )
            ph_obs_appearance[ph.ph_id] = (obs.orientation, obs.embedding, obs.quality)
            ph_floor_calibrated[ph.ph_id] = obs.floor_point.calibrated
            ph_confidence_meta[ph.ph_id] = (len(src_ids), obs.footpoint_reliable)
            # Map all source detection IDs (not just the representative) to this PH.
            for src_det_id in src_ids:
                if src_det_id:
                    det_to_ph[src_det_id] = ph.ph_id

            # Persist ALL source observations in capture-time order so that
            # every camera's raw row lands on the PH.
            for src_obs in cluster_source_obs or [obs]:
                saved_obs_id = await self._obs_repo.save(src_obs, ph_id=ph.ph_id)
                if src_obs.detection_id:
                    det_to_observation_id[src_obs.detection_id] = saved_obs_id

        # 5. Spawn new PHs for unmatched observations (or revive recently-closed).
        # Hoist shared queries outside the per-observation loop to avoid N DB
        # round-trips for N unmatched observations in one frame.
        revived_ph_ids: set[str] = set()
        _revival_lookback = now - timedelta(seconds=cfg.revive_max_age_s)
        _recent_closed: list[PersonHypothesis] = (
            await self._ph_repo.list_closed_since(_revival_lookback, limit=100)
            if assignment.unmatched_obs
            else []
        )
        _topology_edges: list[CameraTopologyEdge] = []
        if assignment.unmatched_obs and self._topology_repo is not None:
            try:
                _topology_edges = await self._topology_repo.list_edges()
            except Exception:
                logger.warning("topology_edges_load_failed", exc_info=True)
        for obs_idx in assignment.unmatched_obs:
            obs = observations[obs_idx]
            fx = obs.floor_point.x_mm / 1000.0
            fy = obs.floor_point.y_mm / 1000.0

            out_of_room = bool(room_polygons) and not is_in_any_room_polygon(fx, fy, room_polygons)
            if obs.floor_point.calibrated and out_of_room:
                _metrics.metrics.world_tracker_spawn_rejected_out_of_room_total.inc()
                continue
            if not obs.floor_point.calibrated and out_of_room:
                _metrics.metrics.identity_shadow_mismatch_total.labels(
                    feature="uncalibrated_spawn"
                ).inc()
                logger.debug(
                    "world_tracker_uncalibrated_spawn_allowed",
                    camera_id=obs.camera_id,
                    calibrated=False,
                )

            # Attempt PH revival before spawning new (same-camera + cross-camera).
            revival_candidate: PersonHypothesis | None = None
            available_closed = [c for c in _recent_closed if c.ph_id not in revived_ph_ids]

            if cfg.enable_ph_revival:
                revival_candidate = select_revival_candidate(
                    obs,
                    available_closed,
                    obs.captured_at,
                    cfg,
                    enable_cross_camera=cfg.enable_cross_camera_revival,
                    topology_edges=_topology_edges,
                )
            else:
                # Shadow: compute revival candidate to measure what it would catch.
                shadow_candidate = select_revival_candidate(
                    obs,
                    available_closed,
                    obs.captured_at,
                    cfg,
                    enable_cross_camera=False,
                    topology_edges=_topology_edges,
                )
                if shadow_candidate is not None:
                    _metrics.metrics.world_tracker_shadow_revival_total.inc()

                # Shadow cross-camera separately to measure its marginal benefit.
                if _topology_edges:
                    shadow_cc = select_revival_candidate(
                        obs,
                        available_closed,
                        obs.captured_at,
                        cfg,
                        enable_cross_camera=True,
                        topology_edges=_topology_edges,
                    )
                    if shadow_cc is not None and shadow_cc.last_seen_camera != obs.camera_id:
                        _metrics.metrics.world_tracker_shadow_cross_camera_revival_total.inc()

            # Include all cameras from the dedup cluster when spawning/reviving.
            spawn_src_ids = cluster_map.get(obs.detection_id, (obs.detection_id,))
            spawn_cameras = frozenset(
                obs_by_det_id[did].camera_id for did in spawn_src_ids if did in obs_by_det_id
            ) or frozenset([obs.camera_id])
            best_spawn_camera = _best_primary_camera(spawn_src_ids, obs_by_det_id, obs)
            room_id, room_name = resolve_room(
                obs.floor_point.x_mm / 1000.0,
                obs.floor_point.y_mm / 1000.0,
                obs.camera_id,
                room_polygons,
                camera_room_map,
                room_names,
            )

            if revival_candidate is not None:
                # Revive: reuse closed PH's ph_id, identity, and gallery state.
                closed = revival_candidate
                ks = initialize(
                    fx,
                    fy,
                    cfg.initial_position_sigma_m,
                    cfg.initial_velocity_sigma_m_s,
                    obs.captured_at,
                )
                # Reviving an existing PH's appearance is an association too; the
                # same contamination guard applies. A rejected embedding reopens
                # the PH and advances its count without touching appearance.
                revival_decision = evaluate_appearance_update(
                    embedding=obs.embedding,
                    orientation=obs.orientation,
                    orientation_confidence=obs.orientation_confidence,
                    quality=obs.quality,
                    existing_prototypes=closed.view_prototypes,
                    cfg=cfg,
                )
                if revival_decision.accept:
                    new_gallery_mean = update_gallery_mean(
                        closed.gallery_mean, obs.embedding, closed.observation_count
                    )
                    new_prototypes_rev = update_view_prototypes(
                        closed.view_prototypes,
                        obs.orientation,
                        obs.embedding,
                        obs.orientation_confidence,
                    )
                else:
                    new_gallery_mean = closed.gallery_mean
                    new_prototypes_rev = closed.view_prototypes
                    _record_appearance_rejection(m, closed.ph_id, obs, revival_decision)
                new_ph = replace(
                    closed,
                    state_mean=(
                        float(ks.mean[0]),
                        float(ks.mean[1]),
                        float(ks.mean[2]),
                        float(ks.mean[3]),
                    ),
                    state_cov=tuple(float(v) for v in ks.covariance.flatten()),
                    last_seen_at=max(closed.last_seen_at, obs.captured_at),
                    last_seen_camera=obs.camera_id,
                    observation_count=closed.observation_count + 1,
                    gallery_mean=new_gallery_mean,
                    active_cameras=closed.active_cameras | spawn_cameras,
                    closed_at=None,  # reopen
                    last_floor_speed_m_s=0.0,
                    metadata=_metadata_with_room(closed.metadata, room_id, room_name),
                    view_prototypes=new_prototypes_rev,
                )
                revived_ph_ids.add(closed.ph_id)
                _metrics.metrics.cts_ph_revived_total.inc()
                _metrics.metrics.world_tracker_ph_spawned_total.labels(reason="revived").inc()
                logger.info(
                    "ph_revived",
                    ph_id=closed.ph_id,
                    camera_id=obs.camera_id,
                    prev_identity=closed.current_identity_id,
                    age_s=(
                        obs.captured_at - (closed.closed_at or closed.last_seen_at)
                    ).total_seconds(),
                )

                # Record cross-camera handoff in the topology model.
                if (
                    closed.last_seen_camera != obs.camera_id
                    and self._topology_repo is not None
                    and closed.closed_at is not None
                ):
                    elapsed_s = (obs.captured_at - closed.closed_at).total_seconds()
                    updated_edge = record_handoff(
                        closed.last_seen_camera,
                        obs.camera_id,
                        elapsed_s,
                        _topology_edges,
                        now,
                    )
                    await self._topology_repo.upsert_edge(updated_edge)
                    logger.debug(
                        "topology_handoff_recorded",
                        from_camera=closed.last_seen_camera,
                        to_camera=obs.camera_id,
                        elapsed_s=round(elapsed_s, 2),
                    )
            else:
                # No revival candidate: spawn a brand-new PH.
                ks = initialize(
                    fx,
                    fy,
                    cfg.initial_position_sigma_m,
                    cfg.initial_velocity_sigma_m_s,
                    obs.captured_at,
                )
                # seed first view prototype from spawn observation.
                spawn_prototypes = update_view_prototypes(
                    (),
                    obs.orientation,
                    obs.embedding,
                    obs.orientation_confidence,
                )
                new_ph = PersonHypothesis(
                    ph_id=str(uuid.uuid4()),
                    state_mean=(
                        float(ks.mean[0]),
                        float(ks.mean[1]),
                        float(ks.mean[2]),
                        float(ks.mean[3]),
                    ),
                    state_cov=tuple(float(v) for v in ks.covariance.flatten()),
                    born_at=obs.captured_at,
                    last_seen_at=obs.captured_at,
                    last_seen_camera=obs.camera_id,
                    observation_count=1,
                    # Born Unknown: the resolver runs in this same step with this round's faces
                    # and is the only writer of identity, so a first face is weighed like any
                    # other instead of becoming an uncommitted prior.
                    current_identity_id=None,
                    gallery_mean=obs.embedding if obs.embedding else None,
                    height_estimate_m=obs.height_estimate_m,
                    height_sigma_m=obs.height_sigma_m,
                    active_cameras=spawn_cameras,
                    last_floor_speed_m_s=0.0,
                    metadata=_metadata_with_room({}, room_id, room_name),
                    mean_quality=obs.quality,
                    view_prototypes=spawn_prototypes,
                )

            updated_phs.append(new_ph)
            self._update_primary_camera(new_ph.ph_id, best_spawn_camera)
            ph_obs_meta[new_ph.ph_id] = (
                obs.frame_index,
                obs.bbox,
                obs.detection_confidence,
            )
            ph_obs_appearance[new_ph.ph_id] = (obs.orientation, obs.embedding, obs.quality)
            ph_floor_calibrated[new_ph.ph_id] = obs.floor_point.calibrated
            if fusion_applies and obs.floor_point.calibrated:
                camera_measurements[new_ph.ph_id] = [
                    (obs.captured_at, fx, fy, _camera_measurement_cov(obs, cfg))
                ]
            ph_confidence_meta[new_ph.ph_id] = (len(spawn_src_ids), obs.footpoint_reliable)
            # Save the PH first so the FK constraint on world_observations is satisfied.
            await self._ph_repo.save(new_ph)
            # Persist all source observations for this PH.
            for src_det_id in spawn_src_ids:
                src_obs = obs_by_det_id.get(src_det_id, obs)
                saved_obs_id = await self._obs_repo.save(src_obs, ph_id=new_ph.ph_id)
                if src_det_id:
                    det_to_observation_id[src_det_id] = saved_obs_id
            for src_det_id in spawn_src_ids:
                if src_det_id:
                    det_to_ph[src_det_id] = new_ph.ph_id

            # 6. Check for PH continuations from recently closed PHs.
            had_continuation = False
            if self._continuation_publisher is not None:
                continuation_lookback = obs.captured_at - timedelta(
                    seconds=cfg.inferred_handoff_max_s
                )
                recent_closed_cont = await self._ph_repo.list_closed_since(
                    continuation_lookback, limit=100
                )
                for closed in recent_closed_cont:
                    if closed.ph_id == new_ph.ph_id:
                        continue
                    if closed.closed_at is None:
                        continue
                    elapsed = (obs.captured_at - closed.closed_at).total_seconds()
                    if elapsed <= 0 or elapsed > cfg.inferred_handoff_max_s:
                        continue
                    # Distance between new spawn point and closed PH's last position.
                    cx, cy = closed.state_mean[0], closed.state_mean[1]
                    dist = float(np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2))
                    if dist > cfg.inferred_handoff_max_distance_m:
                        continue
                    predicted_drift = closed.last_floor_speed_m_s * elapsed
                    candidate = PHContinuationCandidate(
                        predecessor_ph_id=closed.ph_id,
                        successor_ph_id=new_ph.ph_id,
                        predecessor_closed_at=closed.closed_at,
                        successor_born_at=obs.captured_at,
                        distance_m=dist,
                        seconds_elapsed=elapsed,
                        predicted_drift_m=predicted_drift,
                        predecessor_identity_id=closed.current_identity_id,
                    )
                    continuations.append(candidate)
                    await self._continuation_publisher.publish(candidate)
                    _metrics.metrics.world_tracker_continuations_total.inc()
                    had_continuation = True

                    # Record cross-camera handoff in topology model.
                    if closed.last_seen_camera != obs.camera_id and self._topology_repo is not None:
                        updated_edge = record_handoff(
                            closed.last_seen_camera,
                            obs.camera_id,
                            elapsed,
                            _topology_edges,
                            now,
                        )
                        await self._topology_repo.upsert_edge(updated_edge)

            if revival_candidate is not None:
                pass  # Already counted as "revived" above.
            elif had_continuation:
                _metrics.metrics.world_tracker_ph_spawned_total.labels(
                    reason="respawn_after_close"
                ).inc()
            else:
                _metrics.metrics.world_tracker_ph_spawned_total.labels(
                    reason="new_observation"
                ).inc()

        # 5a. Low-confidence second association pass (BYTE-style recovery).
        # Offers low-band detections to unmatched PHs under a tightened geometric gate.
        # Geometry-only cost (alpha_app=0): no embeddings, no identity evidence.
        # Matched PHs get Kalman state + last_seen_at update only; observation_count,
        # gallery_mean, and view_prototypes are intentionally unchanged to prevent
        # ghost persistence from contaminating identity or appearance state.
        # Low-band observations are NOT persisted (WorldObservation has no
        # low_confidence field; adding a DB column is out of scope here).
        lb_matched_ph_indices: set[int] = set()
        if cfg.enable_low_confidence_recovery and low_band_observations:
            lb_obs_raw, _ = dedup_observations(
                low_band_observations, cfg, overlap_groups=self._overlap_groups or None
            )
            # Build the subset of still-unmatched-after-pass-1 PH indices.
            unmatched_after_pass1 = assignment.unmatched_phs
            if unmatched_after_pass1 and lb_obs_raw:
                lb_obs_vecs = _unpack_observations(lb_obs_raw)
                lb_ph_states = [predicted_states[i] for i in unmatched_after_pass1]
                lb_ph_gallery = [ph_vecs.gallery_means[i] for i in unmatched_after_pass1]
                lb_ph_identities = [ph_vecs.identity_ids[i] for i in unmatched_after_pass1]
                lb_ph_heights = [ph_vecs.heights[i] for i in unmatched_after_pass1]
                lb_ph_prototypes = [ph_vecs.view_prototypes[i] for i in unmatched_after_pass1]
                # Geometry-only cost: set alpha_app=0 and tighten the gate.
                recovery_cfg = replace(cfg, gate_chi2=cfg.recovery_gate_chi2, alpha_app=0.0)
                lb_assignment = associate(
                    ph_states=lb_ph_states,
                    ph_gallery_means=lb_ph_gallery,
                    ph_identity_ids=lb_ph_identities,
                    ph_heights=lb_ph_heights,
                    obs_floor_points=lb_obs_vecs.floor_points,
                    obs_embeddings=[None] * len(lb_obs_raw),
                    obs_face_person_ids=[None] * len(lb_obs_raw),
                    obs_face_confidences=[0.0] * len(lb_obs_raw),
                    obs_height_estimates=lb_obs_vecs.heights,
                    cfg=recovery_cfg,
                    obs_calibrated=lb_obs_vecs.calibrated,
                    ph_view_prototypes=lb_ph_prototypes,
                    obs_covs=lb_obs_vecs.covs,
                )
                for local_ph_idx, lb_obs_idx in lb_assignment.matched:
                    ph_idx = unmatched_after_pass1[local_ph_idx]
                    ph = active_phs[ph_idx]
                    obs = lb_obs_raw[lb_obs_idx]
                    if obs.captured_at < ph.last_seen_at:
                        m.worldtracker_out_of_order_observations_total.labels(
                            camera_id=obs.camera_id
                        ).inc()
                    ks = predict(
                        stored_states[ph_idx],
                        obs.captured_at,
                        cfg.process_noise_accel_m_s2,
                        cfg.velocity_decay_s,
                        damped=cfg.damped_velocity_prediction,
                    )
                    fx = obs.floor_point.x_mm / 1000.0
                    fy = obs.floor_point.y_mm / 1000.0
                    obs_r = isotropic_cov(cfg.observation_noise_m)
                    new_state = update(ks, fx, fy, obs_r)
                    new_state = self._maybe_apply_zero_velocity_update(
                        ph_id=ph.ph_id,
                        predicted_state=ks,
                        updated_state=new_state,
                        observation_x_m=fx,
                        observation_y_m=fy,
                        observation_cov_m2=obs_r,
                    )
                    # Kalman + last_seen_at update only; do NOT touch observation_count,
                    # gallery_mean, view_prototypes, or mean_quality.
                    recovered = replace(
                        ph,
                        state_mean=(
                            float(new_state.mean[0]),
                            float(new_state.mean[1]),
                            float(new_state.mean[2]),
                            float(new_state.mean[3]),
                        ),
                        state_cov=tuple(float(v) for v in new_state.covariance.flatten()),
                        last_seen_at=max(ph.last_seen_at, obs.captured_at),
                        last_seen_camera=obs.camera_id,
                        active_cameras=ph.active_cameras | frozenset([obs.camera_id]),
                        last_floor_speed_m_s=speed_m_s(
                            (
                                float(new_state.mean[0]),
                                float(new_state.mean[1]),
                                float(new_state.mean[2]),
                                float(new_state.mean[3]),
                            )
                        ),
                    )
                    updated_phs.append(recovered)
                    lb_matched_ph_indices.add(ph_idx)
                    ph_confidence_meta[ph.ph_id] = (1, obs.footpoint_reliable)
                    _metrics.metrics.worldtracker_low_band_matches_total.inc()
                    logger.debug(
                        "low_band_recovery_match",
                        ph_id=ph.ph_id,
                        camera_id=obs.camera_id,
                        confidence=round(obs.detection_confidence, 3),
                    )
                dropped_lb = len(lb_assignment.unmatched_obs)
                if dropped_lb:
                    _metrics.metrics.worldtracker_low_band_dropped_total.inc(dropped_lb)

        # 7. Close PHs that have not been observed for ph_close_grace_s.
        # PHs matched in the low-band second pass (5a) already have their state
        # advanced and last_seen_at refreshed; skip them here so they are not
        # double-counted or prematurely closed.
        if fusion_on:
            for ph in updated_phs:
                self._camera_seen_at[ph.ph_id] = ph.last_seen_at
        bridge_on = wcfg.spatial_fusion_mode == "enabled" and wcfg.coasting_bridge_max_s > 0.0
        coasting_ph_ids: set[str] = set()
        for ph_idx in assignment.unmatched_phs:
            if ph_idx in lb_matched_ph_indices:
                continue
            ph = active_phs[ph_idx]
            advanced_ph = _advance_unmatched_ph(
                ph,
                predicted_states[ph_idx],
                now,
                cfg.ph_close_grace_s,
                camera_seen_at=self._camera_seen_at.get(ph.ph_id) if bridge_on else None,
                bridge_max_s=wcfg.coasting_bridge_max_s,
                bridge_fix_slack_s=wcfg.spatial_fusion_max_dt_s,
            )
            if advanced_ph.closed_at is None:
                coasting_ph_ids.add(advanced_ph.ph_id)
            updated_phs.append(advanced_ph)

        # 8. Persist all updated PHs.
        for ph in updated_phs:
            await self._ph_repo.save(ph)

        # 8a. Update calibrated PH floor history (only rounds updated by calibrated
        #     observations; never virtual-cell points or unobserved predictions) and
        #     run wearable-to-PH association when wearable mode is shadow or enabled.
        for ph in updated_phs:
            if ph.closed_at is not None or not ph_floor_calibrated.get(ph.ph_id, False):
                continue
            if not fusion_applies:
                self._ph_floor_history.record(
                    ph.ph_id,
                    captured_at=ph.last_seen_at,
                    x_m=ph.state_mean[0],
                    y_m=ph.state_mean[1],
                    cov_m2=(ph.state_cov[0], ph.state_cov[1], ph.state_cov[4], ph.state_cov[5]),
                    speed_m_s=ph.last_floor_speed_m_s,
                )
                continue
            # The speed still comes from the posterior: it only widens the timing term of
            # the association innovation, so it cannot make a fix confirm itself.
            for captured_at, x_m, y_m, cov in camera_measurements.get(ph.ph_id, ()):
                self._ph_floor_history.record(
                    ph.ph_id,
                    captured_at=captured_at,
                    x_m=x_m,
                    y_m=y_m,
                    cov_m2=(float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 0]), float(cov[1, 1])),
                    speed_m_s=ph.last_floor_speed_m_s,
                )
        self._ph_floor_history.prune(now)

        wearable_evidence_by_ph: dict[str, WearableEvidence] = {}
        wearable_transitions: list[WearableAssociationTransition] = []
        if self._wearable_tracker is not None:
            open_phs_now = [ph for ph in updated_phs if ph.closed_at is None]
            closed_ph_ids_now = [ph.ph_id for ph in updated_phs if ph.closed_at is not None]
            wearable_round = self._wearable_tracker.step(
                fixes=wearable_fixes or (),
                open_phs=open_phs_now,
                closed_ph_ids=closed_ph_ids_now,
                ph_history=self._ph_floor_history,
                room_polygons=room_polygons,
                camera_room_ids=camera_room_ids,
                now=now,
            )
            wearable_evidence_by_ph = dict(wearable_round.evidence_by_ph)
            wearable_transitions.extend(wearable_round.transitions)
            if fusion_on:
                # Before identity resolution: the resolver writes identity through targeted
                # repository operations, so a later full save would overwrite them.
                updated_phs = await self._fuse_wearable_positions(
                    updated_phs,
                    evidence_by_ph=wearable_evidence_by_ph,
                    fixes=wearable_fixes or (),
                    coasting_ph_ids=coasting_ph_ids,
                    now=now,
                )

        # 9. Identity resolution: run the Bayesian resolver on PHs that
        #    received observations this frame.  Pass the full open-PH identity
        #    occupancy (updated_phs is a superset of all open PHs) so the
        #    duplicate-active-identity guard also protects incumbents that were
        #    not observed this frame, and closed-world joint elimination can
        #    verify simultaneous observation and spatial separation.
        open_phs_all = [ph for ph in updated_phs if ph.closed_at is None]
        open_ph_identities = {
            ph.ph_id: ph.current_identity_id for ph in open_phs_all if ph.current_identity_id
        }
        open_ph_occupancies: dict[str, OpenPHOccupancy] = {}
        for ph in open_phs_all:
            is_coasting = ph.ph_id in coasting_ph_ids
            primary_cam = self._primary_camera.get(ph.ph_id, ph.last_seen_camera)
            occ_room_id, _ = resolve_room(
                ph.state_mean[0],
                ph.state_mean[1],
                primary_cam,
                room_polygons,
                camera_room_map,
                room_names,
            )
            if ph.ph_id in self._room_wearable_only_phs:
                elim_ev_at = self._last_elim_eligible_evidence_at.get(ph.ph_id)
                elim_eligible = elim_ev_at is not None
            else:
                elim_ev_at = ph.last_independent_identity_evidence_at
                elim_eligible = True
            open_ph_occupancies[ph.ph_id] = OpenPHOccupancy(
                ph_id=ph.ph_id,
                identity_id=ph.current_identity_id,
                last_independent_evidence_at=elim_ev_at,
                last_seen_at=ph.last_seen_at,
                coasting=is_coasting,
                floor_calibrated=bool(ph_floor_calibrated.get(ph.ph_id, False)),
                floor_x_m=ph.state_mean[0],
                floor_y_m=ph.state_mean[1],
                room_id=occ_room_id,
                last_seen_camera=primary_cam,
                active_cameras=ph.active_cameras,
                elim_eligible=elim_eligible,
            )

        copresence_pairs: set[tuple[str, str]] = set()
        if (
            self._closed_world_config.mode != "off"
            and self._copresence_repo is not None
            and len(open_phs_all) >= 2
        ):
            open_id_list = [ph.ph_id for ph in open_phs_all]
            for idx_a, ph_a_id in enumerate(open_id_list):
                for ph_b_id in open_id_list[idx_a + 1 :]:
                    link = await self._copresence_repo.get_active_link(ph_a_id, ph_b_id)
                    if link is not None:
                        copresence_pairs.add(canonical_ph_pair(ph_a_id, ph_b_id))

        wearable_present_identities = (
            self._wearable_tracker.members_with_fresh_non_parked_fix(
                now=now,
                window_s=self._closed_world_config.wearable_home_window_s,
            )
            if self._wearable_tracker is not None
            else frozenset()
        )
        ph_appearance_features: dict[str, AppearanceFeature] = {}
        if self._appearance_feature_extractor is not None:
            for ph in open_phs_all:
                app_obs = ph_obs_appearance.get(ph.ph_id)
                if app_obs is None:
                    continue
                obs_orient, obs_emb, obs_qual = app_obs
                feature = self._appearance_feature_extractor.extract(
                    view_prototypes=ph.view_prototypes,
                    current_orientation=obs_orient,
                    observation_embedding=obs_emb,
                    gallery_mean=ph.gallery_mean,
                    quality=obs_qual,
                )
                if feature is not None:
                    ph_appearance_features[ph.ph_id] = feature

        identity_decisions, revisions, identity_by_ph, image_only = await _resolve_identities(
            resolver=self._identity_resolver,
            obs_repo=self._obs_repo,
            ph_repo=self._ph_repo,
            phs=updated_phs,
            ph_obs_meta=ph_obs_meta,
            face_anchors=face_anchors or [],
            det_to_ph=det_to_ph,
            face_evidence=face_evidence,
            open_ph_identities=open_ph_identities,
            wearable_evidence=wearable_evidence_by_ph or None,
            open_ph_occupancies=open_ph_occupancies,
            copresence_pairs=frozenset(copresence_pairs),
            overlap_groups=tuple(self._overlap_groups),
            camera_room_map=camera_room_map,
            wearable_present_identities=wearable_present_identities,
            ph_appearance_features=ph_appearance_features or None,
            now=now,
            config=cfg,
        )

        # Update elimination-eligibility tracking from this round's decisions so
        # room-class wearable evidence can never feed closed-world joint elimination.
        for decision in identity_decisions:
            if decision.identity_id is None:
                self._room_wearable_only_phs.discard(decision.ph_id)
                self._last_elim_eligible_evidence_at.pop(decision.ph_id, None)
            elif decision.evidence_backed:
                room_only = bool(
                    isinstance(decision.evidence, dict)
                    and decision.evidence.get("room_wearable_only", False)
                )
                if room_only:
                    self._room_wearable_only_phs.add(decision.ph_id)
                else:
                    self._room_wearable_only_phs.discard(decision.ph_id)
                    self._last_elim_eligible_evidence_at[decision.ph_id] = (
                        decision.evidence_at or now
                    )

        # Notify wearable attestation tracker of any calibrated-face vs. wearable
        # conflicts so attestation breaks and repeated conflicts flag swap_suspected.
        if self._wearable_tracker is not None:
            for decision in identity_decisions:
                if decision.conflict == "wearable_face":
                    wev = wearable_evidence_by_ph.get(decision.ph_id)
                    if wev is not None:
                        wearable_transitions.extend(
                            self._wearable_tracker.record_face_conflict(
                                ph_id=decision.ph_id,
                                tag_id=wev.tag_id,
                                at=now,
                            )
                        )

        # Update the held-identity confidence cache: record a positive committed
        # confidence; clear it on a genuine demotion to UNKNOWN.
        for pid, data in identity_by_ph.items():
            conf = float(data.get("identity_confidence") or 0.0)  # type: ignore[arg-type]
            if data.get("identity_id") and conf > 0.0:
                self._last_identity_confidence[pid] = conf
            else:
                self._last_identity_confidence.pop(pid, None)
        # Bound the cache to PHs that are still open after this frame. Intersecting
        # every frame evicts not only PHs closed here (step 7) but also any closed,
        # merged, or deleted OUTSIDE the tracker (e.g. the PH API), which would
        # otherwise never pass through this loop and leak. updated_phs is a
        # superset of all currently-open PHs (every list_open PH is matched or
        # advanced, plus this frame's spawns/revivals), so this is a hard bound:
        # cache size <= number of open PHs.
        open_ph_ids = {ph.ph_id for ph in updated_phs if ph.closed_at is None}
        self._last_identity_confidence = {
            pid: conf for pid, conf in self._last_identity_confidence.items() if pid in open_ph_ids
        }
        self._still_counter = {
            pid: count for pid, count in self._still_counter.items() if pid in open_ph_ids
        }
        self._camera_seen_at = {
            pid: ts for pid, ts in self._camera_seen_at.items() if pid in open_ph_ids
        }
        self._fused_through = {
            pid: ts for pid, ts in self._fused_through.items() if pid in open_ph_ids
        }
        self._primary_camera = {
            pid: camera_id for pid, camera_id in self._primary_camera.items() if pid in open_ph_ids
        }
        self._primary_challenger = {
            pid: challenger
            for pid, challenger in self._primary_challenger.items()
            if pid in open_ph_ids
        }
        self._room_wearable_only_phs &= open_ph_ids
        self._last_elim_eligible_evidence_at = {
            pid: ts
            for pid, ts in self._last_elim_eligible_evidence_at.items()
            if pid in open_ph_ids
        }

        # Gallery candidate creation moved to ReIDCandidateStage: the
        # tracker has no image bytes, and creation needs crop provenance
        # (source_frame_key/crop_key/hashes) that only the pipeline stage has.
        # See app/pipeline/stages/reid_candidates.py.

        # Detect co-presence links for overlapping cameras sharing an identity.
        if self._copresence_repo is not None and self._overlap_groups:
            await self._detect_copresence(updated_phs, identity_by_ph, now)

        # 10. Build snapshots for downstream stages.
        snapshots: list[WorldFrameSnapshot] = []
        for ph in updated_phs:
            if ph.observation_count < cfg.min_observations_to_publish:
                continue
            is_coasting = ph.ph_id in coasting_ph_ids
            if is_coasting and ph.ph_id in predicted_by_ph:
                pred_ks = predicted_by_ph[ph.ph_id]
                snap_mean = (
                    float(pred_ks.mean[0]),
                    float(pred_ks.mean[1]),
                    float(pred_ks.mean[2]),
                    float(pred_ks.mean[3]),
                )
                snap_cov = tuple(float(v) for v in pred_ks.covariance.flatten())
            else:
                snap_mean = ph.state_mean
                snap_cov = ph.state_cov
            primary_camera = self._primary_camera.get(ph.ph_id, ph.last_seen_camera)
            room_id, room_name = resolve_room(
                snap_mean[0],
                snap_mean[1],
                primary_camera,
                room_polygons,
                camera_room_map,
                room_names,
            )
            obs_meta = ph_obs_meta.get(ph.ph_id, (0, None, 0.0))
            obs_frame_index, obs_bbox, obs_det_conf = obs_meta
            confidence_meta = ph_confidence_meta.get(ph.ph_id, (1, True))
            contributing_camera_count, footpoint_reliable = confidence_meta
            id_data = identity_by_ph.get(ph.ph_id, {})
            snapshots.append(
                WorldFrameSnapshot(
                    ph_id=ph.ph_id,
                    camera_id=primary_camera,
                    frame_index=obs_frame_index,
                    captured_at=ph.last_seen_at,
                    floor_x_m=snap_mean[0],
                    floor_y_m=snap_mean[1],
                    floor_vx_m_s=snap_mean[2],
                    floor_vy_m_s=snap_mean[3],
                    position_sigma_m=position_sigma_m(snap_cov),
                    contributing_camera_count=contributing_camera_count,
                    footpoint_reliable=footpoint_reliable,
                    identity_id=_sanitize_identity_id(
                        str(id_data["identity_id"])
                        if id_data.get("identity_id") is not None
                        else ph.current_identity_id
                    ),
                    # Use this frame's posterior when resolved; otherwise replay
                    # the held identity's last positive confidence (coasting /
                    # unresolved frames) instead of emitting a sentinel 0.0.
                    identity_confidence=(
                        float(id_data.get("identity_confidence", 0.0) or 0.0)  # type: ignore[arg-type]
                        or self._last_identity_confidence.get(ph.ph_id, 0.0)
                    ),
                    posterior_entropy=float(id_data.get("posterior_entropy", 0.0) or 0.0),  # type: ignore[arg-type]
                    direct_face_evidence=bool(id_data.get("direct_face_evidence", False) or False),
                    bbox=obs_bbox,
                    detection_confidence=obs_det_conf,
                    height_m=ph.height_estimate_m,
                    room_id=room_id,
                    room_name=room_name,
                    mean_quality=ph.mean_quality,
                    active_cameras=ph.active_cameras,
                    floor_speed_m_s=(
                        speed_m_s(snap_mean) if ph_floor_calibrated.get(ph.ph_id, False) else None
                    ),
                    evidence_json=str(id_data.get("evidence_json", "{}")),
                    inferred_identity_id=str(id_data.get("inferred_identity_id") or ""),
                    effective_identity_id=str(id_data.get("effective_identity_id") or ""),
                    authority=str(id_data.get("authority") or ""),
                    decision_source=str(id_data.get("decision_source") or ""),
                    decision_id=str(id_data.get("decision_id") or ""),
                    conflict=str(id_data.get("conflict") or ""),
                    last_independent_evidence_at_unix_ns=int(
                        id_data.get("last_independent_evidence_at_unix_ns") or 0  # type: ignore[call-overload]
                    ),
                    config_hash=str(id_data.get("config_hash") or ""),
                    model_set_version=str(id_data.get("model_set_version") or ""),
                    coasting=is_coasting,
                )
            )

        return WorldTrackerResult(
            updated_phs=updated_phs,
            snapshots=snapshots,
            continuations=continuations,
            identity_decisions=identity_decisions,
            revisions=revisions,
            det_to_ph=det_to_ph,
            det_to_observation_id=det_to_observation_id,
            revived_ph_ids=frozenset(revived_ph_ids),
            coasting_ph_ids=frozenset(coasting_ph_ids),
            wearable_association_transitions=tuple(wearable_transitions),
            wearable_evidence_by_ph=wearable_evidence_by_ph,
            image_only_evaluations=tuple(image_only),
        )

    def _update_primary_camera(self, ph_id: str, best_camera: str) -> None:
        if not best_camera:
            return

        current = self._primary_camera.get(ph_id)
        if current is None:
            self._primary_camera[ph_id] = best_camera
            self._primary_challenger.pop(ph_id, None)
            return

        if best_camera == current:
            self._primary_challenger.pop(ph_id, None)
            return

        challenger_camera, streak = self._primary_challenger.get(ph_id, ("", 0))
        if best_camera == challenger_camera:
            streak += 1
        else:
            streak = 1

        switch_frames = max(1, self._config.primary_switch_frames)
        if streak >= switch_frames:
            self._primary_camera[ph_id] = best_camera
            self._primary_challenger.pop(ph_id, None)
            return

        self._primary_challenger[ph_id] = (best_camera, streak)

    async def _fuse_wearable_positions(
        self,
        phs: list[PersonHypothesis],
        *,
        evidence_by_ph: Mapping[str, WearableEvidence],
        fixes: Sequence[WearableFix],
        coasting_ph_ids: set[str],
        now: datetime,
    ) -> list[PersonHypothesis]:
        """Fuse this round's trajectory-attested position fixes into PH Kalman states.

        Only an open PH whose wearable evidence this round is trajectory class with a
        trajectory grade is considered, with a fix from the tag that produced that
        evidence. A camera-tracked PH takes fixes up to its state time; a camera-occluded PH
        may be advanced by a newer fix while its last camera observation is within the
        bridge window. "shadow" records what would happen; "enabled" applies it and
        re-saves the PH. PH floor history is never touched here, so association keeps
        scoring raw camera measurements only.
        """
        wcfg = self._wearable_config
        mode = wcfg.spatial_fusion_mode
        cfg = self._config
        motion = MotionModel(
            process_noise_accel_m_s2=cfg.process_noise_accel_m_s2,
            velocity_decay_s=cfg.velocity_decay_s,
            damped=cfg.damped_velocity_prediction,
        )
        m = _metrics.metrics
        result = list(phs)
        for idx, ph in enumerate(phs):
            if ph.closed_at is not None:
                continue
            evidence = evidence_by_ph.get(ph.ph_id)
            # The grade must be trajectory and so must this round's evidence: a latched grade
            # carried by room-class evidence (a watch between bursts) does not show that the
            # fused tag's fixes matched this PH in this round.
            if (
                evidence is None
                or evidence.attestation_grade != WearableAttestationGrade.TRAJECTORY
                or evidence.evidence_class != WearableEvidenceClass.TRAJECTORY
            ):
                continue
            coasting = ph.ph_id in coasting_ph_ids
            if coasting:
                camera_seen_at = self._camera_seen_at.get(ph.ph_id, ph.last_seen_at)
                if (
                    wcfg.coasting_bridge_max_s <= 0.0
                    or (now - camera_seen_at).total_seconds() > wcfg.coasting_bridge_max_s
                ):
                    continue
            fix = select_fusion_fix(
                fixes,
                tag_id=evidence.tag_id,
                state_time=ph.last_seen_at,
                fused_through=self._fused_through.get(ph.ph_id),
                allow_newer=coasting,
                cfg=wcfg,
            )
            if fix is None:
                m.cts_wearable_spatial_fusion_total.labels(
                    mode=mode, outcome=WearableSpatialFusionOutcome.NO_FIX
                ).inc()
                continue
            state = KalmanState(
                mean=np.array(ph.state_mean, dtype=np.float64),
                covariance=np.array(ph.state_cov, dtype=np.float64).reshape(4, 4),
                updated_at=ph.last_seen_at,
            )
            fused = fuse_position_fix(state, fix, cfg=wcfg, motion=motion)
            self._fused_through[ph.ph_id] = fix.measured_at
            if fused.gated or fused.state is None:
                m.cts_wearable_spatial_fusion_total.labels(
                    mode=mode, outcome=WearableSpatialFusionOutcome.GATED
                ).inc()
                continue
            advanced = fused.state.updated_at > ph.last_seen_at
            m.cts_wearable_spatial_fusion_total.labels(
                mode=mode,
                outcome=(
                    WearableSpatialFusionOutcome.BRIDGED
                    if advanced
                    else WearableSpatialFusionOutcome.APPLIED
                ),
            ).inc()
            m.cts_wearable_spatial_fusion_shift_m.labels(mode=mode).observe(fused.shift_m)
            if mode != "enabled":
                continue
            mean = tuple(float(v) for v in fused.state.mean)
            fused_ph = replace(
                ph,
                state_mean=(mean[0], mean[1], mean[2], mean[3]),
                state_cov=tuple(float(v) for v in fused.state.covariance.flatten()),
                last_seen_at=max(ph.last_seen_at, fused.state.updated_at),
                last_floor_speed_m_s=speed_m_s((mean[0], mean[1], mean[2], mean[3])),
            )
            await self._ph_repo.save(fused_ph)
            result[idx] = fused_ph
        return result

    def _maybe_apply_zero_velocity_update(
        self,
        *,
        ph_id: str,
        predicted_state: KalmanState,
        updated_state: KalmanState,
        observation_x_m: float,
        observation_y_m: float,
        observation_cov_m2: np.typing.NDArray[np.float64],
    ) -> KalmanState:
        """Gate and apply ZUPT for one matched PH update."""
        cfg = self._config
        speed = speed_m_s(
            (
                float(updated_state.mean[0]),
                float(updated_state.mean[1]),
                float(updated_state.mean[2]),
                float(updated_state.mean[3]),
            )
        )
        if speed > cfg.zupt_speed_exit_m_s:
            self._still_counter.pop(ph_id, None)
            return updated_state

        innovation_small = (
            mahalanobis2_position(
                predicted_state,
                observation_x_m,
                observation_y_m,
                observation_cov_m2,
            )
            < cfg.zupt_innov_chi2
        )
        if speed < cfg.zupt_speed_enter_m_s and innovation_small:
            self._still_counter[ph_id] = self._still_counter.get(ph_id, 0) + 1

        if self._still_counter.get(ph_id, 0) >= cfg.zupt_consecutive_frames:
            return zero_velocity_update(updated_state, cfg.zupt_velocity_sigma_m_s)

        return updated_state

    async def _detect_copresence(
        self,
        updated_phs: list[PersonHypothesis],
        identity_by_ph: dict[str, dict[str, object]],
        now: datetime,
    ) -> None:
        """Write co-presence links for open PHs in the same overlap group
        that share a committed identity.

        Guardrail: both PHs must have a non-None ``current_identity_id`` and
        those identities must match.  This ensures two strangers in the same
        room are never linked.
        """
        if not self._overlap_groups or self._copresence_repo is None:
            return

        # Build camera → group_id lookup.
        camera_to_group: dict[str, str] = {}
        for group in self._overlap_groups:
            for cam_id in group.camera_ids:
                camera_to_group[cam_id] = group.group_id

        # Filter to open PHs with a committed identity.
        open_phs = [
            ph for ph in updated_phs if ph.closed_at is None and ph.current_identity_id is not None
        ]

        for i, ph_a in enumerate(open_phs):
            group_a = camera_to_group.get(ph_a.last_seen_camera)
            if group_a is None:
                continue
            for ph_b in open_phs[i + 1 :]:
                group_b = camera_to_group.get(ph_b.last_seen_camera)
                if group_b is None or group_b != group_a:
                    continue
                if ph_a.ph_id == ph_b.ph_id:
                    continue
                # Identity-equality guardrail.
                if (
                    ph_a.current_identity_id is not None
                    and ph_a.current_identity_id == ph_b.current_identity_id
                ):
                    aid, bid = sorted([ph_a.ph_id, ph_b.ph_id])
                    link = CoPresenceLink(
                        id=str(uuid.uuid4()),
                        group_id=group_a,
                        ph_id_a=aid,
                        ph_id_b=bid,
                        identity_id=str(ph_a.current_identity_id),
                        first_observed_at=now,
                        last_observed_at=now,
                        observation_count=1,
                    )
                    try:
                        assert self._copresence_repo is not None  # narrow for mypy
                        await self._copresence_repo.upsert_link(link)
                    except Exception:
                        logger.warning(
                            "copresence_link_write_failed",
                            group_id=group_a,
                            ph_id_a=aid,
                            ph_id_b=bid,
                            exc_info=True,
                        )
                        continue
                    _metrics.metrics.worldtracker_copresence_links_total.inc()
                    logger.debug(
                        "copresence_link_written",
                        group_id=group_a,
                        ph_id_a=aid,
                        ph_id_b=bid,
                        identity_id=ph_a.current_identity_id,
                    )


# ---------------------------------------------------------------------------
# PH advance helper (pure, no I/O)
# ---------------------------------------------------------------------------


def _camera_measurement_cov(obs: WorldObservation, cfg: WorldTrackerConfig) -> np.ndarray:
    """The covariance a camera observation's Kalman update uses."""
    if obs.floor_cov_random is not None and obs.floor_point.calibrated:
        return tuple_to_cov2x2(obs.floor_cov_random)
    return isotropic_cov(cfg.observation_noise_m)


def _advance_unmatched_ph(
    ph: PersonHypothesis,
    predicted_ks: KalmanState,
    now: datetime,
    ph_close_grace_s: float,
    *,
    camera_seen_at: datetime | None = None,
    bridge_max_s: float = 0.0,
    bridge_fix_slack_s: float = 0.0,
) -> PersonHypothesis:
    """Close an unmatched PH past ph_close_grace_s or keep its stored state unchanged.

    Stored state_mean and state_cov always represent the posterior at last_seen_at;
    in-round predictions are transient and never written back to the PH.

    A PH whose ``last_seen_at`` a fused wearable fix advanced past its last camera
    observation (*camera_seen_at*) is bridged: it stays open while that fix is within
    ``ph_close_grace_s + bridge_fix_slack_s`` (fixes arrive in delayed micro-batches) and
    the camera observation within ``max(bridge_max_s, ph_close_grace_s)``, and closes once
    either runs out, however fresh its fixes.
    """
    _ = predicted_ks
    since_seen_s = (now - ph.last_seen_at).total_seconds()
    bridged = camera_seen_at is not None and bridge_max_s > 0.0 and camera_seen_at < ph.last_seen_at
    if bridged:
        assert camera_seen_at is not None
        since_camera_s = (now - camera_seen_at).total_seconds()
        expired = (
            since_camera_s > max(bridge_max_s, ph_close_grace_s)
            or since_seen_s > ph_close_grace_s + bridge_fix_slack_s
        )
    else:
        expired = since_seen_s > ph_close_grace_s
    if expired:
        _metrics.metrics.world_tracker_ph_closed_total.inc()
        _metrics.metrics.ph_lifetime_seconds.observe((now - ph.born_at).total_seconds())
        _metrics.metrics.ph_observations_at_close.observe(ph.observation_count)
        return replace(ph, closed_at=now)
    return ph


# ---------------------------------------------------------------------------
# Identity resolution helper (pure orchestration, no Kalman math)
# ---------------------------------------------------------------------------


async def _resolve_identities(
    *,
    resolver: IdentityResolver | None,
    obs_repo: WorldObservationRepositoryProtocol,
    ph_repo: PHRepositoryProtocol,
    phs: list[PersonHypothesis],
    ph_obs_meta: dict[str, tuple[int, BoundingBox | None, float]],
    face_anchors: list[FaceAnchor],
    det_to_ph: dict[str, str] | None = None,
    face_evidence: list[FaceEvidence] | None = None,
    open_ph_identities: Mapping[str, str] | None = None,
    wearable_evidence: Mapping[str, WearableEvidence] | None = None,
    open_ph_occupancies: Mapping[str, OpenPHOccupancy] | None = None,
    copresence_pairs: frozenset[tuple[str, str]] | None = None,
    overlap_groups: Sequence[OverlapGroup] | None = None,
    camera_room_map: Mapping[str, str] | None = None,
    wearable_present_identities: frozenset[str] | None = None,
    ph_appearance_features: Mapping[str, AppearanceFeature] | None = None,
    now: datetime,
    config: WorldTrackerConfig,
) -> tuple[
    list[IdentityDecision],
    list[IdentityRevision],
    dict[str, dict[str, object]],
    list[ImageOnlyEvaluation],
]:
    """Run the Bayesian identity resolver on PHs that received observations.

    In PH mode, FaceAnchor.tracklet_id is empty when produced by
    FaceIdentityStage (ph_id is not yet assigned at that stage).  We remap
    here — after the assignment step has built det_to_ph — so the resolver
    can match anchors to PHs via entity_id without polluting observation_ids.

    Returns:
        (decisions, revisions, identity_by_ph, image_only) where identity_by_ph maps
        ph_id → {identity_id, identity_confidence, posterior_entropy,
                  direct_face_evidence} for populating snapshot fields, and image_only holds
        the round's sampled image-only evaluations of wearable-attested PHs.
    """
    # Remap PH-mode face anchors: fill in tracklet_id = ph_id for any anchor
    # that arrived with an empty tracklet_id but a known detection → PH mapping.
    if det_to_ph and face_anchors:
        face_anchors = [
            replace(fa, tracklet_id=det_to_ph[fa.detection_id])
            if (not fa.tracklet_id and fa.detection_id and fa.detection_id in det_to_ph)
            else fa
            for fa in face_anchors
        ]
    # Same remap for typed evidence records so source weighting works correctly.
    if det_to_ph and face_evidence:
        face_evidence = [
            replace(fe, tracklet_id=det_to_ph[fe.detection_id])
            if (not fe.tracklet_id and fe.detection_id and fe.detection_id in det_to_ph)
            else fe
            for fe in face_evidence
        ]
    identity_by_ph: dict[str, dict[str, object]] = {}

    if resolver is None:
        logger.debug("identity_resolver_not_configured")
        return [], [], identity_by_ph, []

    # Find PHs that received observations this frame.
    active_ph_ids = set(ph_obs_meta.keys())
    resolvable_phs = [ph for ph in phs if ph.ph_id in active_ph_ids]

    if not resolvable_phs:
        return [], [], identity_by_ph, []

    # Build resolvable wrappers with real observation IDs from the repository.
    resolvable: list[IdentityResolvableEntity] = []
    for ph in resolvable_phs:
        obs_list = await obs_repo.list_by_ph(ph.ph_id, limit=20)
        obs_ids = []
        for obs in obs_list:
            if obs.observation_id:
                obs_ids.append(obs.observation_id)
            else:
                # Fallback for observations without floor coordinates.
                obs_ids.append(f"{obs.camera_id}:{obs.frame_index}:{obs.captured_at.isoformat()}")
        resolvable.append(_PHResolvable(_ph=ph, _obs_ids=obs_ids))

    logger.debug(
        "identity_resolution_start",
        ph_count=len(resolvable),
        face_anchor_count=len(face_anchors),
    )

    try:
        outcome = await resolver.resolve(
            hypotheses=resolvable,
            new_face_anchors=face_anchors,
            captured_at=now,
            ph_heights={
                ph.ph_id: ph.height_estimate_m
                for ph in resolvable_phs
                if ph.height_estimate_m is not None
            },
            ph_height_sigmas={
                ph.ph_id: ph.height_sigma_m
                for ph in resolvable_phs
                if ph.height_sigma_m is not None
            },
            ph_qualities={ph.ph_id: ph.mean_quality for ph in resolvable_phs},
            face_evidence=face_evidence,
            open_ph_identities=open_ph_identities or {},
            wearable_evidence=wearable_evidence,
            open_ph_occupancies=open_ph_occupancies,
            copresence_pairs=copresence_pairs,
            overlap_groups=overlap_groups,
            camera_room_map=camera_room_map,
            wearable_present_identities=wearable_present_identities,
            ph_appearance_features=ph_appearance_features,
        )
    except Exception:
        logger.exception("identity_resolution_failed")
        raise

    # Apply identity decisions using explicit repo operations that preserve
    # last_independent_identity_evidence_at semantics.
    for decision in outcome.decisions:
        if decision.identity_id is not None:
            if decision.evidence_backed:
                # Evidence age runs from the newest contributing face, which for a face template
                # can be older than this round; the commit itself happens now.
                await ph_repo.evidence_backed_commit(
                    ph_id=decision.ph_id,
                    identity_id=decision.identity_id,
                    evidence_at=decision.evidence_at or now,
                    committed_at=now,
                )
            else:
                await ph_repo.prior_only_update(
                    ph_id=decision.ph_id,
                    identity_id=decision.identity_id,
                    committed_at=now,
                )
                # Prior-only maintenance must never advance independent identity
                # evidence time; prior_only_update upholds that. Count for the
                # prior-refresh dashboard/alert.
                _metrics.metrics.identity_prior_only_updates_total.inc()
                # Page-now invariant: a prior-only decision carries the OLD
                # evidence timestamp, strictly before this frame. If it equals or
                # exceeds `now`, evidence time was advanced by a prior-only path.
                now_ns = int(now.timestamp() * 1e9)
                ev_ns = decision.last_independent_evidence_at_unix_ns
                if ev_ns and ev_ns >= now_ns:
                    _metrics.metrics.identity_prior_only_evidence_advance_total.inc()
        elif decision.revises_previous and decision.previous_identity_id is not None:
            await ph_repo.clear_to_unknown(
                ph_id=decision.ph_id,
                committed_at=now,
            )
        identity_by_ph[decision.ph_id] = {
            "identity_id": decision.identity_id,
            "identity_confidence": decision.posterior.top_identity()[1]
            if decision.posterior.distribution
            else 0.0,
            "posterior_entropy": decision.posterior.entropy(),
            "direct_face_evidence": bool(
                decision.evidence
                and isinstance(decision.evidence, dict)
                and float(decision.evidence.get("direct_face_confidence", 0.0) or 0.0) > 0.0  # type: ignore[arg-type]
            ),
            "evidence_json": decision.evidence_json,
            "inferred_identity_id": decision.inferred_identity_id,
            "effective_identity_id": decision.effective_identity_id,
            "authority": decision.authority,
            "decision_source": decision.decision_source,
            "decision_id": decision.decision_id,
            "conflict": decision.conflict,
            "last_independent_evidence_at_unix_ns": decision.last_independent_evidence_at_unix_ns,
            "config_hash": decision.config_hash,
            "model_set_version": decision.model_set_version,
        }

    # Collect revisions (publishing is handled by RevisionsStage).
    revisions: list[IdentityRevision] = list(outcome.revisions)

    logger.info(
        "identity_resolution_complete",
        decisions=len(outcome.decisions),
        revisions=len(revisions),
        identities_assigned=sum(1 for d in outcome.decisions if d.identity_id is not None),
    )

    return list(outcome.decisions), revisions, identity_by_ph, list(outcome.image_only)
