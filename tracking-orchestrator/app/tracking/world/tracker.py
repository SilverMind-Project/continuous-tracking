"""WorldTracker: top-level orchestrator for world-coordinate person tracking.

Depends on Protocols (PHRepositoryProtocol, WorldObservationRepositoryProtocol);
no direct I/O. Called by WorldTrackingStage once per frame.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import numpy as np
from structlog import get_logger

if TYPE_CHECKING:
    from ...inference.evidence import FaceEvidence
    from ...storage.base import GalleryRepository
    from ...tracking.identity_resolver import IdentityResolver

from ...domain import (
    BoundingBox,
    CameraTopologyEdge,
    CoPresenceLink,
    FaceAnchor,
    IdentityDecision,
    IdentityResolvableEntity,
    IdentityRevision,
    OrientationBin,
    OverlapGroup,
    PersonHypothesis,
    PHContinuationCandidate,
    ViewPrototype,
    WorldFrameSnapshot,
    WorldObservation,
    tuple_to_cov2x2,
)
from ...observability import metrics as _metrics
from ...storage.base import (
    CameraTopologyRepository,
    CoPresenceRepository,
    PHRepositoryProtocol,
    WorldObservationRepositoryProtocol,
)
from ..orientation import update_view_prototypes
from .appearance_policy import AppearanceDecision, evaluate_appearance_update
from .association import associate
from .config import WorldTrackerConfig
from .dedup import dedup_observations
from .helpers import (
    is_in_any_room_polygon,
    position_sigma_m,
    resolve_room,
    speed_m_s,
    update_gallery_mean,
    update_height_ema,
)
from .kalman import (
    KalmanState,
    initialize,
    isotropic_cov,
    mahalanobis2_position,
    predict,
    update,
    zero_velocity_update,
)
from .revival import select_revival_candidate
from .topology import record_handoff

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
    revived_ph_ids: frozenset[str] = frozenset()


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

    Only recognized face anchors are forwarded to the identity-conflict gate;
    candidate and unrecognized anchors are weak-positive signals that must not
    trigger a hard conflict.
    """
    floor_points = [
        (obs.floor_point.x_mm / 1000.0, obs.floor_point.y_mm / 1000.0) for obs in observations
    ]
    embeddings: list[list[float] | None] = [obs.embedding or None for obs in observations]
    face_person_ids: list[str | None] = [
        obs.face_anchor.person_id
        if obs.face_anchor and obs.face_anchor.recognition_state == "recognized"
        else None
        for obs in observations
    ]
    face_confidences: list[float] = [
        obs.face_anchor.confidence
        if obs.face_anchor and obs.face_anchor.recognition_state == "recognized"
        else 0.0
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
    ) -> None:
        self._ph_repo = ph_repo
        self._obs_repo = obs_repo
        self._config = config or WorldTrackerConfig()
        self._continuation_publisher = continuation_publisher
        self._identity_resolver = identity_resolver
        self._gallery_repo = gallery_repo
        self._topology_repo = topology_repo
        self._copresence_repo = copresence_repo
        self._overlap_groups = overlap_groups or []
        # In-process cache of each PH's last positive identity confidence.
        # On coasting / unresolved frames the resolver produces no decision for a
        # PH, so its per-frame posterior is absent (0.0). Replaying the last
        # committed confidence here keeps the published top_probability meaningful
        # for a held identity instead of emitting a sentinel 0.0 that the UI shows
        # as null. Lost on restart (self-heals on next resolution); pruned on close.
        self._last_identity_confidence: dict[str, float] = {}
        # Snapshot of open PHs from the most recent step() call. Exposed via
        # last_open_phs so ReidNeedPolicy (InferenceStage) can evaluate proximity
        # without issuing an async DB query. Empty until the first step().
        self._last_open_phs: list[PersonHypothesis] = []
        self._still_counter: dict[str, int] = {}
        self._primary_camera: dict[str, str] = {}
        self._primary_challenger: dict[str, tuple[str, int]] = {}

    @property
    def last_open_phs(self) -> list[PersonHypothesis]:
        """Open PHs from the most recent step() call.

        Synchronous snapshot for ReidNeedPolicy proximity checks.
        Empty until the first step() completes.
        """
        return self._last_open_phs

    async def _resolve_verified_reid_identities(
        self, observations: list[WorldObservation]
    ) -> list[str | None]:
        """Resolve each observation's verified-ReID identity from the gallery.

        Only ``operator_verified`` gallery entries vote (program architecture
        decision 6): ``search_similar``'s default state filter now guarantees
        this at the repository boundary (M03), and the explicit
        ``entry.state == "operator_verified"`` check below remains as a
        paged-invariant backstop in case a future caller widens the state
        filter. Returns a list aligned to *observations*; an entry is ``None``
        when there is no embedding, no gallery repository, or no
        operator_verified match clears ``reid_disagreement_min_similarity``.
        The caller only invokes this when ``enable_reid_disagreement_cost`` is
        true, so it adds no per-frame gallery queries while the flag is off.
        """
        if self._gallery_repo is None:
            return [None] * len(observations)
        cfg = self._config
        resolved: list[str | None] = []
        for obs in observations:
            identity: str | None = None
            if obs.embedding:
                hits = await self._gallery_repo.search_similar(obs.embedding, limit=1)
                if hits:
                    entry, similarity = hits[0]
                    if (
                        entry.state == "operator_verified"
                        and entry.identity_id
                        and similarity >= cfg.reid_disagreement_min_similarity
                    ):
                        identity = entry.identity_id
            resolved.append(identity)
        return resolved

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
    ) -> WorldTrackerResult:
        """Run one frame of the world tracker.

        Args:
            observations: calibrated WorldObservations from all cameras.
            now: current wall-clock time (pipeline event_time).
            face_anchors_by_detection: face anchors keyed by detection_id.
            room_polygons: room_id → list of (x_m, y_m) vertices.
            room_names: room_id → display name for polygon-derived rooms.
            camera_room_map: camera_id → room_name fallback.
            face_anchors: flat list of FaceAnchors for identity resolution.

        Returns:
            WorldTrackerResult with updated PHs, snapshots, and continuations.
        """
        cfg = self._config
        room_polygons = room_polygons or {}
        camera_room_map = camera_room_map or {}
        room_names = room_names or {}
        continuations: list[PHContinuationCandidate] = []
        identity_decisions: list[IdentityDecision] = []
        det_to_ph: dict[str, str] = {}

        # 1. Load active PHs and predict forward.
        active_phs = await self._ph_repo.list_open()
        # Cache for ReidNeedPolicy (InferenceStage proximity check next frame).
        self._last_open_phs = list(active_phs)
        predicted_states: list[KalmanState] = []
        for ph in active_phs:
            ks = KalmanState(
                mean=np.array(ph.state_mean, dtype=np.float64),
                covariance=np.array(ph.state_cov, dtype=np.float64).reshape(4, 4),
                updated_at=ph.last_seen_at,
            )
            predicted_states.append(
                predict(ks, now, cfg.process_noise_accel_m_s2, cfg.velocity_decay_s)
            )

        # 2. Build observation lists; run cross-camera dedup before association.
        raw_observations = observations  # keep original for cluster expansion
        observations, cluster_map = dedup_observations(
            raw_observations, cfg, overlap_groups=self._overlap_groups or None
        )

        # Emit dedup metrics.
        m = _metrics.metrics
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

        # M12: resolve each observation's verified-ReID identity from the
        # governed (operator_verified) gallery so association can charge a
        # disagreement cost (M03 cost path). Gated on the flag: while it is
        # off this adds zero per-frame gallery queries and the input stays
        # None, leaving association behaviour unchanged.
        obs_verified_reid_ids: list[str | None] | None = None
        if cfg.enable_reid_disagreement_cost:
            obs_verified_reid_ids = await self._resolve_verified_reid_identities(observations)

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

        # M03: record association integrity diagnostics from the PRIMARY pass
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
        ph_floor_calibrated: dict[str, bool] = {}  # ph_id -> obs.floor_point.calibrated this frame
        ph_confidence_meta: dict[
            str, tuple[int, bool]
        ] = {}  # ph_id -> (contributing_camera_count, footpoint_reliable)
        # Representative observation per PH updated this frame. Consumed after
        # identity resolution (step 9) to seed the multi-view gallery with the
        # COMMITTED identity. Seeding inside step 4 used the pre-resolution
        # identity, which is None on the very frame a face first commits, so in
        # practice the gallery never seeded.
        seed_obs_by_ph: dict[str, WorldObservation] = {}

        # Build a lookup from detection_id to the original (pre-dedup) observation.
        obs_by_det_id = {obs.detection_id: obs for obs in raw_observations if obs.detection_id}

        # 4. Update matched PHs.
        for ph_idx, obs_idx in assignment.matched:
            ph = active_phs[ph_idx]
            ks = predicted_states[ph_idx]
            obs = observations[obs_idx]  # deduped representative

            obs_r = (
                tuple_to_cov2x2(obs.floor_cov_random)
                if obs.floor_cov_random is not None and obs.floor_point.calibrated
                else isotropic_cov(cfg.observation_noise_m)
            )
            new_state = update(
                ks,
                obs.floor_point.x_mm / 1000.0,
                obs.floor_point.y_mm / 1000.0,
                obs_r,
            )
            new_state = self._maybe_apply_zero_velocity_update(
                ph_id=ph.ph_id,
                predicted_state=ks,
                updated_state=new_state,
                observation_x_m=obs.floor_point.x_mm / 1000.0,
                observation_y_m=obs.floor_point.y_mm / 1000.0,
                observation_cov_m2=obs_r,
            )
            # M03 contamination guard: a geometrically valid match may still be
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
            new_height = update_height_ema(ph.height_estimate_m, obs.height_estimate_m, alpha=0.1)
            room_id, room_name = resolve_room(
                obs.floor_point.x_mm / 1000.0,
                obs.floor_point.y_mm / 1000.0,
                obs.camera_id,
                room_polygons,
                camera_room_map,
                room_names,
            )

            # Expand active_cameras to include all cameras from the dedup cluster.
            src_ids = cluster_map.get(obs.detection_id, (obs.detection_id,))
            cluster_cameras = frozenset(
                obs_by_det_id[did].camera_id for did in src_ids if did in obs_by_det_id
            )
            self._update_primary_camera(ph.ph_id, _best_primary_camera(src_ids, obs_by_det_id, obs))

            updated = PersonHypothesis(
                ph_id=ph.ph_id,
                state_mean=(
                    float(new_state.mean[0]),
                    float(new_state.mean[1]),
                    float(new_state.mean[2]),
                    float(new_state.mean[3]),
                ),
                state_cov=tuple(float(v) for v in new_state.covariance.flatten()),
                born_at=ph.born_at,
                last_seen_at=obs.captured_at,
                last_seen_camera=obs.camera_id,
                observation_count=ph.observation_count + 1,
                current_identity_id=ph.current_identity_id,
                current_identity_committed_at=ph.current_identity_committed_at,
                last_independent_identity_evidence_at=ph.last_independent_identity_evidence_at,
                gallery_mean=new_gallery_mean,
                height_estimate_m=new_height,
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
            ph_floor_calibrated[ph.ph_id] = obs.floor_point.calibrated
            ph_confidence_meta[ph.ph_id] = (len(src_ids), obs.footpoint_reliable)
            # Map all source detection IDs (not just the representative) to this PH.
            for src_det_id in src_ids:
                if src_det_id:
                    det_to_ph[src_det_id] = ph.ph_id

            # Persist ALL source observations (not just the representative) so that
            # both cameras' raw rows land on the PH (U1 requirement).
            for src_det_id in src_ids:
                src_obs = obs_by_det_id.get(src_det_id, obs)
                await self._obs_repo.save(src_obs, ph_id=ph.ph_id)

            # Defer multi-view gallery seeding until after identity resolution
            # (step 9) so the committed identity is known. Capture the obs here.
            seed_obs_by_ph[ph.ph_id] = obs

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
            except Exception:  # noqa: BLE001
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

            if cfg.enable_ph_revival:
                revival_candidate = select_revival_candidate(
                    obs,
                    _recent_closed,
                    obs.captured_at,
                    cfg,
                    enable_cross_camera=cfg.enable_cross_camera_revival,
                    topology_edges=_topology_edges,
                )
            else:
                # Shadow: compute revival candidate to measure what it would catch.
                shadow_candidate = select_revival_candidate(
                    obs,
                    _recent_closed,
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
                        _recent_closed,
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
                new_ph = PersonHypothesis(
                    ph_id=closed.ph_id,
                    state_mean=(
                        float(ks.mean[0]),
                        float(ks.mean[1]),
                        float(ks.mean[2]),
                        float(ks.mean[3]),
                    ),
                    state_cov=tuple(float(v) for v in ks.covariance.flatten()),
                    born_at=closed.born_at,
                    last_seen_at=obs.captured_at,
                    last_seen_camera=obs.camera_id,
                    observation_count=closed.observation_count + 1,
                    current_identity_id=closed.current_identity_id,
                    current_identity_committed_at=closed.current_identity_committed_at,
                    last_independent_identity_evidence_at=closed.last_independent_identity_evidence_at,
                    gallery_mean=new_gallery_mean,
                    height_estimate_m=closed.height_estimate_m,
                    active_cameras=closed.active_cameras | spawn_cameras,
                    closed_at=None,  # reopen
                    last_floor_speed_m_s=0.0,
                    last_posture=closed.last_posture,
                    metadata=_metadata_with_room(closed.metadata, room_id, room_name),
                    mean_quality=closed.mean_quality,
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
                    current_identity_id=_sanitize_identity_id(
                        obs.face_anchor.person_id if obs.face_anchor else None
                    ),
                    gallery_mean=obs.embedding if obs.embedding else None,
                    height_estimate_m=obs.height_estimate_m,
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
            ph_floor_calibrated[new_ph.ph_id] = obs.floor_point.calibrated
            ph_confidence_meta[new_ph.ph_id] = (len(spawn_src_ids), obs.footpoint_reliable)
            # Defer gallery seeding to after identity resolution (step 9).
            seed_obs_by_ph[new_ph.ph_id] = obs
            # Save the PH first so the FK constraint on world_observations is satisfied.
            await self._ph_repo.save(new_ph)
            # Persist all source observations for this PH.
            for src_det_id in spawn_src_ids:
                src_obs = obs_by_det_id.get(src_det_id, obs)
                await self._obs_repo.save(src_obs, ph_id=new_ph.ph_id)
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
                    ks = predicted_states[ph_idx]
                    obs = lb_obs_raw[lb_obs_idx]
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
                    recovered = PersonHypothesis(
                        ph_id=ph.ph_id,
                        state_mean=(
                            float(new_state.mean[0]),
                            float(new_state.mean[1]),
                            float(new_state.mean[2]),
                            float(new_state.mean[3]),
                        ),
                        state_cov=tuple(float(v) for v in new_state.covariance.flatten()),
                        born_at=ph.born_at,
                        last_seen_at=obs.captured_at,
                        last_seen_camera=obs.camera_id,
                        observation_count=ph.observation_count,
                        current_identity_id=ph.current_identity_id,
                        current_identity_committed_at=ph.current_identity_committed_at,
                        last_independent_identity_evidence_at=ph.last_independent_identity_evidence_at,
                        gallery_mean=ph.gallery_mean,
                        height_estimate_m=ph.height_estimate_m,
                        active_cameras=ph.active_cameras | frozenset([obs.camera_id]),
                        last_floor_speed_m_s=speed_m_s(
                            (
                                float(new_state.mean[0]),
                                float(new_state.mean[1]),
                                float(new_state.mean[2]),
                                float(new_state.mean[3]),
                            )
                        ),
                        mean_quality=ph.mean_quality,
                        view_prototypes=ph.view_prototypes,
                        metadata=ph.metadata,
                        last_posture=ph.last_posture,
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
        for ph_idx in assignment.unmatched_phs:
            if ph_idx in lb_matched_ph_indices:
                continue
            ph = active_phs[ph_idx]
            updated_phs.append(
                _advance_unmatched_ph(ph, predicted_states[ph_idx], now, cfg.ph_close_grace_s)
            )

        # 8. Persist all updated PHs.
        for ph in updated_phs:
            await self._ph_repo.save(ph)

        # 9. Identity resolution: run the Bayesian resolver on PHs that
        #    received observations this frame.  Pass the full open-PH identity
        #    occupancy (updated_phs is a superset of all open PHs) so the
        #    duplicate-active-identity guard also protects incumbents that were
        #    not observed this frame.
        open_ph_identities = {
            ph.ph_id: ph.current_identity_id
            for ph in updated_phs
            if ph.closed_at is None and ph.current_identity_id
        }
        identity_decisions, revisions, identity_by_ph = await _resolve_identities(
            resolver=self._identity_resolver,
            obs_repo=self._obs_repo,
            ph_repo=self._ph_repo,
            phs=updated_phs,
            ph_obs_meta=ph_obs_meta,
            face_anchors=face_anchors or [],
            det_to_ph=det_to_ph,
            face_evidence=face_evidence,
            open_ph_identities=open_ph_identities,
            now=now,
            config=cfg,
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
        self._primary_camera = {
            pid: camera_id for pid, camera_id in self._primary_camera.items() if pid in open_ph_ids
        }
        self._primary_challenger = {
            pid: challenger
            for pid, challenger in self._primary_challenger.items()
            if pid in open_ph_ids
        }

        # Seed the multi-view gallery AFTER identity resolution so a face that
        # commits (or is held) this frame seeds with its committed identity.
        # _seed_multiview_gallery still requires a recognized face anchor on the
        # observation, so only face-confirmed frames seed (no poisoning).
        if self._gallery_repo is not None and self._identity_resolver is not None:
            for ph_id, seed_obs in seed_obs_by_ph.items():
                resolved = identity_by_ph.get(ph_id, {})
                raw_identity = resolved.get("identity_id")
                seed_identity = _sanitize_identity_id(
                    str(raw_identity) if raw_identity is not None else None
                )
                await self._seed_multiview_gallery(seed_identity, seed_obs)

        # Detect co-presence links for overlapping cameras sharing an identity.
        if self._copresence_repo is not None and self._overlap_groups:
            await self._detect_copresence(updated_phs, identity_by_ph, now)

        # 10. Build snapshots for downstream stages.
        snapshots: list[WorldFrameSnapshot] = []
        for ph in updated_phs:
            if ph.observation_count < cfg.min_observations_to_publish:
                continue
            primary_camera = self._primary_camera.get(ph.ph_id, ph.last_seen_camera)
            room_id, room_name = resolve_room(
                ph.state_mean[0],
                ph.state_mean[1],
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
                    floor_x_m=ph.state_mean[0],
                    floor_y_m=ph.state_mean[1],
                    floor_vx_m_s=ph.state_mean[2],
                    floor_vy_m_s=ph.state_mean[3],
                    position_sigma_m=position_sigma_m(ph.state_cov),
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
                        speed_m_s(ph.state_mean)
                        if ph_floor_calibrated.get(ph.ph_id, False)
                        else None
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
                )
            )

        return WorldTrackerResult(
            updated_phs=updated_phs,
            snapshots=snapshots,
            continuations=continuations,
            identity_decisions=identity_decisions,
            revisions=revisions,
            det_to_ph=det_to_ph,
            revived_ph_ids=frozenset(revived_ph_ids),
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
                    except Exception:  # noqa: BLE001
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

    async def _seed_multiview_gallery(
        self,
        identity_id: str | None,
        obs: WorldObservation,
    ) -> None:
        """Seed an identity's gallery with an orientation-tagged entry.

        ``identity_id`` is the PH's committed identity AFTER identity resolution
        (the caller passes the resolved, sanitized id, not the stale
        pre-resolution one). Only seeds when:
        - ``identity_id`` is a committed (non-UNKNOWN) identity.
        - The observation has a face anchor with recognition_state=="recognized".
        - Observation quality and orientation confidence clear their thresholds.
        - The per-(identity, orientation) cap is not yet reached.

        Gallery entries seeded from non-frontal frames enable the resolver's
        max-over-views query to re-identify a person who turned around.
        """
        from ...domain import GalleryEmbedding

        if self._gallery_repo is None:
            return

        if not identity_id:
            return

        face_anchor = obs.face_anchor
        if face_anchor is None or face_anchor.recognition_state != "recognized":
            return

        # Gate on orientation confidence (from resolver config).
        seed_min_conf = 0.5
        if self._identity_resolver is not None:
            seed_min_conf = self._identity_resolver._config.seed_orientation_min_confidence
        if obs.orientation_confidence < seed_min_conf:
            return

        # Do not seed UNKNOWN orientation.
        if obs.orientation == OrientationBin.UNKNOWN:
            return

        if not obs.embedding:
            return

        # Check per-(identity, orientation) cap.
        orientation_val = int(obs.orientation)
        try:
            existing = await self._gallery_repo.list_gallery_entries(
                identity_id=identity_id,
                active_only=False,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "multiview_gallery_seed_list_failed",
                identity_id=identity_id,
                exc_info=True,
            )
            return

        count_for_orientation = sum(1 for e in existing if e.orientation == orientation_val)
        _max_per_orientation = 10
        if count_for_orientation >= _max_per_orientation:
            logger.debug(
                "multiview_gallery_seed_capped",
                identity_id=identity_id,
                orientation=orientation_val,
                count=count_for_orientation,
                cap=_max_per_orientation,
            )
            return

        # Write the gallery entry.
        import uuid as _uuid

        entry = GalleryEmbedding(
            gallery_entry_id=str(_uuid.uuid4()),
            identity_id=identity_id,
            embedding=obs.embedding,
            seen_at=obs.captured_at,
            quality=obs.quality,
            face_confirmed=True,
            camera_id=obs.camera_id,
            orientation=orientation_val,
        )
        try:
            await self._gallery_repo.upsert_gallery_entry(entry)
        except Exception:  # noqa: BLE001
            logger.warning(
                "multiview_gallery_seed_failed",
                identity_id=identity_id,
                orientation=orientation_val,
                exc_info=True,
            )
            return

        logger.debug(
            "multiview_gallery_seeded",
            identity_id=identity_id,
            orientation=orientation_val,
            orientation_name=obs.orientation.name,
            quality=round(obs.quality, 3),
        )


# ---------------------------------------------------------------------------
# PH advance helper (pure, no I/O)
# ---------------------------------------------------------------------------


def _advance_unmatched_ph(
    ph: PersonHypothesis,
    predicted_ks: KalmanState,
    now: datetime,
    ph_close_grace_s: float,
) -> PersonHypothesis:
    """Return a closed or drift-predicted PH for one unmatched active track.

    Closes the PH (sets closed_at) when it has not been observed for longer
    than ph_close_grace_s and emits lifetime metrics.  Otherwise advances
    the Kalman state to the predicted position without touching identity or
    gallery state.
    """
    grace = (now - ph.last_seen_at).total_seconds()
    if grace > ph_close_grace_s:
        _metrics.metrics.world_tracker_ph_closed_total.inc()
        _metrics.metrics.ph_lifetime_seconds.observe((now - ph.born_at).total_seconds())
        _metrics.metrics.ph_observations_at_close.observe(ph.observation_count)
        return PersonHypothesis(
            ph_id=ph.ph_id,
            state_mean=ph.state_mean,
            state_cov=ph.state_cov,
            born_at=ph.born_at,
            last_seen_at=ph.last_seen_at,
            last_seen_camera=ph.last_seen_camera,
            observation_count=ph.observation_count,
            current_identity_id=ph.current_identity_id,
            current_identity_committed_at=ph.current_identity_committed_at,
            last_independent_identity_evidence_at=ph.last_independent_identity_evidence_at,
            gallery_mean=ph.gallery_mean,
            height_estimate_m=ph.height_estimate_m,
            active_cameras=ph.active_cameras,
            closed_at=now,
            last_floor_speed_m_s=ph.last_floor_speed_m_s,
            last_posture=ph.last_posture,
            metadata=ph.metadata,
            mean_quality=ph.mean_quality,
            view_prototypes=ph.view_prototypes,
        )
    # Keep open but unobserved; advance state to Kalman prediction.
    ks = predicted_ks
    return PersonHypothesis(
        ph_id=ph.ph_id,
        state_mean=(
            float(ks.mean[0]),
            float(ks.mean[1]),
            float(ks.mean[2]),
            float(ks.mean[3]),
        ),
        state_cov=tuple(float(v) for v in ks.covariance.flatten()),
        born_at=ph.born_at,
        last_seen_at=ph.last_seen_at,
        last_seen_camera=ph.last_seen_camera,
        observation_count=ph.observation_count,
        current_identity_id=ph.current_identity_id,
        current_identity_committed_at=ph.current_identity_committed_at,
        last_independent_identity_evidence_at=ph.last_independent_identity_evidence_at,
        gallery_mean=ph.gallery_mean,
        height_estimate_m=ph.height_estimate_m,
        active_cameras=ph.active_cameras,
        closed_at=None,
        last_floor_speed_m_s=ph.last_floor_speed_m_s,
        last_posture=ph.last_posture,
        metadata=ph.metadata,
        mean_quality=ph.mean_quality,
        view_prototypes=ph.view_prototypes,
    )


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
    now: datetime,
    config: WorldTrackerConfig,
) -> tuple[list[IdentityDecision], list[IdentityRevision], dict[str, dict[str, object]]]:
    """Run the Bayesian identity resolver on PHs that received observations.

    In PH mode, FaceAnchor.tracklet_id is empty when produced by
    FaceIdentityStage (ph_id is not yet assigned at that stage).  We remap
    here — after the assignment step has built det_to_ph — so the resolver
    can match anchors to PHs via entity_id without polluting observation_ids.

    Returns:
        (decisions, revisions, identity_by_ph) where identity_by_ph maps
        ph_id → {identity_id, identity_confidence, posterior_entropy,
                  direct_face_evidence} for populating snapshot fields.
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
        return [], [], identity_by_ph

    # Find PHs that received observations this frame.
    active_ph_ids = set(ph_obs_meta.keys())
    resolvable_phs = [ph for ph in phs if ph.ph_id in active_ph_ids]

    if not resolvable_phs:
        return [], [], identity_by_ph

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
            ph_qualities={ph.ph_id: ph.mean_quality for ph in resolvable_phs},
            face_evidence=face_evidence,
            open_ph_identities=open_ph_identities or {},
        )
    except Exception:
        logger.exception("identity_resolution_failed")
        raise

    # Apply identity decisions using explicit repo operations that preserve
    # last_independent_identity_evidence_at semantics.
    for decision in outcome.decisions:
        if decision.identity_id is not None:
            if decision.evidence_backed:
                await ph_repo.evidence_backed_commit(
                    ph_id=decision.ph_id,
                    identity_id=decision.identity_id,
                    evidence_at=now,
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

    return list(outcome.decisions), revisions, identity_by_ph
