"""WorldTracker: top-level orchestrator for world-coordinate person tracking.

Depends on Protocols (PHRepositoryProtocol, WorldObservationRepositoryProtocol);
no direct I/O. Called by WorldTrackingStage once per frame.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import numpy as np
from structlog import get_logger

if TYPE_CHECKING:
    from ...inference.evidence import FaceEvidence
    from ...tracking.identity_resolver import IdentityResolver

from ...domain import (
    BoundingBox,
    FaceAnchor,
    IdentityDecision,
    IdentityResolvableEntity,
    IdentityRevision,
    PersonHypothesis,
    PHContinuationCandidate,
    WorldFrameSnapshot,
    WorldObservation,
)
from ...observability import metrics as _metrics
from ...storage.base import PHRepositoryProtocol, WorldObservationRepositoryProtocol
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
from .kalman import KalmanState, initialize, predict, update

logger = get_logger(__name__)


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
    def last_seen_at(self) -> datetime:
        return self._ph.last_seen_at

    @property
    def started_at(self) -> datetime:
        return self._ph.born_at


@dataclass(frozen=True)
class WorldTrackerResult:
    """Output of one WorldTracker.step() call."""

    updated_phs: list[PersonHypothesis]
    snapshots: list[WorldFrameSnapshot]
    continuations: list[PHContinuationCandidate]
    identity_decisions: list[IdentityDecision] = field(default_factory=list)
    revisions: list[IdentityRevision] = field(default_factory=list)
    det_to_ph: dict[str, str] = field(default_factory=dict)


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
    ) -> None:
        self._ph_repo = ph_repo
        self._obs_repo = obs_repo
        self._config = config or WorldTrackerConfig()
        self._continuation_publisher = continuation_publisher
        self._identity_resolver = identity_resolver

    async def step(
        self,
        observations: list[WorldObservation],
        now: datetime,
        face_anchors_by_detection: dict[str, FaceAnchor] | None = None,
        room_polygons: dict[str, list[tuple[float, float]]] | None = None,
        camera_room_map: dict[str, str] | None = None,
        face_anchors: list[FaceAnchor] | None = None,
        face_evidence: list[FaceEvidence] | None = None,
    ) -> WorldTrackerResult:
        """Run one frame of the world tracker.

        Args:
            observations: calibrated WorldObservations from all cameras.
            now: current wall-clock time (pipeline event_time).
            face_anchors_by_detection: face anchors keyed by detection_id.
            room_polygons: room_id → list of (x_m, y_m) vertices.
            camera_room_map: camera_id → room_name fallback.
            face_anchors: flat list of FaceAnchors for identity resolution.

        Returns:
            WorldTrackerResult with updated PHs, snapshots, and continuations.
        """
        cfg = self._config
        room_polygons = room_polygons or {}
        camera_room_map = camera_room_map or {}
        continuations: list[PHContinuationCandidate] = []
        identity_decisions: list[IdentityDecision] = []
        det_to_ph: dict[str, str] = {}

        # 1. Load active PHs and predict forward.
        active_phs = await self._ph_repo.list_open()
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
        observations, cluster_map = dedup_observations(raw_observations, cfg)

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

        obs_floor_points: list[tuple[float, float]] = [
            (obs.floor_point.x_mm / 1000.0, obs.floor_point.y_mm / 1000.0) for obs in observations
        ]
        obs_embeddings: list[list[float] | None] = [
            obs.embedding if obs.embedding else None for obs in observations
        ]
        obs_face_person_ids: list[str | None] = [
            obs.face_anchor.person_id if obs.face_anchor else None for obs in observations
        ]
        obs_face_confidences: list[float] = [
            obs.face_anchor.confidence if obs.face_anchor else 0.0 for obs in observations
        ]
        obs_heights: list[float | None] = [obs.height_estimate_m for obs in observations]

        ph_gallery_means: list[list[float] | None] = [ph.gallery_mean for ph in active_phs]
        ph_identity_ids: list[str | None] = [ph.current_identity_id for ph in active_phs]
        ph_heights: list[float | None] = [ph.height_estimate_m for ph in active_phs]

        # 3. Associate.
        assignment = associate(
            ph_states=predicted_states,
            ph_gallery_means=ph_gallery_means,
            ph_identity_ids=ph_identity_ids,
            ph_heights=ph_heights,
            obs_floor_points=obs_floor_points,
            obs_embeddings=obs_embeddings,
            obs_face_person_ids=obs_face_person_ids,
            obs_face_confidences=obs_face_confidences,
            obs_height_estimates=obs_heights,
            cfg=cfg,
        )

        updated_phs: list[PersonHypothesis] = []
        # Track per-PH observation metadata for snapshot building.
        ph_obs_meta: dict[
            str, tuple[int, BoundingBox | None, float]
        ] = {}  # ph_id -> (frame_index, bbox, detection_confidence)

        # Build a lookup from detection_id to the original (pre-dedup) observation.
        obs_by_det_id = {obs.detection_id: obs for obs in raw_observations if obs.detection_id}

        # 4. Update matched PHs.
        for ph_idx, obs_idx in assignment.matched:
            ph = active_phs[ph_idx]
            ks = predicted_states[ph_idx]
            obs = observations[obs_idx]  # deduped representative

            new_state = update(
                ks,
                obs.floor_point.x_mm / 1000.0,
                obs.floor_point.y_mm / 1000.0,
                cfg.observation_noise_m,
            )
            new_gallery_mean = update_gallery_mean(
                ph.gallery_mean, obs.embedding, ph.observation_count
            )
            new_height = update_height_ema(ph.height_estimate_m, obs.height_estimate_m, alpha=0.1)
            new_mean_quality: float = 0.1 * obs.quality + 0.9 * ph.mean_quality

            # Expand active_cameras to include all cameras from the dedup cluster.
            src_ids = cluster_map.get(obs.detection_id, (obs.detection_id,))
            cluster_cameras = frozenset(
                obs_by_det_id[did].camera_id for did in src_ids if did in obs_by_det_id
            )

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
            )
            updated_phs.append(updated)
            ph_obs_meta[ph.ph_id] = (
                obs.frame_index,
                obs.bbox,
                obs.detection_confidence,
            )
            # Map all source detection IDs (not just the representative) to this PH.
            for src_det_id in src_ids:
                if src_det_id:
                    det_to_ph[src_det_id] = ph.ph_id

            # Persist ALL source observations (not just the representative) so that
            # both cameras' raw rows land on the PH (U1 requirement).
            for src_det_id in src_ids:
                src_obs = obs_by_det_id.get(src_det_id, obs)
                await self._obs_repo.save(src_obs, ph_id=ph.ph_id)

        # 5. Spawn new PHs for unmatched observations.
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

            ks = initialize(
                fx,
                fy,
                cfg.initial_position_sigma_m,
                cfg.initial_velocity_sigma_m_s,
                obs.captured_at,
            )
            # Include all cameras from the dedup cluster when spawning.
            spawn_src_ids = cluster_map.get(obs.detection_id, (obs.detection_id,))
            spawn_cameras = frozenset(
                obs_by_det_id[did].camera_id for did in spawn_src_ids if did in obs_by_det_id
            ) or frozenset([obs.camera_id])

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
                current_identity_id=obs.face_anchor.person_id if obs.face_anchor else None,
                gallery_mean=obs.embedding if obs.embedding else None,
                height_estimate_m=obs.height_estimate_m,
                active_cameras=spawn_cameras,
                last_floor_speed_m_s=0.0,
                mean_quality=obs.quality,
            )
            updated_phs.append(new_ph)
            _metrics.metrics.world_tracker_ph_spawned_total.inc()
            ph_obs_meta[new_ph.ph_id] = (
                obs.frame_index,
                obs.bbox,
                obs.detection_confidence,
            )
            # Save the new PH first so the FK constraint on world_observations is satisfied.
            await self._ph_repo.save(new_ph)
            # Persist all source observations for this spawned PH.
            for src_det_id in spawn_src_ids:
                src_obs = obs_by_det_id.get(src_det_id, obs)
                await self._obs_repo.save(src_obs, ph_id=new_ph.ph_id)
            for src_det_id in spawn_src_ids:
                if src_det_id:
                    det_to_ph[src_det_id] = new_ph.ph_id

            # 6. Check for PH continuations from recently closed PHs.
            if self._continuation_publisher is not None:
                lookback = obs.captured_at - timedelta(seconds=cfg.inferred_handoff_max_s)
                recent_closed = await self._ph_repo.list_closed_since(lookback, limit=100)
                for closed in recent_closed:
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

        # 7. Close PHs that have not been observed for ph_close_grace_s.
        for ph_idx in assignment.unmatched_phs:
            ph = active_phs[ph_idx]
            grace = (now - ph.last_seen_at).total_seconds()
            if grace > cfg.ph_close_grace_s:
                closed = PersonHypothesis(
                    ph_id=ph.ph_id,
                    state_mean=ph.state_mean,
                    state_cov=ph.state_cov,
                    born_at=ph.born_at,
                    last_seen_at=ph.last_seen_at,
                    last_seen_camera=ph.last_seen_camera,
                    observation_count=ph.observation_count,
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
                updated_phs.append(closed)
                _metrics.metrics.world_tracker_ph_closed_total.inc()
            else:
                # Keep open but unobserved; update state to predicted.
                ks = predicted_states[ph_idx]
                updated = PersonHypothesis(
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
                    gallery_mean=ph.gallery_mean,
                    height_estimate_m=ph.height_estimate_m,
                    active_cameras=ph.active_cameras,
                    closed_at=None,
                    last_floor_speed_m_s=ph.last_floor_speed_m_s,
                    last_posture=ph.last_posture,
                    metadata=ph.metadata,
                    mean_quality=ph.mean_quality,
                )
                updated_phs.append(updated)

        # 8. Persist all updated PHs.
        for ph in updated_phs:
            await self._ph_repo.save(ph)

        # 9. Identity resolution: run the Bayesian resolver on PHs that
        #    received observations this frame.
        identity_decisions, revisions, identity_by_ph = await _resolve_identities(
            resolver=self._identity_resolver,
            obs_repo=self._obs_repo,
            ph_repo=self._ph_repo,
            phs=updated_phs,
            ph_obs_meta=ph_obs_meta,
            face_anchors=face_anchors or [],
            det_to_ph=det_to_ph,
            face_evidence=face_evidence,
            now=now,
            config=cfg,
        )

        # 10. Build snapshots for downstream stages.
        snapshots: list[WorldFrameSnapshot] = []
        for ph in updated_phs:
            if ph.observation_count < cfg.min_observations_to_publish:
                continue
            room_id, room_name = resolve_room(
                ph.state_mean[0],
                ph.state_mean[1],
                ph.last_seen_camera,
                room_polygons,
                camera_room_map,
            )
            obs_meta = ph_obs_meta.get(ph.ph_id, (0, None, 0.0))
            obs_frame_index, obs_bbox, obs_det_conf = obs_meta
            id_data = identity_by_ph.get(ph.ph_id, {})
            snapshots.append(
                WorldFrameSnapshot(
                    ph_id=ph.ph_id,
                    camera_id=ph.last_seen_camera,
                    frame_index=obs_frame_index,
                    captured_at=ph.last_seen_at,
                    floor_x_m=ph.state_mean[0],
                    floor_y_m=ph.state_mean[1],
                    floor_vx_m_s=ph.state_mean[2],
                    floor_vy_m_s=ph.state_mean[3],
                    position_sigma_m=position_sigma_m(ph.state_cov),
                    identity_id=(
                        str(id_data["identity_id"])
                        if id_data.get("identity_id") is not None
                        else ph.current_identity_id
                    ),
                    identity_confidence=float(id_data.get("identity_confidence", 0.0) or 0.0),  # type: ignore[arg-type]
                    posterior_entropy=float(id_data.get("posterior_entropy", 0.0) or 0.0),  # type: ignore[arg-type]
                    direct_face_evidence=bool(id_data.get("direct_face_evidence", False) or False),
                    bbox=obs_bbox,
                    detection_confidence=obs_det_conf,
                    height_m=ph.height_estimate_m,
                    room_id=room_id,
                    room_name=room_name,
                    mean_quality=ph.mean_quality,
                )
            )

        return WorldTrackerResult(
            updated_phs=updated_phs,
            snapshots=snapshots,
            continuations=continuations,
            identity_decisions=identity_decisions,
            revisions=revisions,
            det_to_ph=det_to_ph,
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
        )
    except Exception:
        logger.exception("identity_resolution_failed")
        raise

    # Apply identity decisions.
    for decision in outcome.decisions:
        if decision.identity_id:
            await ph_repo.update_identity(
                ph_id=decision.ph_id,
                identity_id=decision.identity_id,
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
