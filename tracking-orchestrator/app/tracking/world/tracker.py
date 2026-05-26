"""WorldTracker: top-level orchestrator for world-coordinate person tracking.

Depends on Protocols (PHRepositoryProtocol, WorldObservationRepository);
no direct I/O. Called by WorldTrackingStage once per frame.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
from structlog import get_logger

from ...domain import (
    BoundingBox,
    FaceAnchor,
    PersonHypothesis,
    PHContinuationCandidate,
    WorldFrameSnapshot,
    WorldObservation,
)
from .association import associate
from .config import WorldTrackerConfig
from .helpers import (
    is_in_any_room_polygon,
    position_sigma_m,
    resolve_room,
    speed_m_s,
    update_gallery_mean,
    update_height_ema,
)
from .kalman import KalmanState, initialize, predict, update
from .repository import PHRepositoryProtocol, WorldObservationRepository

logger = get_logger(__name__)


class ContinuationPublisher(Protocol):
    """Publishes PHContinuationCandidate events to tracking.continuations."""

    async def publish(self, candidate: PHContinuationCandidate) -> None: ...


@dataclass(frozen=True)
class WorldTrackerResult:
    """Output of one WorldTracker.step() call."""

    updated_phs: list[PersonHypothesis]
    snapshots: list[WorldFrameSnapshot]
    continuations: list[PHContinuationCandidate]


class WorldTracker:
    """Single floor-plane Kalman tracker for multi-camera person tracking.

    One instance per process. Called once per frame with all calibrated
    observations from all cameras. No per-camera tracker, no cross-camera
    merge pass, no healing pass.
    """

    def __init__(
        self,
        ph_repo: PHRepositoryProtocol,
        obs_repo: WorldObservationRepository,
        config: WorldTrackerConfig | None = None,
        continuation_publisher: ContinuationPublisher | None = None,
    ) -> None:
        self._ph_repo = ph_repo
        self._obs_repo = obs_repo
        self._config = config or WorldTrackerConfig()
        self._continuation_publisher = continuation_publisher

    async def step(
        self,
        observations: list[WorldObservation],
        now: datetime,
        face_anchors_by_detection: dict[str, FaceAnchor] | None = None,
        room_polygons: dict[str, list[tuple[float, float]]] | None = None,
        camera_room_map: dict[str, str] | None = None,
    ) -> WorldTrackerResult:
        """Run one frame of the world tracker.

        Args:
            observations: calibrated WorldObservations from all cameras.
            now: current wall-clock time (pipeline event_time).
            face_anchors_by_detection: face anchors keyed by detection_id.
            room_polygons: room_id → list of (x_m, y_m) vertices.
            camera_room_map: camera_id → room_name fallback.

        Returns:
            WorldTrackerResult with updated PHs, snapshots, and continuations.
        """
        cfg = self._config
        _anchors = face_anchors_by_detection or {}
        room_polygons = room_polygons or {}
        camera_room_map = camera_room_map or {}
        continuations: list[PHContinuationCandidate] = []

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

        # 2. Build observation lists for the association step.
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
        ph_obs_meta: dict[str, tuple[int, BoundingBox | None, float]] = (
            {}
        )  # ph_id -> (frame_index, bbox, detection_confidence)

        # 4. Update matched PHs.
        for ph_idx, obs_idx in assignment.matched:
            ph = active_phs[ph_idx]
            ks = predicted_states[ph_idx]
            obs = observations[obs_idx]

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
                active_cameras=ph.active_cameras | {obs.camera_id},
                last_floor_speed_m_s=speed_m_s(
                    (
                        float(new_state.mean[0]),
                        float(new_state.mean[1]),
                        float(new_state.mean[2]),
                        float(new_state.mean[3]),
                    )
                ),
            )
            updated_phs.append(updated)
            ph_obs_meta[ph.ph_id] = (
                obs.frame_index,
                obs.bbox,
                obs.detection_confidence,
            )

            # Persist the observation.
            await self._obs_repo.save(obs, ph_id=ph.ph_id)

        # 5. Spawn new PHs for unmatched observations.
        for obs_idx in assignment.unmatched_obs:
            obs = observations[obs_idx]
            fx = obs.floor_point.x_mm / 1000.0
            fy = obs.floor_point.y_mm / 1000.0

            if not is_in_any_room_polygon(fx, fy, room_polygons):
                continue

            ks = initialize(
                fx,
                fy,
                cfg.initial_position_sigma_m,
                cfg.initial_velocity_sigma_m_s,
                obs.captured_at,
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
                current_identity_id=obs.face_anchor.person_id if obs.face_anchor else None,
                gallery_mean=obs.embedding if obs.embedding else None,
                height_estimate_m=obs.height_estimate_m,
                active_cameras=frozenset([obs.camera_id]),
                last_floor_speed_m_s=0.0,
            )
            updated_phs.append(new_ph)
            ph_obs_meta[new_ph.ph_id] = (
                obs.frame_index,
                obs.bbox,
                obs.detection_confidence,
            )
            await self._obs_repo.save(obs, ph_id=new_ph.ph_id)

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
                )
                updated_phs.append(closed)
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
                )
                updated_phs.append(updated)

        # 8. Persist all updated PHs.
        for ph in updated_phs:
            await self._ph_repo.save(ph)

        # 9. Build snapshots for downstream stages.
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
            obs_meta = ph_obs_meta.get(
                ph.ph_id, (0, None, 0.0)
            )
            obs_frame_index, obs_bbox, obs_det_conf = obs_meta
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
                    identity_id=ph.current_identity_id,
                    identity_confidence=0.0,
                    posterior_entropy=0.0,
                    direct_face_evidence=False,
                    bbox=obs_bbox,
                    detection_confidence=obs_det_conf,
                    height_m=ph.height_estimate_m,
                    room_id=room_id,
                    room_name=room_name,
                )
            )

        return WorldTrackerResult(
            updated_phs=updated_phs,
            snapshots=snapshots,
            continuations=continuations,
        )
