"""Pure helpers that bring device-free position sensors (radar, LiDAR) into the world tracker.

A position-sensor track is geometry only: a calibrated floor point and a covariance, with no
image box, embedding, face, or height. It enters the tracker as a ``WorldObservation`` with
``source = DEVICE_FREE`` and the sensor's source id (``sensor:<sensor_id>``) in the camera
slot, so association, Kalman updates, floor history, persistence, and publication treat it
like any other measurement while every appearance path skips it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
from cts_contracts import is_device_free_source, source_id_for_sensor
from shapely.geometry import Point, Polygon

from app.domain import (
    AnonymousPositionObservation,
    FloorPoint,
    ObservationSource,
    PersonHypothesis,
    WorldObservation,
    tuple_to_cov2x2,
)
from app.observability.metrics import DeviceFreeShadowOutcome
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.dedup import dedup_observations
from app.tracking.world.kalman import (
    KalmanState,
    isotropic_cov,
    mahalanobis2_position,
    predict,
)


def anonymous_track_to_world_observation(
    track: AnonymousPositionObservation, *, frame_index: int = 0
) -> WorldObservation:
    """Convert one position-sensor sample into a geometry-only ``WorldObservation``.

    ``floor_cov_random`` carries the sensor's reported covariance and ``floor_residual_m``
    its registered calibration residual, so the dedup and Kalman paths add the sensor's
    systematic floor exactly as they do a camera's. Quality 0 and an empty embedding keep
    every appearance update from ever seeing it.
    """
    source_id = source_id_for_sensor(track.sensor_id)
    measured_ms = int(track.measured_at.timestamp() * 1000)
    return WorldObservation(
        camera_id=source_id,
        frame_index=frame_index,
        captured_at=track.measured_at,
        floor_point=FloorPoint(
            x_mm=round(track.x_m * 1000.0),
            y_mm=round(track.y_m * 1000.0),
            calibrated=True,
        ),
        bbox=None,
        embedding=[],
        detection_confidence=1.0,
        detection_id=f"pos:{track.sensor_id}:{track.sensor_track_id}:{measured_ms}",
        quality=0.0,
        floor_residual_m=track.calibration_residual_m,
        floor_cov_random=track.cov_m2,
        footpoint_reliable=True,
        primary_score=0.0,
        source=ObservationSource.DEVICE_FREE,
    )


def camera_free_room_at(
    x_m: float,
    y_m: float,
    room_polygons: Mapping[str, list[tuple[float, float]]],
    camera_room_ids: Mapping[str, str],
) -> str | None:
    """Return the room id at ``(x_m, y_m)`` when no camera is bound to that room, else ``None``.

    A device-free observation may spawn or revive a PH only there. In a room with a camera,
    an uncalibrated camera's PH sits on a virtual tile the radar point can never associate
    with, so a radar spawn would count one person twice; there the sensor only refines PHs
    the cameras already track. Without room polygons (no floor-plan scale) nothing qualifies.
    """
    camera_rooms = {
        rid for src, rid in camera_room_ids.items() if rid and not is_device_free_source(src)
    }
    point = Point(x_m, y_m)
    for room_id, vertices in room_polygons.items():
        if len(vertices) < 3 or not Polygon(vertices).contains(point):
            continue
        # Rooms do not overlap; the room holding the point decides.
        return None if room_id in camera_rooms else room_id
    return None


def evaluate_device_free_shadow(
    observations: Sequence[WorldObservation],
    open_phs: Sequence[PersonHypothesis],
    *,
    now: datetime,
    cfg: WorldTrackerConfig,
    room_polygons: Mapping[str, list[tuple[float, float]]],
    camera_room_ids: Mapping[str, str],
) -> list[DeviceFreeShadowOutcome]:
    """Classify what a device-free round would do for each observation without mutating state."""
    if not observations:
        return []
    deduped, _ = dedup_observations(list(observations), cfg)
    predicted_states = [
        predict(
            KalmanState(
                mean=np.array(ph.state_mean, dtype=np.float64),
                covariance=np.array(ph.state_cov, dtype=np.float64).reshape(4, 4),
                updated_at=ph.last_seen_at,
            ),
            now,
            cfg.process_noise_accel_m_s2,
            cfg.velocity_decay_s,
            damped=cfg.damped_velocity_prediction,
        )
        for ph in open_phs
    ]
    outcomes: list[DeviceFreeShadowOutcome] = []
    for obs in deduped:
        x_m = obs.floor_point.x_mm / 1000.0
        y_m = obs.floor_point.y_mm / 1000.0
        obs_cov = (
            tuple_to_cov2x2(obs.floor_cov_random)
            if obs.floor_cov_random is not None
            else isotropic_cov(cfg.observation_noise_m)
        )
        if any(
            mahalanobis2_position(pred, x_m, y_m, obs_cov) <= cfg.gate_chi2
            for pred in predicted_states
        ):
            outcomes.append(DeviceFreeShadowOutcome.WOULD_UPDATE)
        elif camera_free_room_at(x_m, y_m, room_polygons, camera_room_ids) is not None:
            outcomes.append(DeviceFreeShadowOutcome.WOULD_SPAWN)
        else:
            outcomes.append(DeviceFreeShadowOutcome.WOULD_DROP)
    return outcomes


def has_visual_history(ph: PersonHypothesis) -> bool:
    """Whether any camera has ever observed *ph* (position sensors alone do not count).

    A PH seen only by position sensors has no crop, so the resolver's crop-quality commit
    gate has nothing to judge; its only possible commit evidence is the wearable channel.
    """
    return any(not is_device_free_source(source_id) for source_id in ph.active_cameras)
