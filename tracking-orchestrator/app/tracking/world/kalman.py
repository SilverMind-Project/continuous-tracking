"""Floor-plane Kalman filter, constant-velocity model.

Why constant-velocity (not constant-acceleration): pedestrian motion changes
direction faster than acceleration can be estimated reliably from 5-10 Hz
observations. Constant-velocity with a moderate Q matrix tracks well enough
for senior-care indoor scenarios.

Why np.linalg.solve over np.linalg.inv: solving Sx = y is more numerically
stable than computing S^-1 and multiplying. The inverse can amplify noise
when S is poorly conditioned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

NDArrayF8 = np.typing.NDArray[np.float64]

# Cap on position-block trace (m^2) to prevent unbounded covariance growth
# across long unobserved gaps. In healthy operation the cap is never hit.
_POSITION_TRACE_CAP: float = 100.0


@dataclass(frozen=True)
class KalmanState:
    """Kalman filter state on the floor plane.

    mean: [x, y, vx, vy] in metres and m/s.
    covariance: 4x4 matrix, row-major.
    """

    mean: NDArrayF8  # shape (4,)
    covariance: NDArrayF8  # shape (4, 4)
    updated_at: datetime


def initialize(
    floor_x_m: float,
    floor_y_m: float,
    initial_position_sigma_m: float,
    initial_velocity_sigma_m_s: float,
    now: datetime,
) -> KalmanState:
    """Create a new Kalman state at *floor_x_m*, *floor_y_m* with zero velocity."""
    mean = np.array([floor_x_m, floor_y_m, 0.0, 0.0], dtype=np.float64)
    cov = np.diag(
        [
            initial_position_sigma_m**2,
            initial_position_sigma_m**2,
            initial_velocity_sigma_m_s**2,
            initial_velocity_sigma_m_s**2,
        ]
    ).astype(np.float64)
    return KalmanState(mean=mean, covariance=cov, updated_at=now)


def predict(
    state: KalmanState,
    now: datetime,
    process_noise_accel_m_s2: float,
    velocity_decay_s: float = 3.0,
) -> KalmanState:
    """Advance state to *now*. dt may be > 0; identity transform when dt == 0."""
    dt = (now - state.updated_at).total_seconds()
    if dt <= 0:
        return state

    # State transition (constant velocity).
    F = np.array(  # noqa: N806
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Velocity decay: a person we have not observed for a while is more
    # likely stationary than continuing at last-seen velocity.
    decay = float(np.exp(-dt / velocity_decay_s))
    decayed_mean = state.mean.copy()
    decayed_mean[2] *= decay
    decayed_mean[3] *= decay

    predicted_mean = F @ decayed_mean

    # Process noise from white-noise acceleration model.
    sigma_a2 = process_noise_accel_m_s2**2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    Q = sigma_a2 * np.array(  # noqa: N806
        [
            [dt4 / 4, 0.0, dt3 / 2, 0.0],
            [0.0, dt4 / 4, 0.0, dt3 / 2],
            [dt3 / 2, 0.0, dt2, 0.0],
            [0.0, dt3 / 2, 0.0, dt2],
        ],
        dtype=np.float64,
    )

    predicted_cov = F @ state.covariance @ F.T + Q

    # Cap covariance trace to prevent unbounded growth across long gaps.
    pos_trace = float(predicted_cov[0, 0] + predicted_cov[1, 1])
    if pos_trace > _POSITION_TRACE_CAP:
        scale = _POSITION_TRACE_CAP / pos_trace
        predicted_cov[0:2, 0:2] *= scale

    return KalmanState(mean=predicted_mean, covariance=predicted_cov, updated_at=now)


def isotropic_cov(sigma_m: float) -> NDArrayF8:
    """R = sigma_m² · I (2,2). Use for synthetic/uncalibrated floor points with no Jacobian."""
    return (sigma_m**2) * np.eye(2, dtype=np.float64)


def update(
    state: KalmanState,
    observation_x_m: float,
    observation_y_m: float,
    observation_cov_m2: NDArrayF8,  # (2,2) m²
) -> KalmanState:
    """Apply a position observation. Returns the posterior state."""
    z = np.array([observation_x_m, observation_y_m], dtype=np.float64)
    H = np.array(  # noqa: N806
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    R = observation_cov_m2  # noqa: N806

    # Innovation.
    y = z - H @ state.mean  # shape (2,)
    S = H @ state.covariance @ H.T + R  # shape (2, 2)  # noqa: N806

    # Kalman gain: K = P H^T S^-1 via solve for numerical stability.
    K_T = np.linalg.solve(S, H @ state.covariance.T)  # noqa: N806
    K = K_T.T  # shape (4, 2)  # noqa: N806

    new_mean = state.mean + K @ y
    eye = np.eye(4, dtype=np.float64)
    new_cov = (eye - K @ H) @ state.covariance

    # Symmetrize to guard against numerical drift.
    new_cov = 0.5 * (new_cov + new_cov.T)

    return KalmanState(mean=new_mean, covariance=new_cov, updated_at=state.updated_at)


def zero_velocity_update(
    state: KalmanState,
    velocity_meas_sigma_m_s: float,
) -> KalmanState:
    """Apply a zero-velocity pseudo-measurement to the velocity sub-state.

    Measurement model H_v selects ``vx, vy`` with z=0 and
    R_v = sigma² · I. Positions can move via cross-covariance, which removes
    drift caused by noise-induced phantom velocity.
    """
    z = np.array([0.0, 0.0], dtype=np.float64)
    H = np.array(  # noqa: N806
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    R = (velocity_meas_sigma_m_s**2) * np.eye(2, dtype=np.float64)  # noqa: N806

    y = z - H @ state.mean
    S = H @ state.covariance @ H.T + R  # noqa: N806
    K_T = np.linalg.solve(S, H @ state.covariance.T)  # noqa: N806
    K = K_T.T  # noqa: N806

    new_mean = state.mean + K @ y
    eye = np.eye(4, dtype=np.float64)
    new_cov = (eye - K @ H) @ state.covariance
    new_cov = 0.5 * (new_cov + new_cov.T)

    return KalmanState(mean=new_mean, covariance=new_cov, updated_at=state.updated_at)


def mahalanobis2_position(
    state: KalmanState,
    observation_x_m: float,
    observation_y_m: float,
    observation_cov_m2: NDArrayF8,  # (2,2) m²
) -> float:
    """Squared Mahalanobis distance of an observation under the predicted state.

    Returned in chi-squared distribution with 2 dof. Used by the gate.

    Fails closed: returns ``math.inf`` (never raises, never returns ``NaN``) when
    any input is non-finite or the innovation-covariance solve is singular, so an
    invalid pair is gated out instead of leaking ``NaN`` into the cost matrix and
    the Hungarian solver. The caller maps ``inf`` to its ``GATE_INF`` sentinel.
    """
    z = np.array([observation_x_m, observation_y_m], dtype=np.float64)
    R = np.asarray(observation_cov_m2, dtype=np.float64)  # noqa: N806
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(state.mean))):
        return math.inf
    if not (np.all(np.isfinite(state.covariance)) and np.all(np.isfinite(R))):
        return math.inf
    H = np.array(  # noqa: N806
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y = z - H @ state.mean
    S = H @ state.covariance @ H.T + R  # noqa: N806
    try:
        x = np.linalg.solve(S, y)
    except np.linalg.LinAlgError:
        return math.inf
    d2 = float(y @ x)
    if not math.isfinite(d2):
        return math.inf
    return d2
