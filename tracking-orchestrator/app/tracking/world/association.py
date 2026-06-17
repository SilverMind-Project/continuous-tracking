"""Hungarian assignment between active PHs and current-frame observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from ...domain import ViewPrototype, tuple_to_cov2x2
from .config import WorldTrackerConfig
from .cost_matrix import GATE_INF, pair_cost
from .kalman import KalmanState

NDArrayF8 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Assignment:
    """Result of one frame's Hungarian association."""

    matched: list[tuple[int, int]]  # (ph_index, obs_index)
    unmatched_phs: list[int]
    unmatched_obs: list[int]


def associate(
    ph_states: list[KalmanState],
    ph_gallery_means: list[list[float] | None],
    ph_identity_ids: list[str | None],
    ph_heights: list[float | None],
    obs_floor_points: list[tuple[float, float]],
    obs_embeddings: list[list[float] | None],
    obs_face_person_ids: list[str | None],
    obs_face_confidences: list[float],
    obs_height_estimates: list[float | None],
    cfg: WorldTrackerConfig,
    *,
    obs_calibrated: list[bool] | None = None,
    ph_view_prototypes: list[tuple[ViewPrototype, ...]] | None = None,
    obs_covs: list[tuple[float, float, float, float] | None] | None = None,
) -> Assignment:
    """Hungarian assignment with gating.

    Pairs whose cost is GATE_INF are excluded from the matched set.
    When obs_covs is provided, each observation's covariance is used in the
    Mahalanobis gate so uncertain observations gate more permissively.
    """
    n_ph = len(ph_states)
    n_obs = len(obs_floor_points)
    if n_ph == 0:
        return Assignment(matched=[], unmatched_phs=[], unmatched_obs=list(range(n_obs)))
    if n_obs == 0:
        return Assignment(matched=[], unmatched_phs=list(range(n_ph)), unmatched_obs=[])

    calib_flags = obs_calibrated if obs_calibrated is not None else [True] * n_obs
    prototypes = ph_view_prototypes if ph_view_prototypes is not None else [()] * n_ph
    cov_tuples: list[tuple[float, float, float, float] | None] = (
        obs_covs if obs_covs is not None else [None] * n_obs
    )

    cost = np.full((n_ph, n_obs), GATE_INF, dtype=np.float64)
    for i in range(n_ph):
        for j in range(n_obs):
            cov_t = cov_tuples[j]
            obs_cov_m2: NDArrayF8 | None = tuple_to_cov2x2(cov_t) if cov_t is not None else None
            cost[i, j] = pair_cost(
                ph_state=ph_states[i],
                ph_gallery_mean=ph_gallery_means[i],
                ph_current_identity_id=ph_identity_ids[i],
                ph_height_m=ph_heights[i],
                obs_floor_x_m=obs_floor_points[j][0],
                obs_floor_y_m=obs_floor_points[j][1],
                obs_embedding=obs_embeddings[j],
                obs_face_anchor_person_id=obs_face_person_ids[j],
                obs_face_anchor_confidence=obs_face_confidences[j],
                obs_height_estimate_m=obs_height_estimates[j],
                cfg=cfg,
                calibrated=calib_flags[j],
                ph_view_prototypes=prototypes[i],
                obs_cov_m2=obs_cov_m2,
            )

    # If every pair is gated, skip the solver.
    if not np.any(cost < GATE_INF):
        return Assignment(
            matched=[],
            unmatched_phs=list(range(n_ph)),
            unmatched_obs=list(range(n_obs)),
        )

    row_ind, col_ind = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    matched_phs: set[int] = set()
    matched_obs: set[int] = set()
    for r, c in zip(row_ind, col_ind, strict=True):
        if cost[r, c] < GATE_INF:
            matched.append((int(r), int(c)))
            matched_phs.add(int(r))
            matched_obs.add(int(c))

    unmatched_phs = [i for i in range(n_ph) if i not in matched_phs]
    unmatched_obs = [j for j in range(n_obs) if j not in matched_obs]
    return Assignment(matched=matched, unmatched_phs=unmatched_phs, unmatched_obs=unmatched_obs)
