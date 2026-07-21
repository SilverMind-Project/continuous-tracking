"""Per-pair cost between Person Hypotheses and observations.

The cost matrix is the only place where geometry, appearance, identity,
and height come together. Keep all weighting decisions in one file so
tuning is centralized.

Fail-closed contract (M03): every pair resolves to a finite cost in
``[0, GATE_INF]``. ``GATE_INF`` means "do not match" and carries a typed
:class:`RejectionReason` so association can record *why* a pair was excluded
(invalid point, invalid covariance, geometric gate, direct identity conflict,
appearance outlier, solver-invalid cost). No ``NaN`` ever leaves this module.
"""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from ...domain import ViewPrototype
from .config import WorldTrackerConfig
from .helpers import cosine_similarity
from .kalman import KalmanState, isotropic_cov, mahalanobis2_position
from .validation import is_finite_point, is_valid_covariance

NDArrayF8 = npt.NDArray[np.float64]

GATE_INF: float = 1.0e9  # sentinel for "do not match"

# Minimum prototype count to use it for appearance cost.
_PROTOTYPE_MIN_COUNT = 2


class RejectionReason(StrEnum):
    """Why a (PH, observation) pair was gated out of the matched set.

    Wire values are stable metric label values; renaming requires updating
    dashboards. ``MATCHED`` is the sentinel for "not rejected".
    """

    MATCHED = "matched"
    INVALID_POINT = "invalid_point"
    INVALID_COVARIANCE = "invalid_covariance"
    GEOMETRIC_GATE = "geometric_gate"
    IDENTITY_CONFLICT = "identity_conflict"
    APPEARANCE_OUTLIER = "appearance_outlier"
    SOLVER_INVALID_COST = "solver_invalid_cost"


def pair_cost_detail(
    ph_state: KalmanState,
    ph_gallery_mean: list[float] | None,
    ph_current_identity_id: str | None,
    ph_height_m: float | None,
    obs_floor_x_m: float,
    obs_floor_y_m: float,
    obs_embedding: list[float] | None,
    obs_face_anchor_person_id: str | None,
    obs_face_anchor_confidence: float,
    obs_height_estimate_m: float | None,
    cfg: WorldTrackerConfig,
    *,
    calibrated: bool = True,
    ph_view_prototypes: tuple[ViewPrototype, ...] = (),
    obs_cov_m2: NDArrayF8 | None = None,
    ph_identity_is_operator_confirmed: bool = False,
    obs_verified_reid_identity_id: str | None = None,
) -> tuple[float, RejectionReason]:
    """Return ``(cost, reason)`` for one PH/observation pair.

    ``cost`` is finite in ``[0, GATE_INF]``; ``GATE_INF`` is paired with the
    typed ``reason`` for why the pair is unmatchable. A matched pair returns
    ``RejectionReason.MATCHED``.

    Typed authoritative identity evidence (M03 task 5):

    - A qualifying *direct* ArcFace conflict (recognized face whose person_id
      differs from the PH identity, confidence ≥ ``face_conflict_threshold``)
      is a hard gate. Only recognized anchors reach here — candidate / weak /
      propagated face evidence is filtered upstream and never hard-gates.
    - ``ph_identity_is_operator_confirmed`` makes the PH identity absolute: a
      conflicting recognized face hard-gates at *any* confidence (operator
      authority cannot be overridden by a marginal face match).
    - ``obs_verified_reid_identity_id`` disagreement is a configurable strong
      *cost*, never a hard gate. The verified-ReID identity is plumbed by M05;
      until then it is ``None`` here and the branch is inert.

    When *calibrated* is False and ``enable_uncalibrated_gate_relax`` is on, a
    wider geometric gate and appearance-biased weights are used so synthetic
    floor-point jitter does not force association failure. When
    ``enable_multiview_association`` is on and *ph_view_prototypes* is non-empty,
    appearance cost uses max-over-prototypes cosine similarity.
    """
    # 0. Fail-closed input validation. Non-finite points/covariance always gate
    #    out (pure safety); the symmetry/PSD/trace checks are additionally
    #    enforced under enable_covariance_validation.
    if not is_finite_point(obs_floor_x_m, obs_floor_y_m):
        return GATE_INF, RejectionReason.INVALID_POINT

    gate_cov = obs_cov_m2 if obs_cov_m2 is not None else isotropic_cov(cfg.observation_noise_m)
    gate_cov = np.asarray(gate_cov, dtype=np.float64)
    if not bool(np.all(np.isfinite(gate_cov))):
        return GATE_INF, RejectionReason.INVALID_COVARIANCE
    if cfg.enable_covariance_validation and not is_valid_covariance(
        gate_cov,
        symmetry_tol_m2=cfg.covariance_symmetry_tol_m2,
        psd_tol_m2=cfg.covariance_psd_tol_m2,
        max_trace_m2=cfg.covariance_max_trace_m2,
    ):
        return GATE_INF, RejectionReason.INVALID_COVARIANCE

    # Select gate and weights: relaxed for uncalibrated when enabled.
    if not calibrated and cfg.enable_uncalibrated_gate_relax:
        gate = cfg.uncalibrated_gate_chi2
        alpha_app = cfg.uncalibrated_alpha_app
        # Renormalize: alpha_geo + alpha_app + alpha_height ≈ 1.0.
        alpha_geo = 1.0 - alpha_app - cfg.alpha_height
        if alpha_geo < 0.0:
            alpha_geo = 0.0
    else:
        gate = cfg.gate_chi2
        alpha_geo = cfg.alpha_geo
        alpha_app = cfg.alpha_app

    # 1. Geometric gate. mahalanobis2_position fails closed to inf on bad input.
    d2 = mahalanobis2_position(ph_state, obs_floor_x_m, obs_floor_y_m, gate_cov)
    if not math.isfinite(d2):
        return GATE_INF, RejectionReason.SOLVER_INVALID_COST
    if d2 > gate:
        return GATE_INF, RejectionReason.GEOMETRIC_GATE
    geo_cost = d2 / gate  # [0, 1]

    # 2. Identity-conflict hard gate (direct ArcFace + operator authority).
    if obs_face_anchor_person_id is not None and ph_current_identity_id:
        identities_disagree = obs_face_anchor_person_id != ph_current_identity_id
        face_qualifies = obs_face_anchor_confidence >= cfg.face_conflict_threshold
        if identities_disagree and (face_qualifies or ph_identity_is_operator_confirmed):
            return GATE_INF, RejectionReason.IDENTITY_CONFLICT

    # 3. Appearance.
    app_cost = 0.5  # neutral when one side has no embedding yet
    if obs_embedding is not None:
        if cfg.enable_multiview_association and ph_view_prototypes:
            # max-over-prototypes cosine similarity.
            best_sim = 0.0
            qualified = [p for p in ph_view_prototypes if p.count >= _PROTOTYPE_MIN_COUNT]
            if qualified:
                best_sim = max(
                    cosine_similarity(list(p.embedding), obs_embedding) for p in qualified
                )
                app_cost = max(0.0, min(1.0, 1.0 - best_sim))
            elif ph_gallery_mean is not None:
                # Fall back to gallery_mean when no qualified prototypes.
                sim = cosine_similarity(ph_gallery_mean, obs_embedding)
                app_cost = max(0.0, min(1.0, 1.0 - sim))
        elif ph_gallery_mean is not None:
            sim = cosine_similarity(ph_gallery_mean, obs_embedding)
            app_cost = max(0.0, min(1.0, 1.0 - sim))

    # 4. Height plausibility.
    height_cost = 0.0
    if obs_height_estimate_m is not None and ph_height_m is not None:
        z = (obs_height_estimate_m - ph_height_m) / cfg.height_sigma_m
        height_cost = min(1.0, 0.5 * z * z)

    cost = alpha_geo * geo_cost + alpha_app * app_cost + cfg.alpha_height * height_cost

    # 5. Verified-ReID disagreement: configurable strong cost, not a hard gate.
    #    Inert until plumbs obs_verified_reid_identity_id.
    if (
        cfg.enable_reid_disagreement_cost
        and obs_verified_reid_identity_id is not None
        and ph_current_identity_id
        and obs_verified_reid_identity_id != ph_current_identity_id
    ):
        cost += cfg.reid_disagreement_cost

    # 6. Final fail-closed guard: a non-finite cost must never reach the solver.
    if not math.isfinite(cost):
        return GATE_INF, RejectionReason.SOLVER_INVALID_COST

    return cost, RejectionReason.MATCHED


def pair_cost(
    ph_state: KalmanState,
    ph_gallery_mean: list[float] | None,
    ph_current_identity_id: str | None,
    ph_height_m: float | None,
    obs_floor_x_m: float,
    obs_floor_y_m: float,
    obs_embedding: list[float] | None,
    obs_face_anchor_person_id: str | None,
    obs_face_anchor_confidence: float,
    obs_height_estimate_m: float | None,
    cfg: WorldTrackerConfig,
    *,
    calibrated: bool = True,
    ph_view_prototypes: tuple[ViewPrototype, ...] = (),
    obs_cov_m2: NDArrayF8 | None = None,
    ph_identity_is_operator_confirmed: bool = False,
    obs_verified_reid_identity_id: str | None = None,
) -> float:
    """Return finite cost in ``[0, GATE_INF]``; ``GATE_INF`` means do not match.

    Thin wrapper over :func:`pair_cost_detail` that drops the reason. Kept for
    callers (and tests) that only need the scalar cost.
    """
    cost, _reason = pair_cost_detail(
        ph_state=ph_state,
        ph_gallery_mean=ph_gallery_mean,
        ph_current_identity_id=ph_current_identity_id,
        ph_height_m=ph_height_m,
        obs_floor_x_m=obs_floor_x_m,
        obs_floor_y_m=obs_floor_y_m,
        obs_embedding=obs_embedding,
        obs_face_anchor_person_id=obs_face_anchor_person_id,
        obs_face_anchor_confidence=obs_face_anchor_confidence,
        obs_height_estimate_m=obs_height_estimate_m,
        cfg=cfg,
        calibrated=calibrated,
        ph_view_prototypes=ph_view_prototypes,
        obs_cov_m2=obs_cov_m2,
        ph_identity_is_operator_confirmed=ph_identity_is_operator_confirmed,
        obs_verified_reid_identity_id=obs_verified_reid_identity_id,
    )
    return cost
