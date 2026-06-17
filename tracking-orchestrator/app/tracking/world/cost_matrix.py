"""Per-pair cost between Person Hypotheses and observations.

The cost matrix is the only place where geometry, appearance, identity,
and height come together. Keep all weighting decisions in one file so
tuning is centralized.
"""

from __future__ import annotations

from ...domain import ViewPrototype
from .config import WorldTrackerConfig
from .helpers import cosine_similarity
from .kalman import KalmanState, isotropic_cov, mahalanobis2_position

GATE_INF: float = 1.0e9  # sentinel for "do not match"

# Minimum prototype count to use it for appearance cost.
_PROTOTYPE_MIN_COUNT = 2


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
) -> float:
    """Return finite cost in [0, GATE_INF). GATE_INF means do not match.

    When *calibrated* is False, the floor point is synthetic (bbox centre in
    a virtual tile).  If ``enable_uncalibrated_gate_relax`` is on, a wider
    geometric gate and appearance-biased cost weights are used so that
    synthetic-point jitter does not force association failure.

    When ``enable_multiview_association`` is on and *ph_view_prototypes*
    is non-empty, appearance cost uses max-over-prototypes cosine similarity
    instead of the single gallery_mean.
    """
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

    # 1. Geometric gate.
    d2 = mahalanobis2_position(
        ph_state,
        obs_floor_x_m,
        obs_floor_y_m,
        isotropic_cov(cfg.observation_noise_m),
    )
    if d2 > gate:
        return GATE_INF
    geo_cost = d2 / gate  # [0, 1]

    # 2. Identity-conflict hard gate.
    if (
        obs_face_anchor_person_id is not None
        and ph_current_identity_id
        and obs_face_anchor_person_id != ph_current_identity_id
        and obs_face_anchor_confidence >= cfg.face_conflict_threshold
    ):
        return GATE_INF

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

    return alpha_geo * geo_cost + alpha_app * app_cost + cfg.alpha_height * height_cost
