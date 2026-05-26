"""Per-pair cost between Person Hypotheses and observations.

The cost matrix is the only place where geometry, appearance, identity,
and height come together. Keep all weighting decisions in one file so
tuning is centralized.
"""

from __future__ import annotations

from .config import WorldTrackerConfig
from .helpers import cosine_similarity
from .kalman import KalmanState, mahalanobis2_position

GATE_INF: float = 1.0e9  # sentinel for "do not match"


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
) -> float:
    """Return finite cost in [0, GATE_INF). GATE_INF means do not match."""
    # 1. Geometric gate.
    d2 = mahalanobis2_position(
        ph_state,
        obs_floor_x_m,
        obs_floor_y_m,
        cfg.observation_noise_m,
    )
    if d2 > cfg.gate_chi2:
        return GATE_INF
    geo_cost = d2 / cfg.gate_chi2  # [0, 1]

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
    if ph_gallery_mean is not None and obs_embedding is not None:
        sim = cosine_similarity(ph_gallery_mean, obs_embedding)
        app_cost = max(0.0, min(1.0, 1.0 - sim))

    # 4. Height plausibility.
    height_cost = 0.0
    if obs_height_estimate_m is not None and ph_height_m is not None:
        z = (obs_height_estimate_m - ph_height_m) / cfg.height_sigma_m
        height_cost = min(1.0, 0.5 * z * z)

    return cfg.alpha_geo * geo_cost + cfg.alpha_app * app_cost + cfg.alpha_height * height_cost
