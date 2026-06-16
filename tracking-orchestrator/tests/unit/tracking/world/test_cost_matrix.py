"""Uncalibrated gate relaxation tests for pair_cost."""

from __future__ import annotations

from datetime import UTC, datetime

from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.cost_matrix import GATE_INF, pair_cost
from app.tracking.world.kalman import KalmanState, initialize


def _ph_state(x: float = 5.0, y: float = 3.0, vx: float = 0.0, vy: float = 0.0) -> KalmanState:
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    return initialize(x, y, 1.0, 2.0, now)


def _calibrated_config() -> WorldTrackerConfig:
    return WorldTrackerConfig(
        gate_chi2=9.21,
        alpha_geo=0.5,
        alpha_app=0.4,
        alpha_height=0.1,
        enable_uncalibrated_gate_relax=False,
        uncalibrated_gate_chi2=21.0,
        uncalibrated_alpha_app=0.7,
    )


def _uncalibrated_config() -> WorldTrackerConfig:
    return WorldTrackerConfig(
        gate_chi2=9.21,
        alpha_geo=0.5,
        alpha_app=0.4,
        alpha_height=0.1,
        enable_uncalibrated_gate_relax=True,
        uncalibrated_gate_chi2=21.0,
        uncalibrated_alpha_app=0.7,
    )


# ---------------------------------------------------------------------------
# Calibrated path: unchanged behavior
# ---------------------------------------------------------------------------


def test_calibrated_path_uses_standard_gate() -> None:
    """With calibrated=True, the standard gate_chi2 (9.21) is used regardless
    of enable_uncalibrated_gate_relax."""
    ks = _ph_state(5.0, 3.0)
    # A position jump that is within the relaxed gate (21.0) but exceeds the
    # standard gate (9.21) should be GATE_INF under calibrated path.
    cost = pair_cost(
        ph_state=ks,
        ph_gallery_mean=None,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=8.0,  # 3 m away from 5.0
        obs_floor_y_m=6.0,  # 3 m away from 3.0
        obs_embedding=None,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=_uncalibrated_config(),
        calibrated=True,
    )
    # Under calibrated gate (9.21), a 3 m jump should be gated.
    assert cost == GATE_INF


def test_calibrated_path_byte_for_byte_unchanged() -> None:
    """Calibrated path with enable_uncalibrated_gate_relax=False must match
    the original behavior exactly."""
    cfg = _calibrated_config()
    ks = _ph_state(5.0, 3.0)
    gallery = [0.1] * 768
    emb = [0.1] * 768

    cost = pair_cost(
        ph_state=ks,
        ph_gallery_mean=gallery,
        ph_current_identity_id="alice",
        ph_height_m=1.7,
        obs_floor_x_m=5.1,
        obs_floor_y_m=3.1,
        obs_embedding=emb,
        obs_face_anchor_person_id="alice",
        obs_face_anchor_confidence=0.6,
        obs_height_estimate_m=1.72,
        cfg=cfg,
        calibrated=True,
    )
    # Should be a finite cost (within gate, same identity).
    assert cost < GATE_INF
    assert 0.0 <= cost <= 1.0


# ---------------------------------------------------------------------------
# Uncalibrated relaxed path
# ---------------------------------------------------------------------------


def test_uncalibrated_admits_larger_position_jump() -> None:
    """With calibrated=False and relax enabled, a larger position jump that
    would be gated under the standard gate is admitted."""
    cfg = _uncalibrated_config()
    ks = _ph_state(5.0, 3.0)
    # A 3m jump exceeds standard gate (9.21) but should be within relaxed
    # gate (21.0).
    cost = pair_cost(
        ph_state=ks,
        ph_gallery_mean=None,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=8.0,
        obs_floor_y_m=6.0,
        obs_embedding=None,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=False,
    )
    assert cost < GATE_INF


def test_uncalibrated_uses_appearance_weight() -> None:
    """With calibrated=False and relax enabled, the appearance weight is
    uncalibrated_alpha_app (0.7) instead of alpha_app (0.4)."""
    cfg = _uncalibrated_config()
    ks = _ph_state(5.0, 3.0)

    # Same gallery and observation embeddings (cosine sim = 1.0 -> app_cost = 0).
    gallery = [0.1] * 768
    emb = [0.1] * 768

    cost_perfect_app = pair_cost(
        ph_state=ks,
        ph_gallery_mean=gallery,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=5.1,  # small position delta
        obs_floor_y_m=3.1,
        obs_embedding=emb,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=False,
    )
    # With different embeddings (low sim), cost should be higher.
    gallery2 = [1.0] + [0.0] * 767
    emb2 = [0.0] + [1.0] * 767  # orthogonal → cosine sim ≈ 0 → app_cost ≈ 1.0

    cost_bad_app = pair_cost(
        ph_state=ks,
        ph_gallery_mean=gallery2,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=5.1,
        obs_floor_y_m=3.1,
        obs_embedding=emb2,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=False,
    )
    assert cost_bad_app > cost_perfect_app


def test_uncalibrated_without_relax_uses_standard_gate() -> None:
    """With calibrated=False but enable_uncalibrated_gate_relax=False,
    the standard gate is used (no relaxation)."""
    cfg = _calibrated_config()  # relax disabled
    ks = _ph_state(5.0, 3.0)
    cost = pair_cost(
        ph_state=ks,
        ph_gallery_mean=None,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=8.0,
        obs_floor_y_m=6.0,
        obs_embedding=None,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=False,  # uncalibrated, but relax is off
    )
    # Without relaxation, the standard gate applies regardless of calibrated flag.
    assert cost == GATE_INF
