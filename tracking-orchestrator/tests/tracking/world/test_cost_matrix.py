"""Tests for cost matrix (app/tracking/world/cost_matrix.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.cost_matrix import GATE_INF, pair_cost
from app.tracking.world.kalman import initialize, isotropic_cov


def _now() -> datetime:
    return datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)


def _make_state(x: float = 0.0, y: float = 0.0) -> object:
    return initialize(x, y, 0.5, 1.0, _now())


class TestPairCost:
    def test_close_observation_low_cost(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        cost = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=None,
            obs_floor_x_m=0.1,
            obs_floor_y_m=0.1,
            obs_embedding=None,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=None,
            cfg=cfg,
        )
        assert cost < 0.5

    def test_distant_observation_gated(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        cost = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=None,
            obs_floor_x_m=50.0,
            obs_floor_y_m=50.0,
            obs_embedding=None,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=None,
            cfg=cfg,
        )
        assert cost == GATE_INF

    def test_identity_conflict_gates(self) -> None:
        cfg = WorldTrackerConfig(face_conflict_threshold=0.70)
        state = _make_state(0.0, 0.0)
        cost = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id="alice",
            ph_height_m=None,
            obs_floor_x_m=0.1,
            obs_floor_y_m=0.1,
            obs_embedding=None,
            obs_face_anchor_person_id="bob",
            obs_face_anchor_confidence=0.85,
            obs_height_estimate_m=None,
            cfg=cfg,
        )
        assert cost == GATE_INF

    def test_identity_conflict_below_threshold_passes(self) -> None:
        cfg = WorldTrackerConfig(face_conflict_threshold=0.70)
        state = _make_state(0.0, 0.0)
        cost = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id="alice",
            ph_height_m=None,
            obs_floor_x_m=0.1,
            obs_floor_y_m=0.1,
            obs_embedding=None,
            obs_face_anchor_person_id="bob",
            obs_face_anchor_confidence=0.60,
            obs_height_estimate_m=None,
            cfg=cfg,
        )
        assert cost < GATE_INF

    def test_height_penalty_increases_with_discrepancy(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        cost_small = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=1.70,
            obs_floor_x_m=0.0,
            obs_floor_y_m=0.0,
            obs_embedding=None,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=1.72,
            cfg=cfg,
        )
        cost_large = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=1.70,
            obs_floor_x_m=0.0,
            obs_floor_y_m=0.0,
            obs_embedding=None,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=2.00,
            cfg=cfg,
        )
        assert cost_large > cost_small

    def test_appearance_cost_with_embeddings(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        emb = [1.0, 0.0, 0.0]
        cost = pair_cost(
            ph_state=state,
            ph_gallery_mean=emb,
            ph_current_identity_id=None,
            ph_height_m=None,
            obs_floor_x_m=0.0,
            obs_floor_y_m=0.0,
            obs_embedding=emb,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=None,
            cfg=cfg,
        )
        assert cost < 0.1

    def test_appearance_neutral_when_one_side_missing(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        cost = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=None,
            obs_floor_x_m=0.0,
            obs_floor_y_m=0.0,
            obs_embedding=[1.0, 0.0, 0.0],
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=None,
            cfg=cfg,
        )
        assert 0.15 < cost < 0.25


class TestGeometryAwareGate:
    """M04: obs_cov_m2 gates more permissively when observation uncertainty is large."""

    def test_uncertain_obs_admitted_where_fixed_gate_rejects(self) -> None:
        """An observation far enough to exceed the fixed-R gate is admitted when R is large."""
        cfg = WorldTrackerConfig(gate_chi2=9.21, observation_noise_m=0.25)
        state = _make_state(0.0, 0.0)

        # Place observation at distance that exceeds the fixed isotropic gate.
        obs_x, obs_y = 0.8, 0.0  # 0.8 m from origin

        cost_fixed = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=None,
            obs_floor_x_m=obs_x,
            obs_floor_y_m=obs_y,
            obs_embedding=None,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=None,
            cfg=cfg,
        )

        # Same observation but with a very large covariance (uncertain measurement).
        large_cov = 4.0 * np.eye(2, dtype=np.float64)  # 2 m sigma
        cost_uncertain = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=None,
            obs_floor_x_m=obs_x,
            obs_floor_y_m=obs_y,
            obs_embedding=None,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=None,
            cfg=cfg,
            obs_cov_m2=large_cov,
        )

        # The fixed-R gate should reject or assign high cost; uncertain-R should admit.
        assert cost_uncertain < cost_fixed, (
            f"uncertain-obs cost {cost_uncertain:.3f} should be < fixed-obs cost {cost_fixed:.3f}"
        )

    def test_obs_cov_none_falls_back_to_isotropic(self) -> None:
        """When obs_cov_m2=None, pair_cost behaves identically to the old fixed-R path."""
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        cost_none = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=None,
            obs_floor_x_m=0.1,
            obs_floor_y_m=0.0,
            obs_embedding=None,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=None,
            cfg=cfg,
            obs_cov_m2=None,
        )
        cost_explicit = pair_cost(
            ph_state=state,
            ph_gallery_mean=None,
            ph_current_identity_id=None,
            ph_height_m=None,
            obs_floor_x_m=0.1,
            obs_floor_y_m=0.0,
            obs_embedding=None,
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_height_estimate_m=None,
            cfg=cfg,
            obs_cov_m2=isotropic_cov(cfg.observation_noise_m),
        )
        assert abs(cost_none - cost_explicit) < 1e-9
