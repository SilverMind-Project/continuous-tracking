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


class TestFailClosed:
    """M03: no NaN/invalid input ever reaches the Hungarian solver."""

    def _base_kwargs(self, cfg: WorldTrackerConfig, state: object) -> dict:
        return {
            "ph_state": state,
            "ph_gallery_mean": None,
            "ph_current_identity_id": None,
            "ph_height_m": None,
            "obs_floor_x_m": 0.1,
            "obs_floor_y_m": 0.0,
            "obs_embedding": None,
            "obs_face_anchor_person_id": None,
            "obs_face_anchor_confidence": 0.0,
            "obs_height_estimate_m": None,
            "cfg": cfg,
        }

    def test_non_finite_point_gates_out(self) -> None:
        from app.tracking.world.cost_matrix import RejectionReason, pair_cost_detail

        cfg = WorldTrackerConfig()
        kwargs = self._base_kwargs(cfg, _make_state())
        kwargs["obs_floor_x_m"] = np.nan
        cost, reason = pair_cost_detail(**kwargs)
        assert cost == GATE_INF
        assert reason is RejectionReason.INVALID_POINT

    def test_non_finite_covariance_gates_out(self) -> None:
        from app.tracking.world.cost_matrix import RejectionReason, pair_cost_detail

        cfg = WorldTrackerConfig()
        kwargs = self._base_kwargs(cfg, _make_state())
        kwargs["obs_cov_m2"] = np.array([[np.nan, 0.0], [0.0, 0.25]])
        cost, reason = pair_cost_detail(**kwargs)
        assert cost == GATE_INF
        assert reason is RejectionReason.INVALID_COVARIANCE

    def test_non_psd_covariance_gates_out(self) -> None:
        from app.tracking.world.cost_matrix import RejectionReason, pair_cost_detail

        cfg = WorldTrackerConfig()
        kwargs = self._base_kwargs(cfg, _make_state())
        kwargs["obs_cov_m2"] = np.array([[0.0, 1.0], [1.0, 0.0]])  # indefinite
        cost, reason = pair_cost_detail(**kwargs)
        assert cost == GATE_INF
        assert reason is RejectionReason.INVALID_COVARIANCE

    def test_over_cap_covariance_gates_out(self) -> None:
        from app.tracking.world.cost_matrix import RejectionReason, pair_cost_detail

        cfg = WorldTrackerConfig()
        kwargs = self._base_kwargs(cfg, _make_state())
        kwargs["obs_cov_m2"] = 1.0e6 * np.eye(2)  # trace far over cap
        cost, reason = pair_cost_detail(**kwargs)
        assert cost == GATE_INF
        assert reason is RejectionReason.INVALID_COVARIANCE

    def test_large_uncertainty_cannot_rescue_implausible_jump(self) -> None:
        """A 20 m jump with hugely inflated covariance is still gated out."""
        cfg = WorldTrackerConfig()
        kwargs = self._base_kwargs(cfg, _make_state(0.0, 0.0))
        kwargs["obs_floor_x_m"] = 20.0
        kwargs["obs_cov_m2"] = 1.0e6 * np.eye(2)  # would otherwise pass the gate
        assert pair_cost(**kwargs) == GATE_INF

    def test_legitimate_large_cov_still_matchable(self) -> None:
        """The 2 m-sigma fixture covariance (trace 8) remains under the cap."""
        cfg = WorldTrackerConfig()
        kwargs = self._base_kwargs(cfg, _make_state(0.0, 0.0))
        kwargs["obs_floor_x_m"] = 0.8
        kwargs["obs_cov_m2"] = 4.0 * np.eye(2)
        assert pair_cost(**kwargs) < GATE_INF

    def test_validation_can_be_disabled_for_ab(self) -> None:
        """With enable_covariance_validation off, only non-finite guards remain."""
        cfg = WorldTrackerConfig(enable_covariance_validation=False)
        kwargs = self._base_kwargs(cfg, _make_state(0.0, 0.0))
        kwargs["obs_floor_x_m"] = 0.1
        kwargs["obs_cov_m2"] = 1.0e6 * np.eye(2)  # over-cap but finite
        # No longer rejected for being over-cap; the wide cov admits the match.
        assert pair_cost(**kwargs) < GATE_INF


class TestTypedIdentityEvidence:
    """M03 task 5: operator authority hard-gates; verified-ReID only costs."""

    def _kwargs(self, cfg: WorldTrackerConfig) -> dict:
        return {
            "ph_state": _make_state(0.0, 0.0),
            "ph_gallery_mean": None,
            "ph_current_identity_id": "grandma",
            "ph_height_m": None,
            "obs_floor_x_m": 0.1,
            "obs_floor_y_m": 0.0,
            "obs_embedding": None,
            "obs_height_estimate_m": None,
            "cfg": cfg,
        }

    def test_qualifying_direct_conflict_hard_gates(self) -> None:
        from app.tracking.world.cost_matrix import RejectionReason, pair_cost_detail

        cfg = WorldTrackerConfig()
        cost, reason = pair_cost_detail(
            **self._kwargs(cfg),
            obs_face_anchor_person_id="grandpa",
            obs_face_anchor_confidence=0.95,  # >= face_conflict_threshold
        )
        assert cost == GATE_INF
        assert reason is RejectionReason.IDENTITY_CONFLICT

    def test_subthreshold_conflict_does_not_hard_gate(self) -> None:
        cfg = WorldTrackerConfig()
        cost = pair_cost(
            **self._kwargs(cfg),
            obs_face_anchor_person_id="grandpa",
            obs_face_anchor_confidence=0.40,  # below threshold
        )
        assert cost < GATE_INF

    def test_operator_confirmed_identity_hard_gates_subthreshold_conflict(self) -> None:
        from app.tracking.world.cost_matrix import RejectionReason, pair_cost_detail

        cfg = WorldTrackerConfig()
        cost, reason = pair_cost_detail(
            **self._kwargs(cfg),
            obs_face_anchor_person_id="grandpa",
            obs_face_anchor_confidence=0.40,  # below threshold, but...
            ph_identity_is_operator_confirmed=True,  # operator authority is absolute
        )
        assert cost == GATE_INF
        assert reason is RejectionReason.IDENTITY_CONFLICT

    def test_verified_reid_disagreement_adds_cost_not_gate(self) -> None:
        cfg = WorldTrackerConfig(enable_reid_disagreement_cost=True, reid_disagreement_cost=0.6)
        baseline = pair_cost(
            **self._kwargs(cfg),
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_verified_reid_identity_id=None,
        )
        disagree = pair_cost(
            **self._kwargs(cfg),
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_verified_reid_identity_id="grandpa",  # != PH "grandma"
        )
        assert disagree < GATE_INF, "ReID disagreement must not hard-gate"
        assert abs((disagree - baseline) - 0.6) < 1e-9

    def test_verified_reid_disagreement_inert_when_flag_off(self) -> None:
        cfg = WorldTrackerConfig(enable_reid_disagreement_cost=False)
        baseline = pair_cost(
            **self._kwargs(cfg),
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_verified_reid_identity_id=None,
        )
        disagree = pair_cost(
            **self._kwargs(cfg),
            obs_face_anchor_person_id=None,
            obs_face_anchor_confidence=0.0,
            obs_verified_reid_identity_id="grandpa",
        )
        assert abs(disagree - baseline) < 1e-9
