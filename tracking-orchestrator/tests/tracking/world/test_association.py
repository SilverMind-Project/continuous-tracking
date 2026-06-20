"""Tests for Hungarian association (app/tracking/world/association.py)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.tracking.world.association import associate
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.kalman import initialize


def _now() -> datetime:
    return datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)


def _make_state(x: float = 0.0, y: float = 0.0) -> object:
    return initialize(x, y, 0.5, 1.0, _now())


class TestAssociate:
    def test_empty_phs_returns_all_unmatched(self) -> None:
        cfg = WorldTrackerConfig()
        result = associate(
            ph_states=[],
            ph_gallery_means=[],
            ph_identity_ids=[],
            ph_heights=[],
            obs_floor_points=[(1.0, 1.0)],
            obs_embeddings=[None],
            obs_face_person_ids=[None],
            obs_face_confidences=[0.0],
            obs_height_estimates=[None],
            cfg=cfg,
        )
        assert result.matched == []
        assert result.unmatched_phs == []
        assert result.unmatched_obs == [0]

    def test_empty_observations_returns_all_unmatched(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        result = associate(
            ph_states=[state],
            ph_gallery_means=[None],
            ph_identity_ids=[None],
            ph_heights=[None],
            obs_floor_points=[],
            obs_embeddings=[],
            obs_face_person_ids=[],
            obs_face_confidences=[],
            obs_height_estimates=[],
            cfg=cfg,
        )
        assert result.matched == []
        assert result.unmatched_phs == [0]
        assert result.unmatched_obs == []

    def test_one_ph_one_observation_matches(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        result = associate(
            ph_states=[state],
            ph_gallery_means=[None],
            ph_identity_ids=[None],
            ph_heights=[None],
            obs_floor_points=[(0.1, 0.1)],
            obs_embeddings=[None],
            obs_face_person_ids=[None],
            obs_face_confidences=[0.0],
            obs_height_estimates=[None],
            cfg=cfg,
        )
        assert len(result.matched) == 1
        assert result.matched[0] == (0, 0)

    def test_distant_observation_goes_unmatched(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        result = associate(
            ph_states=[state],
            ph_gallery_means=[None],
            ph_identity_ids=[None],
            ph_heights=[None],
            obs_floor_points=[(100.0, 100.0)],
            obs_embeddings=[None],
            obs_face_person_ids=[None],
            obs_face_confidences=[0.0],
            obs_height_estimates=[None],
            cfg=cfg,
        )
        assert result.matched == []
        assert result.unmatched_obs == [0]

    def test_two_observations_same_ph_only_best_matches(self) -> None:
        cfg = WorldTrackerConfig()
        state = _make_state(0.0, 0.0)
        result = associate(
            ph_states=[state],
            ph_gallery_means=[None],
            ph_identity_ids=[None],
            ph_heights=[None],
            obs_floor_points=[(0.1, 0.1), (0.15, 0.15)],
            obs_embeddings=[None, None],
            obs_face_person_ids=[None, None],
            obs_face_confidences=[0.0, 0.0],
            obs_height_estimates=[None, None],
            cfg=cfg,
        )
        assert len(result.matched) == 1

    def test_two_phs_two_observations_both_match(self) -> None:
        cfg = WorldTrackerConfig()
        ph0 = _make_state(0.0, 0.0)
        ph1 = _make_state(5.0, 0.0)
        result = associate(
            ph_states=[ph0, ph1],
            ph_gallery_means=[None, None],
            ph_identity_ids=[None, None],
            ph_heights=[None, None],
            obs_floor_points=[(0.1, 0.1), (5.1, 0.1)],
            obs_embeddings=[None, None],
            obs_face_person_ids=[None, None],
            obs_face_confidences=[0.0, 0.0],
            obs_height_estimates=[None, None],
            cfg=cfg,
        )
        assert len(result.matched) == 2


class TestM03AssociationIntegrity:
    """M03: fail-closed gating produces clean unmatched + a reason tally."""

    def _one_ph_one_obs(self, cfg: WorldTrackerConfig, obs_point, **extra):
        return associate(
            ph_states=[_make_state(0.0, 0.0)],
            ph_gallery_means=[None],
            ph_identity_ids=[None],
            ph_heights=[None],
            obs_floor_points=[obs_point],
            obs_embeddings=[None],
            obs_face_person_ids=[None],
            obs_face_confidences=[0.0],
            obs_height_estimates=[None],
            cfg=cfg,
            **extra,
        )

    def test_non_finite_obs_unmatched_without_exception(self) -> None:
        cfg = WorldTrackerConfig()
        result = self._one_ph_one_obs(cfg, (float("nan"), 0.0))
        assert result.matched == []
        assert result.unmatched_phs == [0]
        assert result.unmatched_obs == [0]
        assert result.rejection_reasons.get("invalid_point", 0) == 1

    def test_all_pairs_gated_yields_no_least_bad_assignment(self) -> None:
        cfg = WorldTrackerConfig()
        # Observation 50 m away: geometrically gated for the single PH.
        result = self._one_ph_one_obs(cfg, (50.0, 0.0))
        assert result.matched == []
        assert result.unmatched_phs == [0]
        assert result.unmatched_obs == [0]
        assert result.rejection_reasons.get("geometric_gate", 0) == 1

    def test_reason_tally_counts_every_gated_pair(self) -> None:
        cfg = WorldTrackerConfig()
        # 2 PHs near origin, 1 far observation → 2 gated (geometric) pairs.
        result = associate(
            ph_states=[_make_state(0.0, 0.0), _make_state(0.5, 0.0)],
            ph_gallery_means=[None, None],
            ph_identity_ids=[None, None],
            ph_heights=[None, None],
            obs_floor_points=[(40.0, 0.0)],
            obs_embeddings=[None],
            obs_face_person_ids=[None],
            obs_face_confidences=[0.0],
            obs_height_estimates=[None],
            cfg=cfg,
        )
        assert result.matched == []
        assert result.rejection_reasons.get("geometric_gate", 0) == 2

    def test_valid_pair_records_no_rejection(self) -> None:
        cfg = WorldTrackerConfig()
        result = self._one_ph_one_obs(cfg, (0.1, 0.0))
        assert result.matched == [(0, 0)]
        assert result.rejection_reasons == {}
