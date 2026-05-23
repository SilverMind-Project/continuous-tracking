"""Test that _build_*_config() helpers produce correct config objects from settings."""

from dataclasses import asdict

import pytest

from app.config import Settings
from app.main import (
    _build_cross_cam_config,
    _build_resolver_config,
    _build_sampler_config,
    _build_tracklet_config,
)
from app.sampling.keyframe_sampler import SamplerConfig
from app.tracking.cross_camera import CrossCamConfig
from app.tracking.identity_resolver import ResolverConfig
from app.tracking.tracklet_manager import TrackletConfig

# ---------------------------------------------------------------------------
# ResolverConfig
# ---------------------------------------------------------------------------


def test_resolver_config_defaults_match_dataclass():
    rc = ResolverConfig()
    s = Settings.from_dict({"resolver": asdict(rc)})
    cfg = _build_resolver_config(s)
    assert cfg.commit_prob == pytest.approx(rc.commit_prob)
    assert cfg.commit_margin == pytest.approx(rc.commit_margin)
    assert cfg.reid_decision_sim == pytest.approx(rc.reid_decision_sim)
    assert cfg.revision_horizon_s == pytest.approx(rc.revision_horizon_s)
    assert cfg.max_revisions_per_gt_per_minute == rc.max_revisions_per_gt_per_minute
    assert cfg.unknown_mass == pytest.approx(rc.unknown_mass)
    assert cfg.prior_weight == pytest.approx(rc.prior_weight)
    assert cfg.face_weight_multiplier == pytest.approx(rc.face_weight_multiplier)
    assert cfg.commit_prob_dense == pytest.approx(rc.commit_prob_dense)
    assert cfg.commit_margin_dense == pytest.approx(rc.commit_margin_dense)
    assert cfg.prior_maintenance_max_age_s == pytest.approx(rc.prior_maintenance_max_age_s)
    assert cfg.identified_entry_boost_min_sim == pytest.approx(rc.identified_entry_boost_min_sim)
    assert cfg.identified_entry_min_likelihood == pytest.approx(rc.identified_entry_min_likelihood)
    assert cfg.enable_embedding_coherence_boost == rc.enable_embedding_coherence_boost
    assert cfg.embedding_coherence_window == rc.embedding_coherence_window
    assert cfg.embedding_coherence_min_sim == pytest.approx(rc.embedding_coherence_min_sim)
    assert cfg.embedding_coherence_boost == pytest.approx(rc.embedding_coherence_boost)
    assert cfg.face_commit_min_confidence == pytest.approx(rc.face_commit_min_confidence)
    assert cfg.face_lock_maintenance_max_age_s == pytest.approx(rc.face_lock_maintenance_max_age_s)
    assert cfg.cross_gt_face_propagation_threshold == pytest.approx(
        rc.cross_gt_face_propagation_threshold
    )
    assert cfg.cross_gt_face_propagation_max_gts == rc.cross_gt_face_propagation_max_gts


def test_resolver_config_built_from_settings():
    values = asdict(ResolverConfig())
    values.update(
        {
            "commit_prob": "0.80",
            "commit_margin": "0.10",
            "prior_weight": "0.70",
            "face_weight_multiplier": "5.0",
            "face_commit_min_confidence": "0.85",
        }
    )
    s = Settings.from_dict({"resolver": values})
    cfg = _build_resolver_config(s)
    assert cfg.commit_prob == pytest.approx(0.80)
    assert cfg.commit_margin == pytest.approx(0.10)
    assert cfg.prior_weight == pytest.approx(0.70)
    assert cfg.face_weight_multiplier == pytest.approx(5.0)
    assert cfg.face_commit_min_confidence == pytest.approx(0.85)
    assert cfg.reid_decision_sim == pytest.approx(0.70)
    assert cfg.revision_horizon_s == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# TrackletConfig
# ---------------------------------------------------------------------------


def test_tracklet_config_defaults_match_dataclass():
    tc = TrackletConfig()
    s = Settings.from_dict(
        {
            "tracklet": asdict(tc),
            "pipeline": {"tracker": {"min_frames_to_publish": tc.min_frames_to_publish}},
        }
    )
    cfg = _build_tracklet_config(s)
    assert cfg.min_hit_ratio == pytest.approx(tc.min_hit_ratio)
    assert cfg.close_grace_frames == tc.close_grace_frames
    assert cfg.gallery_min_quality == pytest.approx(tc.gallery_min_quality)
    assert cfg.gallery_max_per_tracklet == tc.gallery_max_per_tracklet
    assert cfg.min_detection_confidence == pytest.approx(tc.min_detection_confidence)
    assert cfg.enabled == tc.enabled
    assert cfg.min_frames_to_publish == tc.min_frames_to_publish


def test_tracklet_config_built_from_settings():
    values = asdict(TrackletConfig())
    values.update(
        {
            "min_hit_ratio": "0.7",
            "close_grace_frames": "30",
            "gallery_min_quality": "0.6",
            "gallery_max_per_tracklet": "30",
            "min_detection_confidence": "0.4",
            "enabled": "false",
        }
    )
    s = Settings.from_dict(
        {
            "tracklet": values,
            "pipeline": {"tracker": {"min_frames_to_publish": 5}},
        }
    )
    cfg = _build_tracklet_config(s)
    assert cfg.min_hit_ratio == pytest.approx(0.7)
    assert cfg.close_grace_frames == 30
    assert cfg.gallery_min_quality == pytest.approx(0.6)
    assert cfg.gallery_max_per_tracklet == 30
    assert cfg.min_detection_confidence == pytest.approx(0.4)
    assert cfg.enabled is False
    assert cfg.min_frames_to_publish == 5


# ---------------------------------------------------------------------------
# CrossCamConfig
# ---------------------------------------------------------------------------


def test_cross_cam_config_defaults_match_dataclass():
    cc = CrossCamConfig()
    s = Settings.from_dict({"cross_camera": asdict(cc)})
    cfg = _build_cross_cam_config(s)
    assert cfg.alpha == pytest.approx(cc.alpha)
    assert cfg.floor_sigma_m == pytest.approx(cc.floor_sigma_m)
    assert cfg.max_floor_distance_m == pytest.approx(cc.max_floor_distance_m)
    assert cfg.min_link_score == pytest.approx(cc.min_link_score)
    assert cfg.unknown_merge_appearance_threshold == pytest.approx(
        cc.unknown_merge_appearance_threshold
    )
    assert cfg.within_group_min_score == pytest.approx(cc.within_group_min_score)
    assert cfg.inter_gt_consolidation_appearance_threshold == pytest.approx(
        cc.inter_gt_consolidation_appearance_threshold
    )
    assert cfg.known_identity_reentry_threshold == pytest.approx(
        cc.known_identity_reentry_threshold
    )
    assert cfg.same_camera_reentry_max_gap_s == pytest.approx(cc.same_camera_reentry_max_gap_s)


def test_cross_cam_config_built_from_settings():
    values = asdict(CrossCamConfig())
    values.update(
        {
            "alpha": "0.80",
            "floor_sigma_m": "3.0",
            "min_link_score": "0.60",
            "unknown_merge_appearance_threshold": "0.95",
        }
    )
    s = Settings.from_dict({"cross_camera": values})
    cfg = _build_cross_cam_config(s)
    assert cfg.alpha == pytest.approx(0.80)
    assert cfg.floor_sigma_m == pytest.approx(3.0)
    assert cfg.min_link_score == pytest.approx(0.60)
    assert cfg.unknown_merge_appearance_threshold == pytest.approx(0.95)
    assert cfg.max_floor_distance_m == pytest.approx(8.0)
    assert cfg.within_group_min_score == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# SamplerConfig
# ---------------------------------------------------------------------------


def test_sampler_config_defaults_match_dataclass():
    sc = SamplerConfig()
    s = Settings.from_dict({"sampler": asdict(sc)})
    cfg = _build_sampler_config(s)
    assert cfg.keyframe_min_interval_s == pytest.approx(sc.keyframe_min_interval_s)
    assert cfg.periodic_expires_hours == sc.periodic_expires_hours
    assert cfg.trigger_expires_days == sc.trigger_expires_days


def test_sampler_config_built_from_settings():
    s = Settings.from_dict(
        {
            "sampler": {
                "keyframe_min_interval_s": "15.0",
                "periodic_expires_hours": "48",
                "trigger_expires_days": "14",
            }
        }
    )
    cfg = _build_sampler_config(s)
    assert cfg.keyframe_min_interval_s == pytest.approx(15.0)
    assert cfg.periodic_expires_hours == 48
    assert cfg.trigger_expires_days == 14


def test_incomplete_config_raises_instead_of_using_hidden_defaults():
    with pytest.raises(KeyError):
        _build_resolver_config(Settings.from_dict({"resolver": {}}))

    with pytest.raises(KeyError):
        _build_cross_cam_config(Settings.from_dict({"cross_camera": {}}))


# ---------------------------------------------------------------------------
# Settings typed accessors
# ---------------------------------------------------------------------------


def test_settings_typed_accessors_require_and_convert_values():
    s = Settings.from_dict(
        {
            "redis": {"ack_ttl_seconds": "300"},
            "pipeline": {"allow_skeleton": "true"},
            "resolver": {"commit_prob": "0.75"},
        }
    )

    assert s.as_int("redis.ack_ttl_seconds") == 300
    assert s.as_bool("pipeline.allow_skeleton") is True
    assert s.section("resolver").as_float("commit_prob") == pytest.approx(0.75)


def test_settings_typed_accessors_raise_on_missing_or_invalid_values():
    s = Settings.from_dict({"redis": {"ack_ttl_seconds": "not-an-int"}})

    with pytest.raises(KeyError):
        s.as_int("redis.missing")

    with pytest.raises(ValueError, match=r"redis\.ack_ttl_seconds"):
        s.as_int("redis.ack_ttl_seconds")
