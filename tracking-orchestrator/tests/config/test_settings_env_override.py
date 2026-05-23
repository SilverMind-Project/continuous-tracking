"""Test that env vars override settings.yaml defaults via ${VAR:-default} interpolation."""

import pytest

from app.config import Settings


def _reload_with_env(monkeypatch) -> Settings:
    """Reload settings from disk with monkeypatched env, then return a fresh instance."""
    s = Settings()
    s.reload()
    return s


def test_resolver_commit_prob_reads_from_env(monkeypatch):
    monkeypatch.setenv("RESOLVER_COMMIT_PROB", "0.99")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("resolver.commit_prob")) == pytest.approx(0.99)


def test_resolver_commit_margin_reads_from_env(monkeypatch):
    monkeypatch.setenv("RESOLVER_COMMIT_MARGIN", "0.25")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("resolver.commit_margin")) == pytest.approx(0.25)


def test_resolver_revision_horizon_reads_from_env(monkeypatch):
    monkeypatch.setenv("RESOLVER_REVISION_HORIZON_S", "300.0")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("resolver.revision_horizon_s")) == pytest.approx(300.0)


def test_resolver_unknown_mass_reads_from_env(monkeypatch):
    monkeypatch.setenv("RESOLVER_UNKNOWN_MASS", "0.15")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("resolver.unknown_mass")) == pytest.approx(0.15)


def test_resolver_prior_weight_reads_from_env(monkeypatch):
    monkeypatch.setenv("RESOLVER_PRIOR_WEIGHT", "0.80")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("resolver.prior_weight")) == pytest.approx(0.80)


def test_resolver_face_weight_multiplier_reads_from_env(monkeypatch):
    monkeypatch.setenv("RESOLVER_FACE_WEIGHT_MULTIPLIER", "5.0")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("resolver.face_weight_multiplier")) == pytest.approx(5.0)


def test_resolver_face_commit_min_confidence_reads_from_env(monkeypatch):
    monkeypatch.setenv("RESOLVER_FACE_COMMIT_MIN_CONFIDENCE", "0.85")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("resolver.face_commit_min_confidence")) == pytest.approx(0.85)


def test_tracklet_close_grace_frames_reads_from_env(monkeypatch):
    monkeypatch.setenv("TRACKLET_CLOSE_GRACE_FRAMES", "42")
    s = _reload_with_env(monkeypatch)
    assert int(s.require("tracklet.close_grace_frames")) == 42


def test_tracklet_gallery_min_quality_reads_from_env(monkeypatch):
    monkeypatch.setenv("TRACKLET_GALLERY_MIN_QUALITY", "0.75")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("tracklet.gallery_min_quality")) == pytest.approx(0.75)


def test_tracklet_gallery_max_per_tracklet_reads_from_env(monkeypatch):
    monkeypatch.setenv("TRACKLET_GALLERY_MAX_PER_TRACKLET", "50")
    s = _reload_with_env(monkeypatch)
    assert int(s.require("tracklet.gallery_max_per_tracklet")) == 50


def test_tracklet_min_detection_confidence_reads_from_env(monkeypatch):
    monkeypatch.setenv("TRACKLET_MIN_DETECTION_CONFIDENCE", "0.55")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("tracklet.min_detection_confidence")) == pytest.approx(0.55)


def test_cross_camera_alpha_reads_from_env(monkeypatch):
    monkeypatch.setenv("CC_ALPHA", "0.85")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("cross_camera.alpha")) == pytest.approx(0.85)


def test_cross_camera_min_link_score_reads_from_env(monkeypatch):
    monkeypatch.setenv("CC_MIN_LINK_SCORE", "0.70")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("cross_camera.min_link_score")) == pytest.approx(0.70)


def test_cross_camera_known_identity_reentry_threshold_reads_from_env(monkeypatch):
    monkeypatch.setenv("CC_KNOWN_IDENTITY_REENTRY_THRESHOLD", "0.80")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("cross_camera.known_identity_reentry_threshold")) == pytest.approx(0.80)


def test_cross_camera_same_camera_reentry_max_gap_s_reads_from_env(monkeypatch):
    monkeypatch.setenv("CC_SAME_CAMERA_REENTRY_MAX_GAP_S", "60.0")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("cross_camera.same_camera_reentry_max_gap_s")) == pytest.approx(60.0)


def test_sampler_keyframe_min_interval_s_reads_from_env(monkeypatch):
    monkeypatch.setenv("SAMPLER_KEYFRAME_MIN_INTERVAL_S", "15.0")
    s = _reload_with_env(monkeypatch)
    assert float(s.require("sampler.keyframe_min_interval_s")) == pytest.approx(15.0)


def test_sampler_periodic_expires_hours_reads_from_env(monkeypatch):
    monkeypatch.setenv("SAMPLER_PERIODIC_EXPIRES_HOURS", "48")
    s = _reload_with_env(monkeypatch)
    assert int(s.require("sampler.periodic_expires_hours")) == 48


def test_sampler_trigger_expires_days_reads_from_env(monkeypatch):
    monkeypatch.setenv("SAMPLER_TRIGGER_EXPIRES_DAYS", "14")
    s = _reload_with_env(monkeypatch)
    assert int(s.require("sampler.trigger_expires_days")) == 14
