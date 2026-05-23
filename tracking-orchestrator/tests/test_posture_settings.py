"""Tests for posture strategy settings."""

from __future__ import annotations

import pytest

from app.config import Settings


def _reload_with_env(monkeypatch) -> Settings:
    s = Settings()
    s.reload()
    return s


def test_depth_slow_path_disabled_by_default():
    s = Settings.from_dict({"pipeline": {"posture": {"depth_slow_path_enabled": False}}})
    assert s.as_bool("pipeline.posture.depth_slow_path_enabled") is False


def test_depth_slow_path_reads_from_env(monkeypatch):
    monkeypatch.setenv("POSTURE_DEPTH_SLOW_PATH_ENABLED", "true")
    monkeypatch.setenv("POSTURE_DEPTH_SLOW_PATH_MIN_INTERVAL_S", "30.0")
    s = _reload_with_env(monkeypatch)
    assert s.as_bool("pipeline.posture.depth_slow_path_enabled") is True
    interval = s.as_float("pipeline.posture.depth_slow_path_min_interval_s")
    assert interval == pytest.approx(30.0)


def test_depth_slow_path_max_age_reads_from_env(monkeypatch):
    monkeypatch.setenv("POSTURE_DEPTH_SLOW_PATH_MAX_AGE_S", "120.0")
    s = _reload_with_env(monkeypatch)
    max_age = s.as_float("pipeline.posture.depth_slow_path_max_age_s")
    assert max_age == pytest.approx(120.0)


def test_depth_slow_path_missing_setting_raises():
    s = Settings.from_dict({"pipeline": {"posture": {}}})
    with pytest.raises(KeyError):
        s.require("pipeline.posture.depth_slow_path_min_interval_s")
