"""Tests for posture strategy settings."""

from __future__ import annotations

import pytest

from app.config import Settings


def _reload_with_env(monkeypatch) -> Settings:
    s = Settings()
    s.reload()
    return s


def test_depth_slow_path_disabled_by_default():
    s = Settings.from_dict({"pipeline": {"posture": {}}})
    assert s.get("pipeline.posture.depth_slow_path_enabled", False) is False


def test_depth_slow_path_reads_from_env(monkeypatch):
    monkeypatch.setenv("POSTURE_DEPTH_SLOW_PATH_ENABLED", "true")
    monkeypatch.setenv("POSTURE_DEPTH_SLOW_PATH_MIN_INTERVAL_S", "30.0")
    s = _reload_with_env(monkeypatch)
    enabled = str(s.get("pipeline.posture.depth_slow_path_enabled", "false")).lower()
    assert enabled in ("1", "true", "yes")
    interval = float(s.get("pipeline.posture.depth_slow_path_min_interval_s", "15.0"))
    assert interval == pytest.approx(30.0)


def test_depth_slow_path_max_age_reads_from_env(monkeypatch):
    monkeypatch.setenv("POSTURE_DEPTH_SLOW_PATH_MAX_AGE_S", "120.0")
    s = _reload_with_env(monkeypatch)
    max_age = float(s.get("pipeline.posture.depth_slow_path_max_age_s", "60.0"))
    assert max_age == pytest.approx(120.0)


def test_depth_slow_path_defaults_when_not_set():
    s = Settings.from_dict({"pipeline": {"posture": {}}})
    interval = float(s.get("pipeline.posture.depth_slow_path_min_interval_s", "15.0"))
    assert interval == pytest.approx(15.0)
    max_age = float(s.get("pipeline.posture.depth_slow_path_max_age_s", "60.0"))
    assert max_age == pytest.approx(60.0)
