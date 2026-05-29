"""Tests for posture strategy settings."""

import pytest

from app.config import Settings


def test_depth_slow_path_disabled_by_default():
    s = Settings.from_dict({"pipeline": {"posture": {"depth_slow_path_enabled": False}}})
    assert s.as_bool("pipeline.posture.depth_slow_path_enabled") is False


def test_depth_slow_path_reads_literal_settings():
    s = Settings.from_dict(
        {
            "pipeline": {
                "posture": {
                    "depth_slow_path_enabled": True,
                    "depth_slow_path_min_interval_s": 30.0,
                }
            }
        }
    )
    assert s.as_bool("pipeline.posture.depth_slow_path_enabled") is True
    interval = s.as_float("pipeline.posture.depth_slow_path_min_interval_s")
    assert interval == pytest.approx(30.0)


def test_depth_slow_path_max_age_reads_literal_settings():
    s = Settings.from_dict({"pipeline": {"posture": {"depth_slow_path_max_age_s": 120.0}}})
    max_age = s.as_float("pipeline.posture.depth_slow_path_max_age_s")
    assert max_age == pytest.approx(120.0)


def test_depth_slow_path_missing_setting_raises():
    s = Settings.from_dict({"pipeline": {"posture": {}}})
    with pytest.raises(KeyError):
        s.require("pipeline.posture.depth_slow_path_min_interval_s")
