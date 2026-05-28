"""WT6: sanity tests verifying orphaned settings.yaml sections are absent.

Acts as a regression guard against the deleted keys creeping back.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def _load_yaml() -> dict[str, object]:
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f) or {}


class TestSettingsNoOrphanedKeys:
    def test_settings_yaml_has_no_tracklet_section(self) -> None:
        data = _load_yaml()
        assert "tracklet" not in data, "tracklet section was deleted in WT6 and must not reappear"

    def test_settings_yaml_has_no_cross_camera_section(self) -> None:
        data = _load_yaml()
        assert "cross_camera" not in data, (
            "cross_camera section was deleted in WT6 and must not reappear"
        )

    def test_settings_yaml_has_no_stream_maxlen(self) -> None:
        data = _load_yaml()
        redis_section = data.get("redis", {})
        assert isinstance(redis_section, dict)
        assert "stream_maxlen" not in redis_section, (
            "redis.stream_maxlen was removed in WT6 (per-publisher defaults are used)"
        )

    def test_world_tracker_has_no_enabled(self) -> None:
        data = _load_yaml()
        wt = data.get("world_tracker", {})
        assert isinstance(wt, dict)
        assert "enabled" not in wt, (
            "world_tracker.enabled was removed in WT6 (config has no enabled field)"
        )
