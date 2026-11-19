"""Unit tests for rf-ingress YAML configuration loading and env interpolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rf_ingress.config import ConfigInterpolationError, load_settings

_DEFAULT_SETTINGS = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def test_load_default_settings_with_env_vars() -> None:
    settings = load_settings(
        _DEFAULT_SETTINGS,
        env={
            "REDIS_URL": "redis://localhost:6379/0",
            "CC_API_URL": "http://localhost:8000",
            "CC_INGRESS_API_KEY": "test-key",
        },
    )
    assert settings.service.port == 8095
    assert settings.redis.fixes_stream == "wearable.fixes"
    assert settings.redis.capabilities_stream == "wearable.capabilities"
    assert settings.redis.retention_hours == 72.0
    assert settings.solver.max_sigma_m is None
    assert settings.adapters == []


def test_missing_required_env_var_fails_fast() -> None:
    with pytest.raises(ConfigInterpolationError, match="REDIS_URL"):
        load_settings(_DEFAULT_SETTINGS, env={"CC_API_URL": "http://x", "CC_INGRESS_API_KEY": "k"})


def test_defaulted_env_interpolation_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
redis:
  url: "${REDIS_URL:-redis://localhost:6379/0}"
site_config:
  cc_api_url: "http://localhost:8000"
  api_key: "key"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigInterpolationError, match="Defaulted env interpolation"):
        load_settings(bad, env={})
