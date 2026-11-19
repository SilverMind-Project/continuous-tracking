"""Unit tests for AdapterRegistry fail-fast startup validation."""

from __future__ import annotations

import pytest

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.registry import build_default_registry
from rf_ingress.config import AdapterConfig
from rf_ingress.site_config import InMemorySiteConfigProvider
from tests.conftest import SCENARIOS_DIR


def test_unknown_adapter_kind_fails_startup_clearly() -> None:
    registry = build_default_registry()
    deps = AdapterDeps(site_config=InMemorySiteConfigProvider())
    cfg = AdapterConfig(id="bad-1", kind="nonexistent_radio")
    with pytest.raises(ValueError, match="Unknown adapter kind 'nonexistent_radio'"):
        registry.build(cfg, deps)


def test_unknown_payload_profile_fails_startup_clearly() -> None:
    registry = build_default_registry()
    deps = AdapterDeps(site_config=InMemorySiteConfigProvider())
    cfg = AdapterConfig(
        id="sim-bad",
        kind="simulator",
        payload_profile="unknown_profile_v9",
        scenario=str(SCENARIOS_DIR / "walker_uwb.yaml"),
    )
    with pytest.raises(ValueError, match="Unknown simulator payload_profile"):
        registry.build(cfg, deps)


def test_valid_simulator_config_builds_adapter() -> None:
    registry = build_default_registry()
    deps = AdapterDeps(site_config=InMemorySiteConfigProvider())
    cfg = AdapterConfig(
        id="sim-1",
        kind="simulator",
        payload_profile="scenario_yaml_v1",
        scenario=str(SCENARIOS_DIR / "walker_uwb.yaml"),
    )
    adapter = registry.build(cfg, deps)
    assert adapter.adapter_id == "sim-1"
