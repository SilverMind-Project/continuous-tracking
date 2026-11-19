"""Unit tests for raw report JSONL recording, replay, and 60-minute cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.simulator import (
    SimulatorAdapter,
    build_site_config_from_scenario,
    load_scenario,
)
from rf_ingress.replay import record_adapter_to_jsonl, replay_jsonl_file
from rf_ingress.site_config import InMemorySiteConfigProvider
from tests.conftest import SCENARIOS_DIR


async def test_record_and_replay_roundtrip(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIOS_DIR / "walker_uwb.yaml")
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
    provider = InMemorySiteConfigProvider(snap)
    source = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=provider),
        scenario=scenario,
    )
    out_file = tmp_path / "walk.replay.jsonl"
    recorded = await record_adapter_to_jsonl(source, out_path=out_file, max_minutes=5.0)
    assert recorded > 0
    assert out_file.is_file()

    # Replaying the recorded raw ranges through the simulator yields TagFix messages.
    pub = await replay_jsonl_file(out_file, site_config=provider, adapter_id="sim-replay")
    assert len(pub.fixes) >= 15
    await source.close()


async def test_record_rejects_duration_above_60_minutes(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIOS_DIR / "walker_uwb.yaml")
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
    provider = InMemorySiteConfigProvider(snap)
    source = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=provider),
        scenario=scenario,
    )
    with pytest.raises(ValueError, match="max_minutes"):
        await record_adapter_to_jsonl(source, out_path=tmp_path / "bad.jsonl", max_minutes=61.0)
    await source.close()
