"""Deterministic end-to-end verification of all 7 rf-ingress simulator scenarios."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from cts_contracts.wearables import FixKind, MotionState, WearState

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.simulator import (
    SimulatorAdapter,
    build_site_config_from_scenario,
    load_scenario,
)
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher
from rf_ingress.site_config import InMemorySiteConfigProvider
from tests.conftest import SCENARIOS_DIR

REQUIRED_SCENARIOS: tuple[str, ...] = (
    "walker_uwb.yaml",
    "parked_tag_table.yaml",
    "two_tags_one_person.yaml",
    "sole_occupant_room.yaml",
    "two_wearers_one_room.yaml",
    "uncalibrated_room_only.yaml",
    "two_people_uwb_burst.yaml",
)


async def _run_scenario(path: Path) -> InMemoryTagFixPublisher:
    scenario = load_scenario(path)
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
    provider = InMemorySiteConfigProvider(snap)
    adapter = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=provider),
        scenario=scenario,
    )
    publisher = InMemoryTagFixPublisher()
    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=provider,
        publisher=publisher,
        clock=lambda: scenario.start_time,
    )
    await runtime.run_once()
    await adapter.close()
    return publisher


@pytest.mark.parametrize("filename", REQUIRED_SCENARIOS)
async def test_all_required_scenarios_replay_deterministically(filename: str) -> None:
    path = SCENARIOS_DIR / filename
    assert path.is_file(), f"Missing required scenario file: {path}"
    pub1 = await _run_scenario(path)
    pub2 = await _run_scenario(path)
    assert len(pub1.fixes) > 0
    assert len(pub1.capabilities) == 1
    assert [f.payload for f in pub1.fixes] == [f.payload for f in pub2.fixes]


async def test_sole_occupant_scenario_emits_core_room_fallback_fixes() -> None:
    pub = await _run_scenario(SCENARIOS_DIR / "sole_occupant_room.yaml")
    assert all(f.fix.kind == FixKind.ROOM for f in pub.fixes)
    rooms_seen = {f.fix.room_id for f in pub.fixes}
    assert rooms_seen == {"1", "2"}
    assert all(
        f.fix.room_probability is not None and 0.5 <= f.fix.room_probability <= 0.99
        for f in pub.fixes
    )


async def test_parked_tag_scenario_passes_off_body_and_still_truthfully() -> None:
    pub = await _run_scenario(SCENARIOS_DIR / "parked_tag_table.yaml")
    parked = [f.fix for f in pub.fixes if f.fix.tag_id == "tag-parked-band"]
    walker = [f.fix for f in pub.fixes if f.fix.tag_id == "tag-caregiver-band"]
    assert len(parked) > 0 and len(walker) > 0
    assert all(
        f.wear_state == WearState.OFF_BODY and f.motion_state == MotionState.STILL for f in parked
    )
    assert all(f.motion_state == MotionState.MOVING for f in walker)


async def test_two_people_uwb_burst_scenario_has_12s_burst_and_60s_silence() -> None:
    pub = await _run_scenario(SCENARIOS_DIR / "two_people_uwb_burst.yaml")
    uwb_fixes = [f.fix for f in pub.fixes if f.fix.tag_id == "tag-watch-uwb"]
    ble_watch = [f.fix for f in pub.fixes if f.fix.tag_id == "tag-watch-ble"]
    ble_senior = [f.fix for f in pub.fixes if f.fix.tag_id == "tag-senior-ble"]

    # 12-second UWB burst at 1 Hz -> 12 tight-covariance trajectory fixes.
    assert len(uwb_fixes) == 12
    assert all(f.kind == FixKind.POSITION for f in uwb_fixes)
    uwb_sigmas = [math.sqrt(0.5 * (f.cov_xx_m2 + f.cov_yy_m2)) for f in uwb_fixes]
    assert max(uwb_sigmas) < 0.35

    # Continuous BLE RSSI across the full 72 s window with wide covariance.
    assert len(ble_watch) >= 70
    assert len(ble_senior) >= 70
    ble_sigmas = [math.sqrt(0.5 * (f.cov_xx_m2 + f.cov_yy_m2)) for f in ble_watch]
    assert min(ble_sigmas) > 0.9
