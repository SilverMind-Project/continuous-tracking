"""Unit tests for SimulatorAdapter determinism, schedules, and BLE RSSI modeling."""

from __future__ import annotations

import math

from cts_contracts.wearables import MotionState, RangingTechnology, WearState

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.simulator import (
    SimulatorAdapter,
    build_site_config_from_scenario,
    load_scenario,
)
from rf_ingress.site_config import InMemorySiteConfigProvider
from tests.conftest import SCENARIOS_DIR


async def _drain_ranges(adapter: SimulatorAdapter) -> list[tuple[str, str, float, float]]:
    return [(r.tag_id, r.anchor_id, r.distance_m, r.variance_m2) async for r in adapter.ranges()]


async def test_simulator_same_seed_is_bit_identical() -> None:
    scenario = load_scenario(SCENARIOS_DIR / "walker_uwb.yaml")
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")

    ad1 = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=InMemorySiteConfigProvider(snap)),
        scenario=scenario,
    )
    ad2 = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=InMemorySiteConfigProvider(snap)),
        scenario=scenario,
    )
    run1 = await _drain_ranges(ad1)
    run2 = await _drain_ranges(ad2)
    assert len(run1) > 0
    assert run1 == run2


async def test_simulator_wear_motion_and_ble_rssi_variance_models() -> None:
    scenario = load_scenario(SCENARIOS_DIR / "parked_tag_table.yaml")
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
    ad = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=InMemorySiteConfigProvider(snap)),
        scenario=scenario,
    )
    reports = [r async for r in ad.ranges()]
    parked = [r for r in reports if r.tag_id == "tag-parked-band"]
    walker = [r for r in reports if r.tag_id == "tag-caregiver-band"]

    assert len(parked) > 0 and len(walker) > 0
    assert all(r.wear_state == WearState.OFF_BODY for r in parked)
    assert all(r.motion_state == MotionState.STILL for r in parked)
    assert all(r.wear_state == WearState.UNSPECIFIED for r in walker)
    assert all(r.motion_state == MotionState.MOVING for r in walker)
    assert all(r.technology == RangingTechnology.BLE_RSSI for r in reports)
    # BLE RSSI variance is always >= 1.0 m^2 (sigma >= max(1.0, 0.5 * range)).
    assert all(r.variance_m2 >= 1.0 for r in reports)
    assert any(math.sqrt(r.variance_m2) > 1.5 for r in reports)
