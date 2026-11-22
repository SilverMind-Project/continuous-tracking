"""Unit tests for the ble_scanner_mqtt adapter (espresense_node_v1)."""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from cts_contracts.wearables import (
    AdapterKind,
    DeviceKind,
    FixKind,
    RangingTechnology,
    RfNodeRole,
    WearState,
)
from pydantic import ValidationError

from rf_ingress.adapters._mqtt_transport import InMemoryMqttMessageStream
from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.ble_scanner_mqtt import (
    BeaconModelConfig,
    BleNodeCalibrationConfig,
    BleScannerMqttAdapter,
    BleScannerMqttConfig,
    EspresenseNodeReportV1,
    range_variance_m2,
    rssi_to_range_m,
)
from rf_ingress.adapters.registry import build_default_registry
from rf_ingress.config import AdapterConfig
from rf_ingress.domain import (
    RfNodeGeometry,
    SiteBinding,
    SiteConfigSnapshot,
    SiteDevice,
)
from rf_ingress.observability import FIXES_DROPPED_TOTAL, UNREGISTERED_TAG_TOTAL, DropReason
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher
from rf_ingress.site_config import InMemorySiteConfigProvider

_T0 = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
_BEACON_HW_KEY = "iBeacon:e2c56db5-dffb-48d2-b060-d0f5a71096e0-1-42"


def _site_config(*, num_nodes: int = 3) -> SiteConfigSnapshot:
    all_nodes = (
        RfNodeGeometry(
            anchor_id="living-room",
            x_m=0.0,
            y_m=0.0,
            z_m=1.2,
            adapter_id="ble-home",
            node_role=RfNodeRole.RECEIVER,
            technology=RangingTechnology.BLE_RSSI,
            room_id="room-living",
        ),
        RfNodeGeometry(
            anchor_id="kitchen",
            x_m=6.0,
            y_m=0.0,
            z_m=1.2,
            adapter_id="ble-home",
            node_role=RfNodeRole.RECEIVER,
            technology=RangingTechnology.BLE_RSSI,
            room_id="room-kitchen",
        ),
        RfNodeGeometry(
            anchor_id="hallway",
            x_m=3.0,
            y_m=5.0,
            z_m=1.2,
            adapter_id="ble-home",
            node_role=RfNodeRole.RECEIVER,
            technology=RangingTechnology.BLE_RSSI,
            room_id="room-hallway",
        ),
    )
    return SiteConfigSnapshot(
        floor_plan_id="household:1",
        nodes=all_nodes[:num_nodes],
        devices=(
            SiteDevice(
                tag_id="senior-band-1",
                adapter_id="ble-home",
                hardware_key=_BEACON_HW_KEY,
                device_kind=DeviceKind.BLE_BEACON,
                is_active=True,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="senior-band-1",
                member_id="resident-1",
                valid_from=_T0 - timedelta(hours=1),
                valid_to=None,
            ),
        ),
        version="ble:v1",
    )


def _espresense_msg(
    *,
    device_id: str,
    node_id: str,
    rssi: float,
    measured_at: datetime,
    distance: float | None = 1.23,
    battery: float | None = None,
    moving: bool | None = None,
) -> tuple[str, bytes]:
    body: dict[str, object] = {
        "id": device_id,
        "rssi": rssi,
        "distance": distance,
        "measured_at": measured_at.isoformat(),
    }
    if battery is not None:
        body["battery"] = battery
    if moving is not None:
        body["moving"] = moving
    return (f"espresense/devices/{device_id}/{node_id}", json.dumps(body).encode("utf-8"))


def test_rssi_to_range_m_known_values() -> None:
    # At rssi == rssi_at_1m, distance is exactly 1.0 m.
    assert math.isclose(
        rssi_to_range_m(-59.0, rssi_at_1m_dbm=-59.0, path_loss_exponent=2.5),
        1.0,
        rel_tol=1e-9,
    )
    # With n=2.0, a 20 dB drop equals 10x distance (10.0 m).
    assert math.isclose(
        rssi_to_range_m(-79.0, rssi_at_1m_dbm=-59.0, path_loss_exponent=2.0),
        10.0,
        rel_tol=1e-9,
    )
    # With n=2.5, a 12.5 dB drop equals 10**0.5 m.
    assert math.isclose(
        rssi_to_range_m(-71.5, rssi_at_1m_dbm=-59.0, path_loss_exponent=2.5),
        math.sqrt(10.0),
        rel_tol=1e-9,
    )


def test_range_variance_grows_with_range_and_respects_floor() -> None:
    v_near = range_variance_m2(0.5, sigma_floor_m=1.0, sigma_range_factor=0.5)
    v_mid = range_variance_m2(2.0, sigma_floor_m=1.0, sigma_range_factor=0.5)
    v_far = range_variance_m2(6.0, sigma_floor_m=1.0, sigma_range_factor=0.5)
    assert math.isclose(v_near, 1.0, rel_tol=1e-9)
    assert math.isclose(v_mid, 1.0, rel_tol=1e-9)
    assert math.isclose(v_far, 9.0, rel_tol=1e-9)
    assert v_far > v_mid


def test_espresense_v1_rejects_unknown_field_layout() -> None:
    valid = EspresenseNodeReportV1.model_validate(
        {
            "id": _BEACON_HW_KEY,
            "rssi": -68,
            "distance": 2.4,
            "rssi@1m": -59,
            "var": 0.4,
            "int": 500,
        }
    )
    assert valid.rssi == -68.0
    assert valid.rssi_at_1m == -59.0
    assert valid.interval_ms == 500

    with pytest.raises(ValidationError):
        EspresenseNodeReportV1.model_validate(
            {
                "id": _BEACON_HW_KEY,
                "rssi": -68,
                "unexpected_firmware_v4_field": True,
            }
        )


async def test_malformed_and_unregistered_dropped_and_counted() -> None:
    snap = _site_config()
    provider = InMemorySiteConfigProvider(snap)

    before_malformed = FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT)._value.get()
    before_unreg = UNREGISTERED_TAG_TOTAL.labels(adapter_id="ble-home")._value.get()

    stream = InMemoryMqttMessageStream(
        [
            _espresense_msg(
                device_id=_BEACON_HW_KEY,
                node_id="living-room",
                rssi=-65.0,
                measured_at=_T0,
                battery=84.0,
            ),
            ("espresense/devices/bad/living-room", b"{not valid json"),
            (
                "espresense/devices/bad/living-room",
                json.dumps({"id": _BEACON_HW_KEY, "rssi": -65, "unknown_key": 1}).encode("utf-8"),
            ),
            ("wrong/topic", json.dumps({"id": _BEACON_HW_KEY, "rssi": -65}).encode("utf-8")),
            _espresense_msg(
                device_id="iBeacon:stranger-phone-0000-0-0",
                node_id="living-room",
                rssi=-62.0,
                measured_at=_T0 + timedelta(milliseconds=200),
            ),
        ]
    )
    adapter = BleScannerMqttAdapter(
        adapter_id="ble-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream,
    )
    items = [m async for m in adapter.ranges()]

    assert len(items) == 1
    assert items[0].tag_id == "senior-band-1"
    assert items[0].anchor_id == "living-room"
    assert items[0].technology == RangingTechnology.BLE_RSSI
    assert items[0].wear_state == WearState.UNSPECIFIED
    assert items[0].battery_fraction == pytest.approx(0.84)

    after_malformed = FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT)._value.get()
    after_unreg = UNREGISTERED_TAG_TOTAL.labels(adapter_id="ble-home")._value.get()
    assert after_malformed - before_malformed == 3
    assert after_unreg - before_unreg == 1
    assert adapter.health().malformed_reports == 3
    assert adapter.health().unregistered_reports == 1
    assert adapter.capability.adapter_kind == AdapterKind.ANCHOR_RANGING
    assert adapter.capability.reports_wear_state is False
    assert adapter.capability.reports_motion_state is True


async def test_per_epoch_median_rssi_and_max_range_drop() -> None:
    snap = _site_config()
    provider = InMemorySiteConfigProvider(snap)

    # Three hearings in the same 1 s bucket for living-room: median of (-60, -64, -90) is -64.
    # One very weak hearing for hallway (-105 dBm -> > 12 m) must be dropped.
    stream = InMemoryMqttMessageStream(
        [
            _espresense_msg(
                device_id=_BEACON_HW_KEY,
                node_id="living-room",
                rssi=-60.0,
                measured_at=_T0 + timedelta(milliseconds=100),
            ),
            _espresense_msg(
                device_id=_BEACON_HW_KEY,
                node_id="living-room",
                rssi=-90.0,
                measured_at=_T0 + timedelta(milliseconds=400),
            ),
            _espresense_msg(
                device_id=_BEACON_HW_KEY,
                node_id="living-room",
                rssi=-64.0,
                measured_at=_T0 + timedelta(milliseconds=800),
            ),
            _espresense_msg(
                device_id=_BEACON_HW_KEY,
                node_id="hallway",
                rssi=-105.0,
                measured_at=_T0 + timedelta(milliseconds=500),
            ),
        ]
    )
    cfg = BleScannerMqttConfig(
        max_range_m=12.0,
        beacon_models={"default": BeaconModelConfig(rssi_at_1m_dbm=-59.0)},
        nodes={"living-room": BleNodeCalibrationConfig(path_loss_exponent=2.5)},
    )
    adapter = BleScannerMqttAdapter(
        adapter_id="ble-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream,
        config=cfg,
    )
    items = [m async for m in adapter.ranges()]

    assert len(items) == 1
    expected_dist = rssi_to_range_m(-64.0, rssi_at_1m_dbm=-59.0, path_loss_exponent=2.5)
    assert math.isclose(items[0].distance_m, expected_dist, rel_tol=1e-9)
    assert items[0].anchor_id == "living-room"


async def test_three_nodes_covariance_contains_true_point_at_least_90_percent() -> None:
    """Seeded synthetic layout: 3 BLE nodes yield position fixes whose 95% ellipse covers truth."""
    snap = _site_config(num_nodes=3)
    provider = InMemorySiteConfigProvider(snap)
    publisher = InMemoryTagFixPublisher(clock=lambda: _T0)

    true_xy = (2.8, 1.9)
    true_z = 1.0
    rssi_at_1m = -59.0
    n_exp = 2.5
    rng = random.Random(42)
    num_epochs = 50
    messages: list[tuple[str, bytes]] = []

    for epoch_idx in range(num_epochs):
        t_base = _T0 + timedelta(seconds=epoch_idx)
        for node in snap.nodes:
            true_dist = math.dist((true_xy[0], true_xy[1], true_z), (node.x_m, node.y_m, node.z_m))
            mean_rssi = rssi_at_1m - 10.0 * n_exp * math.log10(max(true_dist, 0.25))
            # Two hearings per 1 s epoch (~500 ms interval) with 3.5 dB scatter.
            for sub in (0.15, 0.65):
                noisy_rssi = mean_rssi + rng.gauss(0.0, 3.5)
                messages.append(
                    _espresense_msg(
                        device_id=_BEACON_HW_KEY,
                        node_id=node.anchor_id,
                        rssi=noisy_rssi,
                        measured_at=t_base + timedelta(seconds=sub),
                        battery=92.0,
                    )
                )

    adapter = BleScannerMqttAdapter(
        adapter_id="ble-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=InMemoryMqttMessageStream(messages),
        config=BleScannerMqttConfig(
            sigma_floor_m=1.0,
            sigma_range_factor=0.5,
            beacon_models={"default": BeaconModelConfig(rssi_at_1m_dbm=rssi_at_1m)},
            default_path_loss_exponent=n_exp,
        ),
    )
    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=provider,
        publisher=publisher,
        clock=lambda: _T0 + timedelta(seconds=num_epochs),
    )
    emitted = await runtime.run_once()
    pos_fixes = [f for f in emitted if f.kind == FixKind.POSITION]
    assert len(pos_fixes) >= 45

    # 2-DOF chi-square 95% containment threshold is 5.991.
    contained = 0
    for fix in pos_fixes:
        assert fix.x_mm is not None and fix.y_mm is not None
        err = np.array([fix.x_mm / 1000.0 - true_xy[0], fix.y_mm / 1000.0 - true_xy[1]])
        cov = np.array([[fix.cov_xx_m2, fix.cov_xy_m2], [fix.cov_xy_m2, fix.cov_yy_m2]])
        d2 = float(err @ np.linalg.inv(cov) @ err)
        if d2 <= 5.991:
            contained += 1

    containment_rate = contained / len(pos_fixes)
    assert containment_rate >= 0.90, (
        f"Expected >=90% covariance containment, got {containment_rate}"
    )


async def test_two_nodes_produce_room_fix_for_strongest_node() -> None:
    snap = _site_config(num_nodes=2)
    provider = InMemorySiteConfigProvider(snap)
    publisher = InMemoryTagFixPublisher(clock=lambda: _T0)

    # Strongest signal in kitchen (-61 dBm) vs weaker in living-room (-79 dBm).
    stream = InMemoryMqttMessageStream(
        [
            _espresense_msg(
                device_id=_BEACON_HW_KEY,
                node_id="kitchen",
                rssi=-61.0,
                measured_at=_T0 + timedelta(milliseconds=200),
            ),
            _espresense_msg(
                device_id=_BEACON_HW_KEY,
                node_id="living-room",
                rssi=-79.0,
                measured_at=_T0 + timedelta(milliseconds=300),
            ),
        ]
    )
    adapter = BleScannerMqttAdapter(
        adapter_id="ble-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        stream=stream,
    )
    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=provider,
        publisher=publisher,
        clock=lambda: _T0 + timedelta(seconds=2),
    )
    emitted = await runtime.run_once()
    assert len(emitted) == 1
    fix = emitted[0]
    assert fix.kind == FixKind.ROOM
    assert fix.room_id == "room-kitchen"
    assert fix.room_probability is not None
    assert 0.5 < fix.room_probability <= 0.99


def test_registry_builds_ble_scanner_and_rejects_unknown_profile() -> None:
    registry = build_default_registry()
    assert "ble_scanner_mqtt" in registry.registered_kinds()
    deps = AdapterDeps(site_config=InMemorySiteConfigProvider())

    good_cfg = AdapterConfig(
        id="ble-home",
        kind="ble_scanner_mqtt",
        payload_profile="espresense_node_v1",
        broker_url="mqtt://localhost:1883",
        topic="espresense/devices/+/+",
        epoch_window_s=1.0,
    )
    built = registry.build(good_cfg, deps)
    assert built.adapter_id == "ble-home"
    assert built.capability.adapter_kind == AdapterKind.ANCHOR_RANGING

    bad_cfg = AdapterConfig(
        id="ble-home",
        kind="ble_scanner_mqtt",
        payload_profile="unknown_profile_v99",
    )
    with pytest.raises(ValueError, match="Unknown ble_scanner_mqtt payload_profile"):
        registry.build(bad_cfg, deps)
