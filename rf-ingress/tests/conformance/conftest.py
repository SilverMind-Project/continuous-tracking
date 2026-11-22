"""Adapter conformance kit registration.

Every adapter added to rf-ingress (present and future) adds one or more
:class:`AdapterConformanceCase` factories to ``CONFORMANCE_CASES`` and must pass
``tests/conformance/test_adapter_conformance.py`` unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cts_contracts.wearables import (
    DeviceKind,
    MotionState,
    RangingTechnology,
    RfNodeRole,
    WearState,
)

from rf_ingress.adapters._mqtt_transport import InMemoryMqttMessageStream
from rf_ingress.adapters.anchor_ranging_mqtt import AnchorRangingMqttAdapter
from rf_ingress.adapters.base import AdapterDeps, AnyAdapter
from rf_ingress.adapters.ble_scanner_mqtt import BleScannerMqttAdapter
from rf_ingress.adapters.device_ranging_mqtt import DeviceRangingMqttAdapter
from rf_ingress.adapters.simulator import (
    SimulatorAdapter,
    build_site_config_from_scenario,
    load_scenario,
)
from rf_ingress.domain import (
    RangeMeasurement,
    RfNodeGeometry,
    SiteBinding,
    SiteConfigSnapshot,
    SiteDevice,
)
from rf_ingress.publish import datetime_to_unix_ns
from rf_ingress.replay import serialize_report_jsonl
from rf_ingress.site_config import InMemorySiteConfigProvider

_SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "config" / "scenarios"


@dataclass(frozen=True)
class AdapterConformanceContext:
    """Built adapter instance together with the site config it was wired against."""

    adapter: AnyAdapter
    site_config: InMemorySiteConfigProvider
    expected_registered_tags: frozenset[str]
    expected_unregistered_tags: frozenset[str] = frozenset()
    expects_malformed_drop: bool = False


@dataclass(frozen=True)
class AdapterConformanceCase:
    """Parameterized conformance fixture entry for one adapter configuration."""

    name: str
    build: Callable[[], AdapterConformanceContext]


def _build_simulator_range_case() -> AdapterConformanceContext:
    scenario = load_scenario(_SCENARIOS_DIR / "walker_uwb.yaml")
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
    provider = InMemorySiteConfigProvider(snap)
    adapter = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=provider),
        scenario=scenario,
    )
    return AdapterConformanceContext(
        adapter=adapter,
        site_config=provider,
        expected_registered_tags=frozenset({"tag-resident-uwb"}),
    )


def _build_simulator_uncalibrated_pos_room_case() -> AdapterConformanceContext:
    scenario = load_scenario(_SCENARIOS_DIR / "uncalibrated_room_only.yaml")
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
    provider = InMemorySiteConfigProvider(snap)
    adapter = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=provider),
        scenario=scenario,
    )
    return AdapterConformanceContext(
        adapter=adapter,
        site_config=provider,
        expected_registered_tags=frozenset({"tag-room-presence", "tag-room-pos-fallback"}),
    )


def _build_simulator_replay_with_malformed_and_unregistered_case() -> AdapterConformanceContext:
    t0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)
    snap = SiteConfigSnapshot(
        floor_plan_id="household:1",
        nodes=(
            RfNodeGeometry(
                anchor_id="n1",
                x_m=0.0,
                y_m=0.0,
                z_m=2.0,
                adapter_id="sim-replay",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
            ),
        ),
        devices=(
            SiteDevice(
                tag_id="tag-known",
                adapter_id="sim-replay",
                hardware_key="sim:tag-known",
                device_kind=DeviceKind.UWB_TAG,
                is_active=True,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="tag-known",
                member_id="resident-1",
                valid_from=t0 - timedelta(hours=1),
                valid_to=None,
            ),
        ),
        version="conf:v1",
    )
    provider = InMemorySiteConfigProvider(snap)
    valid_1 = serialize_report_jsonl(
        RangeMeasurement(
            tag_id="tag-known",
            anchor_id="n1",
            distance_m=1.5,
            variance_m2=0.01,
            technology=RangingTechnology.UWB,
            measured_at=t0,
            wear_state=WearState.ON_BODY,
            motion_state=MotionState.MOVING,
            adapter_id="sim-replay",
            epoch_seq=1,
        )
    )
    unregistered = serialize_report_jsonl(
        RangeMeasurement(
            tag_id="tag-stranger",
            anchor_id="n1",
            distance_m=2.0,
            variance_m2=0.01,
            technology=RangingTechnology.UWB,
            measured_at=t0 + timedelta(seconds=1),
            wear_state=WearState.ON_BODY,
            motion_state=MotionState.MOVING,
            adapter_id="sim-replay",
            epoch_seq=2,
        )
    )
    valid_2 = serialize_report_jsonl(
        RangeMeasurement(
            tag_id="tag-known",
            anchor_id="n1",
            distance_m=1.7,
            variance_m2=0.01,
            technology=RangingTechnology.UWB,
            measured_at=t0 + timedelta(seconds=2),
            wear_state=WearState.ON_BODY,
            motion_state=MotionState.MOVING,
            adapter_id="sim-replay",
            epoch_seq=3,
        )
    )
    lines = [
        valid_1,
        "{not valid json",
        '{"report_type":"range","tag_id":"tag-known","distance_m":-5}',
        unregistered,
        valid_2,
    ]
    adapter = SimulatorAdapter(
        adapter_id="sim-replay",
        deps=AdapterDeps(site_config=provider),
        replay_lines=lines,
    )
    return AdapterConformanceContext(
        adapter=adapter,
        site_config=provider,
        expected_registered_tags=frozenset({"tag-known"}),
        expected_unregistered_tags=frozenset({"tag-stranger"}),
        expects_malformed_drop=True,
    )


def _build_device_ranging_mqtt_case() -> AdapterConformanceContext:
    t0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)
    snap = SiteConfigSnapshot(
        floor_plan_id="household:1",
        nodes=(
            RfNodeGeometry(
                anchor_id="living-room-uwb-1",
                x_m=0.2,
                y_m=0.2,
                z_m=2.4,
                adapter_id="watch-uwb-1",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
                uwb_address_hex="00:01",
            ),
        ),
        devices=(
            SiteDevice(
                tag_id="watch-pw5-1",
                adapter_id="watch-uwb-1",
                hardware_key="watch-pw5-1",
                device_kind=DeviceKind.WATCH,
                is_active=True,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="watch-pw5-1",
                member_id="caregiver-1",
                valid_from=t0 - timedelta(hours=1),
                valid_to=None,
            ),
        ),
        version="conf:uwb:v1",
    )
    provider = InMemorySiteConfigProvider(snap)
    valid_batch = json.dumps(
        {
            "schema": "device_ranging.v1",
            "tag_id": "watch-pw5-1",
            "sent_at_unix_ns": datetime_to_unix_ns(t0 + timedelta(seconds=3)),
            "wear_state": "on_body",
            "motion_state": "moving",
            "battery_fraction": 0.78,
            "epochs": [
                {
                    "epoch_seq": 1,
                    "measured_at_unix_ns": datetime_to_unix_ns(t0),
                    "ranges": [
                        {
                            "anchor_id": "living-room-uwb-1",
                            "uwb_address_hex": "00:01",
                            "distance_m": 2.41,
                            "azimuth_deg": -14.2,
                            "elevation_deg": 8.5,
                            "los_confidence": 0.94,
                        }
                    ],
                },
                {
                    "epoch_seq": 2,
                    "measured_at_unix_ns": datetime_to_unix_ns(t0 + timedelta(seconds=1)),
                    "ranges": [
                        {
                            "anchor_id": "living-room-uwb-1",
                            "uwb_address_hex": "00:01",
                            "distance_m": 2.10,
                        }
                    ],
                },
            ],
        }
    ).encode("utf-8")
    unregistered_batch = json.dumps(
        {
            "schema": "device_ranging.v1",
            "tag_id": "watch-stranger",
            "sent_at_unix_ns": datetime_to_unix_ns(t0 + timedelta(seconds=4)),
            "wear_state": "on_body",
            "motion_state": "moving",
            "epochs": [
                {
                    "epoch_seq": 1,
                    "measured_at_unix_ns": datetime_to_unix_ns(t0 + timedelta(seconds=2)),
                    "ranges": [{"anchor_id": "living-room-uwb-1", "distance_m": 1.8}],
                }
            ],
        }
    ).encode("utf-8")
    stream = InMemoryMqttMessageStream(
        [
            ("rf/devices/watch-pw5-1/ranges", valid_batch),
            ("rf/devices/watch-pw5-1/ranges", b"{not valid json"),
            ("rf/devices/watch-pw5-1/ranges", b'{"schema":"wrong.v0"}'),
            ("rf/devices/watch-stranger/ranges", unregistered_batch),
        ]
    )
    adapter = DeviceRangingMqttAdapter(
        adapter_id="watch-uwb-1",
        deps=AdapterDeps(site_config=provider),
        stream=stream,
    )
    return AdapterConformanceContext(
        adapter=adapter,
        site_config=provider,
        expected_registered_tags=frozenset({"watch-pw5-1"}),
        expected_unregistered_tags=frozenset({"watch-stranger"}),
        expects_malformed_drop=True,
    )


def _build_ble_scanner_mqtt_case() -> AdapterConformanceContext:
    t0 = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    hw_key = "iBeacon:e2c56db5-dffb-48d2-b060-d0f5a71096e0-1-42"
    snap = SiteConfigSnapshot(
        floor_plan_id="household:1",
        nodes=(
            RfNodeGeometry(
                anchor_id="living-room",
                x_m=0.0,
                y_m=0.0,
                z_m=1.2,
                adapter_id="ble-home",
                node_role=RfNodeRole.RECEIVER,
                technology=RangingTechnology.BLE_RSSI,
                room_id="1",
            ),
            RfNodeGeometry(
                anchor_id="kitchen",
                x_m=5.0,
                y_m=0.0,
                z_m=1.2,
                adapter_id="ble-home",
                node_role=RfNodeRole.RECEIVER,
                technology=RangingTechnology.BLE_RSSI,
                room_id="2",
            ),
            RfNodeGeometry(
                anchor_id="hallway",
                x_m=2.5,
                y_m=4.5,
                z_m=1.2,
                adapter_id="ble-home",
                node_role=RfNodeRole.RECEIVER,
                technology=RangingTechnology.BLE_RSSI,
                room_id="3",
            ),
        ),
        devices=(
            SiteDevice(
                tag_id="senior-band-1",
                adapter_id="ble-home",
                hardware_key=hw_key,
                device_kind=DeviceKind.BLE_BEACON,
                is_active=True,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="senior-band-1",
                member_id="resident-1",
                valid_from=t0 - timedelta(hours=1),
                valid_to=None,
            ),
        ),
        version="conf:ble:v1",
    )
    provider = InMemorySiteConfigProvider(snap)

    def _msg(device_id: str, node_id: str, rssi: float, offset_s: float) -> tuple[str, bytes]:
        payload = json.dumps(
            {
                "id": device_id,
                "rssi": rssi,
                "distance": 2.0,
                "measured_at": (t0 + timedelta(seconds=offset_s)).isoformat(),
                "battery": 88.0,
            }
        ).encode("utf-8")
        return (f"espresense/devices/{device_id}/{node_id}", payload)

    stream = InMemoryMqttMessageStream(
        [
            _msg(hw_key, "living-room", -64.0, 0.1),
            _msg(hw_key, "kitchen", -70.0, 0.2),
            _msg(hw_key, "hallway", -72.0, 0.3),
            ("espresense/devices/bad/living-room", b"{not valid json"),
            (
                "espresense/devices/bad/living-room",
                json.dumps({"id": hw_key, "rssi": -65, "unknown_v9_field": 1}).encode("utf-8"),
            ),
            _msg("iBeacon:stranger-0000-0-0", "living-room", -61.0, 0.4),
            _msg(hw_key, "living-room", -65.0, 1.2),
            _msg(hw_key, "kitchen", -69.0, 1.3),
        ]
    )
    adapter = BleScannerMqttAdapter(
        adapter_id="ble-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: t0),
        stream=stream,
    )
    return AdapterConformanceContext(
        adapter=adapter,
        site_config=provider,
        expected_registered_tags=frozenset({"senior-band-1"}),
        expected_unregistered_tags=frozenset({"iBeacon:stranger-0000-0-0"}),
        expects_malformed_drop=True,
    )


def _build_anchor_ranging_mqtt_case() -> AdapterConformanceContext:
    t0 = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    hw_key = "uwb:AA:BB:CC:01"
    snap = SiteConfigSnapshot(
        floor_plan_id="household:1",
        nodes=(
            RfNodeGeometry(
                anchor_id="living-room-uwb-1",
                x_m=0.2,
                y_m=0.2,
                z_m=2.3,
                adapter_id="uwb-home",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
                uwb_address_hex="00:01",
                uwb_channel=9,
                uwb_preamble_index=10,
            ),
            RfNodeGeometry(
                anchor_id="living-room-uwb-2",
                x_m=5.8,
                y_m=0.2,
                z_m=1.4,
                adapter_id="uwb-home",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
                uwb_address_hex="00:02",
                uwb_channel=9,
                uwb_preamble_index=10,
            ),
            RfNodeGeometry(
                anchor_id="living-room-uwb-3",
                x_m=5.8,
                y_m=4.8,
                z_m=2.4,
                adapter_id="uwb-home",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
                uwb_address_hex="00:03",
                uwb_channel=9,
                uwb_preamble_index=10,
            ),
        ),
        devices=(
            SiteDevice(
                tag_id="senior-uwb-clip-1",
                adapter_id="uwb-home",
                hardware_key=hw_key,
                device_kind=DeviceKind.UWB_TAG,
                is_active=True,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="senior-uwb-clip-1",
                member_id="resident-1",
                valid_from=t0 - timedelta(hours=1),
                valid_to=None,
            ),
        ),
        version="conf:anchor-uwb:v1",
    )
    provider = InMemorySiteConfigProvider(snap)

    def _msg(
        anchor_id: str,
        tag_hw_key: str,
        dist_m: float,
        epoch_seq: int,
        offset_s: float,
    ) -> tuple[str, bytes]:
        ts = t0 + timedelta(seconds=offset_s)
        payload = json.dumps(
            {
                "schema": "fira_anchor.v1",
                "anchor_id": anchor_id,
                "hardware_key": tag_hw_key,
                "epoch_seq": epoch_seq,
                "measured_at_unix_ns": datetime_to_unix_ns(ts),
                "sent_at_unix_ns": datetime_to_unix_ns(ts + timedelta(milliseconds=25)),
                "distance_m": dist_m,
                "wear_state": "on_body",
                "motion_state": "moving",
                "charging": False,
                "battery_fraction": 0.84,
            }
        ).encode("utf-8")
        return (f"rf/anchors/{anchor_id}/ranges", payload)

    stream = InMemoryMqttMessageStream(
        [
            _msg("living-room-uwb-1", hw_key, 3.10, 1, 0.0),
            _msg("living-room-uwb-2", hw_key, 3.25, 1, 0.02),
            _msg("living-room-uwb-3", hw_key, 3.40, 1, 0.04),
            ("rf/anchors/living-room-uwb-1/ranges", b"{not valid json"),
            (
                "rf/anchors/living-room-uwb-1/ranges",
                json.dumps({"schema": "fira_anchor.v1", "unexpected": 1}).encode("utf-8"),
            ),
            _msg("living-room-uwb-1", "uwb:FF:FF:FF:FF", 2.0, 2, 0.5),
            _msg("living-room-uwb-1", hw_key, 2.95, 2, 1.0),
            _msg("living-room-uwb-2", hw_key, 3.15, 2, 1.02),
        ]
    )
    adapter = AnchorRangingMqttAdapter(
        adapter_id="uwb-home",
        deps=AdapterDeps(site_config=provider, clock=lambda: t0),
        stream=stream,
    )
    return AdapterConformanceContext(
        adapter=adapter,
        site_config=provider,
        expected_registered_tags=frozenset({"senior-uwb-clip-1"}),
        expected_unregistered_tags=frozenset({"uwb:FF:FF:FF:FF"}),
        expects_malformed_drop=True,
    )


CONFORMANCE_CASES: tuple[AdapterConformanceCase, ...] = (
    AdapterConformanceCase(name="simulator-range", build=_build_simulator_range_case),
    AdapterConformanceCase(
        name="simulator-position-and-room",
        build=_build_simulator_uncalibrated_pos_room_case,
    ),
    AdapterConformanceCase(
        name="simulator-replay-malformed-and-unregistered",
        build=_build_simulator_replay_with_malformed_and_unregistered_case,
    ),
    AdapterConformanceCase(
        name="device-ranging-mqtt-v1",
        build=_build_device_ranging_mqtt_case,
    ),
    AdapterConformanceCase(
        name="ble-scanner-mqtt-v1",
        build=_build_ble_scanner_mqtt_case,
    ),
    AdapterConformanceCase(
        name="anchor-ranging-mqtt-v1",
        build=_build_anchor_ranging_mqtt_case,
    ),
)


@pytest.fixture(params=CONFORMANCE_CASES, ids=lambda c: c.name)
def conformance_ctx(request: pytest.FixtureRequest) -> AdapterConformanceContext:
    case: AdapterConformanceCase = request.param
    return case.build()
