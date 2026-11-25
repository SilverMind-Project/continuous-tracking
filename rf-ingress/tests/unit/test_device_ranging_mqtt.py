"""Unit tests for the device_ranging_mqtt adapter (device_ranging.v1)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import pytest
from cts_contracts.wearables import (
    DeviceKind,
    MotionState,
    RangingTechnology,
    RfNodeRole,
    WearState,
)

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.device_ranging_mqtt import (
    DeviceRangingMqttAdapter,
    InMemoryMqttMessageStream,
)
from rf_ingress.adapters.registry import build_default_registry
from rf_ingress.config import AdapterConfig
from rf_ingress.domain import (
    RfNodeGeometry,
    SiteBinding,
    SiteConfigSnapshot,
    SiteDevice,
)
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher, datetime_to_unix_ns
from rf_ingress.site_config import InMemorySiteConfigProvider

_T0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)


def _site_config() -> SiteConfigSnapshot:
    return SiteConfigSnapshot(
        floor_plan_id="household:1",
        nodes=(
            RfNodeGeometry(
                anchor_id="living-room-uwb-1",
                x_m=0.0,
                y_m=0.0,
                z_m=2.4,
                adapter_id="watch-uwb",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
                uwb_address_hex="00:01",
            ),
            RfNodeGeometry(
                anchor_id="living-room-uwb-2",
                x_m=6.0,
                y_m=0.0,
                z_m=1.4,
                adapter_id="watch-uwb",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
                uwb_address_hex="00:02",
            ),
            RfNodeGeometry(
                anchor_id="living-room-uwb-3",
                x_m=6.0,
                y_m=5.0,
                z_m=2.5,
                adapter_id="watch-uwb",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
                uwb_address_hex="00:03",
            ),
            RfNodeGeometry(
                anchor_id="living-room-uwb-4",
                x_m=0.0,
                y_m=5.0,
                z_m=1.5,
                adapter_id="watch-uwb",
                node_role=RfNodeRole.RANGING_ANCHOR,
                technology=RangingTechnology.UWB,
                room_id="1",
                uwb_address_hex="00:04",
            ),
        ),
        devices=(
            SiteDevice(
                tag_id="watch-pw5",
                adapter_id="watch-uwb",
                hardware_key="hw:watch-pw5",
                device_kind=DeviceKind.WATCH,
                is_active=True,
            ),
        ),
        bindings=(
            SiteBinding(
                tag_id="watch-pw5",
                member_id="caregiver-1",
                valid_from=_T0 - timedelta(hours=1),
                valid_to=None,
            ),
        ),
        version="uwb:v1",
    )


async def test_device_ranging_v1_parses_uwb_and_channel_sounding_and_sets_device_local() -> None:
    snap = _site_config()
    provider = InMemorySiteConfigProvider(snap)
    payload = json.dumps(
        {
            "schema": "device_ranging.v1",
            "tag_id": "hw:watch-pw5",
            "sent_at_unix_ns": datetime_to_unix_ns(_T0 + timedelta(seconds=3)),
            "wear_state": "on_body",
            "motion_state": "moving",
            "battery_fraction": 0.78,
            "epochs": [
                {
                    "epoch_seq": 412,
                    "measured_at_unix_ns": datetime_to_unix_ns(_T0),
                    "ranges": [
                        {
                            "anchor_id": "",
                            "uwb_address_hex": "00:01",
                            "distance_m": 2.41,
                            "azimuth_deg": -14.2,
                            "elevation_deg": 8.5,
                            "los_confidence": 0.94,
                        },
                        {
                            "anchor_id": "living-room-uwb-2",
                            "distance_m": 3.85,
                            "technology": "ble_channel_sounding",
                        },
                    ],
                }
            ],
        }
    ).encode("utf-8")

    adapter = DeviceRangingMqttAdapter(
        adapter_id="watch-uwb",
        deps=AdapterDeps(site_config=provider),
        stream=InMemoryMqttMessageStream([("rf/devices/watch-pw5/ranges", payload)]),
    )
    items = [m async for m in adapter.ranges()]
    assert len(items) == 2

    uwb_item = items[0]
    assert uwb_item.tag_id == "watch-pw5"
    assert uwb_item.anchor_id == "living-room-uwb-1"
    assert uwb_item.technology == RangingTechnology.UWB
    assert math.isclose(uwb_item.variance_m2, 0.25**2, rel_tol=1e-9)
    assert uwb_item.azimuth_deg == -14.2
    assert uwb_item.elevation_deg == 8.5
    assert uwb_item.azimuth_reference_frame == "device_local"
    assert uwb_item.wear_state == WearState.ON_BODY
    assert uwb_item.motion_state == MotionState.MOVING
    assert uwb_item.battery_fraction == 0.78
    assert uwb_item.epoch_seq == 412
    assert uwb_item.sent_at == _T0 + timedelta(seconds=3)

    cs_item = items[1]
    assert cs_item.technology == RangingTechnology.BLE_CHANNEL_SOUNDING
    assert math.isclose(cs_item.variance_m2, 0.50**2, rel_tol=1e-9)
    assert cs_item.azimuth_reference_frame is None


async def test_device_ranging_v1_end_to_end_12s_burst_solves_and_aligns_clocks() -> None:
    snap = _site_config()
    provider = InMemorySiteConfigProvider(snap)
    publisher = InMemoryTagFixPublisher()
    true_xyz = (2.8, 2.2, 1.0)
    skew_s = 0.150
    jitters_s = [0.050, 0.350, 0.750, 0.120]

    batches: list[tuple[datetime, bytes]] = []
    true_times: dict[int, datetime] = {}
    for batch_idx, jitter_s in enumerate(jitters_s):
        start_s = batch_idx * 3
        true_send = _T0 + timedelta(seconds=start_s + 2)
        sent_watch = true_send + timedelta(seconds=skew_s)
        arrival = true_send + timedelta(seconds=jitter_s)
        epochs_json: list[dict[str, object]] = []
        for offset in range(3):
            seq = start_s + offset + 1
            true_cap = _T0 + timedelta(seconds=start_s + offset)
            true_times[seq] = true_cap
            cap_watch = true_cap + timedelta(seconds=skew_s)
            ranges_json = [
                {
                    "anchor_id": n.anchor_id,
                    "uwb_address_hex": n.uwb_address_hex,
                    "distance_m": math.dist(true_xyz, (n.x_m, n.y_m, n.z_m)),
                    "azimuth_deg": float(((seq * 53) % 360) - 180),
                    "elevation_deg": 5.0,
                }
                for n in snap.nodes
            ]
            epochs_json.append(
                {
                    "epoch_seq": seq,
                    "measured_at_unix_ns": datetime_to_unix_ns(cap_watch),
                    "ranges": ranges_json,
                }
            )
        payload = json.dumps(
            {
                "schema": "device_ranging.v1",
                "tag_id": "watch-pw5",
                "sent_at_unix_ns": datetime_to_unix_ns(sent_watch),
                "wear_state": "on_body",
                "motion_state": "moving",
                "battery_fraction": 0.80,
                "epochs": epochs_json,
            }
        ).encode("utf-8")
        batches.append((arrival, payload))

    runtime = RfIngressRuntime(adapters=[], site_config=provider, publisher=publisher)
    for arrival, payload in batches:
        adapter = DeviceRangingMqttAdapter(
            adapter_id="watch-uwb",
            deps=AdapterDeps(site_config=provider, clock=lambda a=arrival: a),
            stream=InMemoryMqttMessageStream([("rf/devices/watch-pw5/ranges", payload)]),
        )
        async for meas in adapter.ranges():
            await runtime.handle_range_measurement(adapter, meas, arrival_at=arrival)
        await adapter.close()

    assert len(publisher.fixes) == 12
    for pub in publisher.fixes:
        err_m = math.hypot(
            pub.proto.x_mm / 1000.0 - true_xyz[0], pub.proto.y_mm / 1000.0 - true_xyz[1]
        )
        assert err_m < 0.01
        expected_ns = datetime_to_unix_ns(true_times[pub.proto.epoch_seq])
        err_ms = abs(pub.proto.measured_at_unix_ns - expected_ns) / 1_000_000.0
        assert err_ms <= 60.0


def test_registry_rejects_unknown_device_ranging_profile() -> None:
    registry = build_default_registry()
    deps = AdapterDeps(site_config=InMemorySiteConfigProvider())
    cfg = AdapterConfig(
        id="watch-uwb",
        kind="device_ranging_mqtt",
        payload_profile="unknown_v9",
    )
    with pytest.raises(ValueError, match="Unknown device_ranging_mqtt payload_profile"):
        registry.build(cfg, deps)
