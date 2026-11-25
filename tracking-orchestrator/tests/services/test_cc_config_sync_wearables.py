"""Tests for wearable site-config syncing, coordinate conversion, and identity purge."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from cts_contracts import DeviceKind, RangingTechnology, RfNodeRole
from prometheus_client import CollectorRegistry

from app.calibration.state import CalibrationState
from app.domain.wearables import WearableBinding, WearableSiteConfig
from app.observability import metrics as metrics_module
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.services.cc_config_sync import (
    WEARABLES_SITE_CONFIG_PATH,
    CCConfigSyncContractError,
    CCConfigSyncService,
    parse_wearable_site_config,
)
from app.services.identity_roster_sync import IdentityRosterSyncService
from app.services.transit_zone_map import TransitZoneMap
from app.services.wearable_site_config_map import WearableSiteConfigMap
from app.storage.base import InMemoryGalleryRepository


class _FakeCCClient:
    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses

    async def get(self, path: str) -> Any:
        result = self._responses[path]
        if isinstance(result, Exception):
            raise result
        return result


def _sample_site_config_payload() -> dict[str, Any]:
    return {
        "floor_plan_id": "floor-plans/main.png",
        "version": "sha256:abc123",
        "nodes": [
            {
                "anchor_id": "node-lr-ble",
                "adapter_id": "ble-home",
                "node_role": "receiver",
                "technology": "ble_rssi",
                "x_m": 2.3456,
                "y_m": 4.5,
                "z_m": 1.25,
                "room_id": "1",
                "uwb_address_hex": None,
                "uwb_channel": None,
                "uwb_preamble_index": None,
            },
            {
                "anchor_id": "node-lr-uwb-1",
                "adapter_id": "uwb-watch",
                "node_role": "ranging_anchor",
                "technology": "uwb",
                "x_m": 0.0,
                "y_m": 6.0,
                "z_m": 2.1,
                "room_id": "1",
                "uwb_address_hex": "00:01",
                "uwb_channel": 9,
                "uwb_preamble_index": 10,
            },
        ],
        "bindings": [
            {
                "tag_id": "watch-ble-1",
                "member_id": "sriram",
                "valid_from": "2026-11-19T10:00:00+00:00",
                "valid_to": None,
            },
            {
                "tag_id": "watch-uwb-1",
                "member_id": "sriram",
                "valid_from": "2026-11-19T10:00:00+00:00",
                "valid_to": None,
            },
            {
                "tag_id": "band-1",
                "member_id": "amma",
                "valid_from": "2026-11-19T08:00:00+00:00",
                "valid_to": "2026-11-19T12:00:00+00:00",
            },
        ],
        "devices": [
            {
                "tag_id": "watch-ble-1",
                "adapter_id": "ble-home",
                "hardware_key": "e2c56db5-dffb-48d2-b060-d0f5a71096e0:1:101",
                "device_kind": "watch",
                "is_active": True,
            },
            {
                "tag_id": "watch-uwb-1",
                "adapter_id": "uwb-watch",
                "hardware_key": "pw5-uwb-01",
                "device_kind": "watch",
                "is_active": True,
            },
            {
                "tag_id": "band-1",
                "adapter_id": "ble-home",
                "hardware_key": "e2c56db5-dffb-48d2-b060-d0f5a71096e0:1:102",
                "device_kind": "ble_beacon",
                "is_active": True,
            },
        ],
    }


def test_parse_wearable_site_config_converts_metres_to_millimetres_and_resolves_dual_mode() -> None:
    cfg = parse_wearable_site_config(_sample_site_config_payload())

    assert cfg.floor_plan_id == "floor-plans/main.png"
    assert cfg.version == "sha256:abc123"
    assert len(cfg.nodes) == 2

    ble_node, uwb_node = cfg.nodes
    assert ble_node.node_role is RfNodeRole.RECEIVER
    assert ble_node.technology is RangingTechnology.BLE_RSSI
    assert ble_node.x_mm == 2346
    assert ble_node.y_mm == 4500
    assert ble_node.z_mm == 1250
    assert ble_node.room_id == "1"

    assert uwb_node.node_role is RfNodeRole.RANGING_ANCHOR
    assert uwb_node.technology is RangingTechnology.UWB
    assert uwb_node.x_mm == 0
    assert uwb_node.y_mm == 6000
    assert uwb_node.z_mm == 2100
    assert uwb_node.uwb_address_hex == "00:01"
    assert uwb_node.uwb_channel == 9
    assert uwb_node.uwb_preamble_index == 10

    assert [d.device_kind for d in cfg.devices] == [
        DeviceKind.WATCH,
        DeviceKind.WATCH,
        DeviceKind.BLE_BEACON,
    ]

    # Dual-mode watch: both BLE and UWB tag_ids resolve to the same member_id.
    t = datetime(2026, 11, 19, 11, 0, tzinfo=UTC)
    assert cfg.member_at("watch-ble-1", t) == "sriram"
    assert cfg.member_at("watch-uwb-1", t) == "sriram"


def test_parse_carries_physical_device_id_and_resolves_sibling_tags() -> None:
    payload = _sample_site_config_payload()
    payload["devices"][0]["physical_device_id"] = "watch-sriram"
    payload["devices"][1]["physical_device_id"] = "watch-sriram"
    payload["devices"][2]["physical_device_id"] = None
    cfg = parse_wearable_site_config(payload)

    assert cfg.physical_device_of("watch-ble-1") == "watch-sriram"
    assert cfg.physical_device_of("band-1") is None
    assert cfg.physical_device_of("unknown-tag") is None
    assert cfg.sibling_tag_ids("watch-uwb-1") == ("watch-uwb-1", "watch-ble-1")
    assert cfg.sibling_tag_ids("band-1") == ("band-1",)


def test_parse_accepts_payload_without_physical_device_id() -> None:
    # A CC build that predates the field omits it; every tag is then its own device.
    cfg = parse_wearable_site_config(_sample_site_config_payload())
    assert all(d.physical_device_id is None for d in cfg.devices)


@pytest.mark.parametrize("bad", ["", 7, ["watch"]])
def test_parse_rejects_malformed_physical_device_id(bad: object) -> None:
    payload = _sample_site_config_payload()
    payload["devices"][0]["physical_device_id"] = bad
    with pytest.raises(CCConfigSyncContractError, match="physical_device_id"):
        parse_wearable_site_config(payload)


def test_member_at_respects_inclusive_start_and_exclusive_end() -> None:
    t_start = datetime(2026, 11, 19, 8, 0, tzinfo=UTC)
    t_end = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
    cfg = WearableSiteConfig(
        version="v1",
        bindings=(
            WearableBinding(
                tag_id="band-1",
                member_id="amma",
                valid_from=t_start,
                valid_to=t_end,
            ),
            WearableBinding(
                tag_id="band-1",
                member_id="sriram",
                valid_from=t_end,
                valid_to=None,
            ),
        ),
    )

    assert cfg.member_at("band-1", t_start - timedelta(microseconds=1)) is None
    assert cfg.member_at("band-1", t_start) == "amma"
    assert cfg.member_at("band-1", t_end - timedelta(microseconds=1)) == "amma"
    assert cfg.member_at("band-1", t_end) == "sriram"
    assert cfg.member_at("band-1", t_end + timedelta(hours=5)) == "sriram"
    assert cfg.member_at("unknown-tag", t_start) is None


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.update({"version": ""}),
        lambda p: p["nodes"][0].update({"node_role": "bad_role"}),
        lambda p: p["nodes"][0].update({"technology": "zigbee"}),
        lambda p: p["nodes"][0].update({"x_m": float("nan")}),
        lambda p: p["devices"][0].update({"device_kind": "tag"}),
        lambda p: p["bindings"][0].update({"valid_from": "2026-11-19T10:00:00"}),
        lambda p: p["bindings"][2].update({"valid_to": "2026-11-19T08:00:00+00:00"}),
    ],
)
def test_parse_wearable_site_config_rejects_malformed_payload(mutator: Any) -> None:
    payload = _sample_site_config_payload()
    mutator(payload)
    with pytest.raises(CCConfigSyncContractError):
        parse_wearable_site_config(payload)


@pytest.mark.asyncio
async def test_cc_config_sync_keeps_previous_config_and_counts_failure_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CollectorRegistry()
    test_metrics = metrics_module.build_metrics(registry=registry)
    monkeypatch.setattr(metrics_module, "metrics", test_metrics)

    responses: dict[str, Any] = {
        "/api/v1/cts/cameras": [],
        "/api/v1/cts/overlap_groups": [],
        "/api/v1/household/floor-plan": {
            "floor_plan_width": 1000,
            "floor_plan_height": 800,
            "floor_meters_per_pixel": 0.01,
        },
        "/api/v1/rooms": [],
        "/api/v1/cts/transit-zones": [],
        WEARABLES_SITE_CONFIG_PATH: _sample_site_config_payload(),
    }
    holder = WearableSiteConfigMap()
    svc = CCConfigSyncService(
        client=_FakeCCClient(responses),
        calibration_state=CalibrationState(),
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
        transit_zone_map=TransitZoneMap(),
        fetch_retry_attempts=1,
        wearable_site_config_map=holder,
    )

    await svc._poll()
    good = holder.get_sync()
    assert good.version == "sha256:abc123"
    assert len(good.nodes) == 2
    assert test_metrics.cc_sync_failures_total.labels(resource="wearables")._value.get() == 0.0

    # Corrupt payload on next poll: previous config stays active and metric increments.
    responses[WEARABLES_SITE_CONFIG_PATH] = {"version": "broken", "nodes": "not-a-list"}
    await svc._poll()
    assert holder.get_sync() == good
    assert test_metrics.cc_sync_failures_total.labels(resource="wearables")._value.get() == 1.0


@pytest.mark.asyncio
async def test_roster_sync_purge_identity_drops_in_memory_wearable_bindings() -> None:
    holder = WearableSiteConfigMap(parse_wearable_site_config(_sample_site_config_payload()))
    gallery_repo = InMemoryGalleryRepository()
    roster_sync = IdentityRosterSyncService(
        gallery_repo=gallery_repo,
        wearable_site_config=holder,
    )

    t = datetime(2026, 11, 19, 11, 0, tzinfo=UTC)
    assert holder.get_sync().member_at("watch-ble-1", t) == "sriram"
    assert holder.get_sync().member_at("watch-uwb-1", t) == "sriram"
    assert holder.get_sync().member_at("band-1", t) == "amma"

    purged = await roster_sync.purge_identity("sriram")
    assert purged is True
    assert holder.get_sync().member_at("watch-ble-1", t) is None
    assert holder.get_sync().member_at("watch-uwb-1", t) is None
    # Unrelated member's binding stays intact.
    assert holder.get_sync().member_at("band-1", t) == "amma"


@pytest.mark.asyncio
async def test_parse_wearable_site_config_position_sensors_and_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CollectorRegistry()
    test_metrics = metrics_module.build_metrics(registry)
    monkeypatch.setattr(metrics_module, "metrics", test_metrics)

    payload = _sample_site_config_payload()
    # Omitted position_sensors key stays valid and produces an empty tuple.
    cfg_without_sensors = parse_wearable_site_config(payload)
    assert cfg_without_sensors.position_sensors == ()
    assert cfg_without_sensors.position_sensor("radar-bath-1") is None

    payload["position_sensors"] = [
        {
            "sensor_id": "radar-bath-1",
            "room_id": "2",
            "x_m": 1.25,
            "y_m": 3.5,
            "yaw_deg": 90.0,
            "calibration_residual_m": 0.15,
        }
    ]
    cfg = parse_wearable_site_config(payload)
    assert len(cfg.position_sensors) == 1
    sensor = cfg.position_sensor("radar-bath-1")
    assert sensor is not None
    assert sensor.room_id == "2"
    assert sensor.x_mm == 1250
    assert sensor.y_mm == 3500
    assert sensor.yaw_deg == pytest.approx(90.0)
    assert sensor.calibration_residual_m == pytest.approx(0.15)

    for bad_sensors in (
        [
            {
                "sensor_id": "bad:id",
                "room_id": "2",
                "x_m": 1.0,
                "y_m": 1.0,
                "calibration_residual_m": 0.1,
            }
        ],
        [
            {
                "sensor_id": "radar-bath-1",
                "room_id": "2",
                "x_m": 1.0,
                "y_m": 1.0,
                "calibration_residual_m": 0.1,
            },
            {
                "sensor_id": "radar-bath-1",
                "room_id": "2",
                "x_m": 2.0,
                "y_m": 2.0,
                "calibration_residual_m": 0.1,
            },
        ],
        [
            {
                "sensor_id": "radar-bath-1",
                "room_id": "2",
                "x_m": 1.0,
                "y_m": 1.0,
                "calibration_residual_m": 0.0,
            }
        ],
        [
            {
                "sensor_id": "radar-bath-1",
                "room_id": "2",
                "x_m": 1.0,
                "y_m": 1.0,
                "calibration_residual_m": 1.5,
            }
        ],
    ):
        bad_payload = _sample_site_config_payload()
        bad_payload["position_sensors"] = bad_sensors
        with pytest.raises(CCConfigSyncContractError):
            parse_wearable_site_config(bad_payload)

    # Corrupt position_sensors during live poll retains the last valid config.
    responses: dict[str, Any] = {
        "/api/v1/cts/cameras": [],
        "/api/v1/cts/overlap_groups": [],
        "/api/v1/household/floor-plan": {
            "floor_plan_width": 1000,
            "floor_plan_height": 800,
            "floor_meters_per_pixel": 0.01,
        },
        "/api/v1/rooms": [],
        "/api/v1/cts/transit-zones": [],
        WEARABLES_SITE_CONFIG_PATH: payload,
    }
    holder = WearableSiteConfigMap()
    svc = CCConfigSyncService(
        client=_FakeCCClient(responses),
        calibration_state=CalibrationState(),
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
        transit_zone_map=TransitZoneMap(),
        fetch_retry_attempts=1,
        wearable_site_config_map=holder,
    )
    await svc._poll()
    assert len(holder.get_sync().position_sensors) == 1

    bad_poll_payload = _sample_site_config_payload()
    bad_poll_payload["position_sensors"] = [{"sensor_id": "bad:id"}]
    responses[WEARABLES_SITE_CONFIG_PATH] = bad_poll_payload
    await svc._poll()
    assert len(holder.get_sync().position_sensors) == 1
    assert test_metrics.cc_sync_failures_total.labels(resource="wearables")._value.get() == 1.0
