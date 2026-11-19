"""Unit tests for HttpSiteConfigClient polling, caching, and epoch drop on version change."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
from cts_contracts.wearables import RangingTechnology

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.simulator import (
    SimulatorAdapter,
    build_site_config_from_scenario,
    load_scenario,
)
from rf_ingress.domain import RangeMeasurement, SiteConfigSnapshot
from rf_ingress.observability import FIXES_DROPPED_TOTAL, DropReason
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher
from rf_ingress.site_config import HttpSiteConfigClient, InMemorySiteConfigProvider
from tests.conftest import SCENARIOS_DIR

_T0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)


def _payload(version: str) -> dict[str, object]:
    return {
        "floor_plan_id": "household:1",
        "version": version,
        "nodes": [
            {
                "anchor_id": "n1",
                "adapter_id": "sim-1",
                "node_role": "ranging_anchor",
                "technology": "uwb",
                "x_m": 1.0,
                "y_m": 2.0,
                "z_m": 2.2,
                "room_id": "1",
            }
        ],
        "devices": [
            {
                "tag_id": "tag-1",
                "adapter_id": "sim-1",
                "hardware_key": "hw-1",
                "device_kind": "uwb_tag",
                "is_active": True,
            }
        ],
        "bindings": [
            {
                "tag_id": "tag-1",
                "member_id": "resident-1",
                "valid_from": "2026-11-19T00:00:00Z",
                "valid_to": None,
            }
        ],
    }


async def test_http_site_config_client_polls_authenticates_and_keeps_last_good() -> None:
    calls: list[str] = []
    responses = [
        httpx.Response(503, text="unavailable"),
        httpx.Response(200, content=json.dumps(_payload("sha256:v1")).encode("utf-8")),
        httpx.Response(500, text="server error"),
        httpx.Response(200, content=json.dumps(_payload("sha256:v2")).encode("utf-8")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/wearables/site-config"
        calls.append(request.headers.get("X-API-Key", ""))
        return responses.pop(0)

    client = HttpSiteConfigClient(
        cc_api_url="http://cc.local",
        api_key="secret-ingress-key",
        transport=httpx.MockTransport(handler),
        clock=lambda: _T0,
    )
    try:
        # First start without config returns None cleanly.
        assert await client.refresh() is None
        assert client.snapshot() is None

        # Second poll succeeds.
        s1 = await client.refresh()
        assert s1 is not None and s1.version == "sha256:v1"

        # Transient 500 retains the previous valid snapshot.
        s_err = await client.refresh()
        assert s_err is not None and s_err.version == "sha256:v1"

        # Fourth poll updates to v2.
        s2 = await client.refresh()
        assert s2 is not None and s2.version == "sha256:v2"
        assert calls == ["secret-ingress-key"] * 4
    finally:
        await client.close()


async def test_runtime_drops_in_flight_epochs_when_site_config_version_changes() -> None:
    scenario = load_scenario(SCENARIOS_DIR / "walker_uwb.yaml")
    snap_v1 = build_site_config_from_scenario(scenario, adapter_id="sim-1", version="sha256:v1")
    snap_v2 = SiteConfigSnapshot(
        floor_plan_id=snap_v1.floor_plan_id,
        nodes=snap_v1.nodes,
        bindings=snap_v1.bindings,
        devices=snap_v1.devices,
        version="sha256:v2",
    )
    provider = InMemorySiteConfigProvider(snap_v1)
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
        clock=lambda: _T0,
    )

    # Seed one partial epoch under v1.
    partial = RangeMeasurement(
        tag_id="tag-resident-uwb",
        anchor_id="uwb-lr-nw",
        distance_m=2.0,
        variance_m2=0.01,
        technology=RangingTechnology.UWB,
        measured_at=_T0,
        adapter_id="sim-1",
        epoch_seq=1,
    )
    await runtime.handle_range_measurement(adapter, partial, arrival_at=_T0)
    assert len(publisher.fixes) == 0

    before_reloaded = FIXES_DROPPED_TOTAL.labels(
        reason=DropReason.SITE_CONFIG_RELOADED
    )._value.get()
    # Switch site config version before the next anchor arrives.
    provider.set_snapshot(snap_v2)
    next_meas = RangeMeasurement(
        tag_id="tag-resident-uwb",
        anchor_id="uwb-lr-ne",
        distance_m=3.0,
        variance_m2=0.01,
        technology=RangingTechnology.UWB,
        measured_at=_T0,
        adapter_id="sim-1",
        epoch_seq=1,
    )
    await runtime.handle_range_measurement(adapter, next_meas, arrival_at=_T0)
    after_reloaded = FIXES_DROPPED_TOTAL.labels(reason=DropReason.SITE_CONFIG_RELOADED)._value.get()
    assert after_reloaded == before_reloaded + 1
    await adapter.close()
