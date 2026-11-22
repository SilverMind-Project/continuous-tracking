"""Unit tests for FastAPI /health and /metrics endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.simulator import (
    SimulatorAdapter,
    build_site_config_from_scenario,
    load_scenario,
)
from rf_ingress.health import create_health_app
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher
from rf_ingress.site_config import InMemorySiteConfigProvider
from tests.conftest import SCENARIOS_DIR

_T0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)


async def test_health_and_metrics_endpoints() -> None:
    scenario = load_scenario(SCENARIOS_DIR / "walker_uwb.yaml")
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
    provider = InMemorySiteConfigProvider(snap, clock=lambda: _T0)
    adapter = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=provider, clock=lambda: _T0),
        scenario=scenario,
    )
    publisher = InMemoryTagFixPublisher(clock=lambda: _T0)
    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=provider,
        publisher=publisher,
        clock=lambda: _T0,
    )
    await runtime.run_once()

    app = create_health_app(runtime, clock=lambda: _T0)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["publisher"]["healthy"] is True
    assert body["site_config"]["loaded"] is True
    assert body["site_config"]["version"] == "sim:v1"
    assert body["site_config"]["node_count"] == 4
    assert len(body["adapters"]) == 1
    assert body["adapters"][0]["adapter_id"] == "sim-1"
    assert body["adapters"][0]["connected"] is True
    assert "uwb-lr-nw" in body["node_last_seen_age_s"]
    assert "tag-resident-uwb" in body["tag_last_seen_age_s"]
    assert isinstance(body["tag_battery_fraction"], dict)

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "rf_ingress_fixes_published_total" in metrics_resp.text
    assert "rf_ingress_solve_seconds" in metrics_resp.text
    await adapter.close()
