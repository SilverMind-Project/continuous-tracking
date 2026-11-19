"""End-to-end integration tests publishing simulator fixes to a real Redis container."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
from redis.asyncio import Redis
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.simulator import (
    SimulatorAdapter,
    build_site_config_from_scenario,
    load_scenario,
)
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.proto.continuoustracking.v1 import wearable_pb2
from rf_ingress.publish import (
    FIELD_WEARABLE_CAPABILITY,
    FIELD_WEARABLE_FIX,
    RedisTagFixPublisher,
    validate_tag_fix_proto,
)
from rf_ingress.site_config import InMemorySiteConfigProvider
from tests.conftest import SCENARIOS_DIR


@pytest.fixture(scope="module")
def redis_url() -> Iterator[str]:
    container = (
        DockerContainer("redis:7-alpine")
        .with_exposed_ports(6379)
        .waiting_for(LogMessageWaitStrategy("Ready to accept connections"))
    )
    with container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest.mark.integration
async def test_simulator_walker_uwb_publishes_accurate_protobuf_fixes_to_redis(
    redis_url: str,
) -> None:
    scenario = load_scenario(SCENARIOS_DIR / "walker_uwb.yaml")
    snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
    provider = InMemorySiteConfigProvider(snap)
    adapter = SimulatorAdapter(
        adapter_id="sim-1",
        deps=AdapterDeps(site_config=provider),
        scenario=scenario,
    )

    fixes_stream = "test.wearable.fixes.walker"
    caps_stream = "test.wearable.capabilities.walker"
    redis_client = Redis.from_url(redis_url, decode_responses=False)
    publisher = RedisTagFixPublisher(
        redis_client,
        fixes_stream=fixes_stream,
        capabilities_stream=caps_stream,
        retention_hours=72.0,
    )
    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=provider,
        publisher=publisher,
    )

    try:
        await runtime.run_once()

        raw_fixes = await redis_client.xrange(fixes_stream, "-", "+")
        raw_caps = await redis_client.xrange(caps_stream, "-", "+")

        assert len(raw_fixes) == 21
        assert len(raw_caps) == 1

        cap = wearable_pb2.AdapterCapability()
        cap.ParseFromString(raw_caps[0][1][FIELD_WEARABLE_CAPABILITY.encode("utf-8")])
        assert cap.adapter_id == "sim-1"
        assert cap.adapter_kind == wearable_pb2.ADAPTER_KIND_SIMULATOR
        assert cap.reports_covariance is True
        assert cap.reports_wear_state is True
        assert cap.reports_motion_state is True

        decoded_fixes: list[wearable_pb2.TagFix] = []
        for _entry_id, fields in raw_fixes:
            fix = wearable_pb2.TagFix()
            fix.ParseFromString(fields[FIELD_WEARABLE_FIX.encode("utf-8")])
            validate_tag_fix_proto(fix)
            decoded_fixes.append(fix)

        assert all(f.kind == wearable_pb2.FIX_KIND_POSITION for f in decoded_fixes)
        assert all(f.has_position for f in decoded_fixes)
        assert all(f.floor_plan_id == "household:1" for f in decoded_fixes)
        assert all(f.wear_state == wearable_pb2.WEAR_STATE_ON_BODY for f in decoded_fixes)
        assert all(f.motion_state == wearable_pb2.MOTION_STATE_MOVING for f in decoded_fixes)

        # Verify first and last waypoint accuracy (start=(1.0, 1.0), end=(2.0, 4.0)).
        first = decoded_fixes[0]
        last = decoded_fixes[-1]
        assert math.hypot(first.x_mm / 1000.0 - 1.0, first.y_mm / 1000.0 - 1.0) < 0.35
        assert math.hypot(last.x_mm / 1000.0 - 2.0, last.y_mm / 1000.0 - 4.0) < 0.35
    finally:
        await runtime.stop()


@pytest.mark.integration
async def test_simulator_burst_and_sole_occupant_scenarios_on_redis(redis_url: str) -> None:
    for scenario_name, expected_kinds in (
        ("sole_occupant_room.yaml", {wearable_pb2.FIX_KIND_ROOM}),
        ("two_people_uwb_burst.yaml", {wearable_pb2.FIX_KIND_POSITION}),
    ):
        scenario = load_scenario(SCENARIOS_DIR / scenario_name)
        snap = build_site_config_from_scenario(scenario, adapter_id="sim-1")
        provider = InMemorySiteConfigProvider(snap)
        adapter = SimulatorAdapter(
            adapter_id="sim-1",
            deps=AdapterDeps(site_config=provider),
            scenario=scenario,
        )
        stream_key = f"test.wearable.fixes.{scenario.name}"
        redis_client = Redis.from_url(redis_url, decode_responses=False)
        publisher = RedisTagFixPublisher(
            redis_client,
            fixes_stream=stream_key,
            capabilities_stream=f"test.wearable.caps.{scenario.name}",
            retention_hours=72.0,
        )
        runtime = RfIngressRuntime(
            adapters=[adapter],
            site_config=provider,
            publisher=publisher,
        )
        try:
            await runtime.run_once()
            entries = await redis_client.xrange(stream_key, "-", "+")
            assert len(entries) > 0
            kinds_seen: set[int] = set()
            for _entry_id, fields in entries:
                fix = wearable_pb2.TagFix()
                fix.ParseFromString(fields[FIELD_WEARABLE_FIX.encode("utf-8")])
                validate_tag_fix_proto(fix)
                kinds_seen.add(fix.kind)
            assert expected_kinds.issubset(kinds_seen)
        finally:
            await runtime.stop()
