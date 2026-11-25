"""Runtime tests for the background epoch flush loop and POSITION room attribution."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from cts_contracts.wearables import FixKind, RangingTechnology

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.simulator import (
    SimulatorAdapter,
    build_site_config_from_scenario,
    load_scenario,
)
from rf_ingress.domain import EmittedFix, RangeMeasurement, RfNodeGeometry, SiteConfigSnapshot
from rf_ingress.observability import FIXES_DROPPED_TOTAL, DropReason
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher
from rf_ingress.site_config import InMemorySiteConfigProvider
from tests.conftest import SCENARIOS_DIR

_T0 = datetime(2026, 11, 19, 12, 0, 30, tzinfo=UTC)
_TAG = "tag-resident-uwb"
_TRUE_XY = (2.0, 2.5)


class _ManualClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class _FailingOncePublisher(InMemoryTagFixPublisher):
    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1

    async def publish(self, fix: EmittedFix) -> str:
        if self.failures_left > 0:
            self.failures_left -= 1
            raise ConnectionError("redis unavailable")
        return await super().publish(fix)


def _snapshot() -> SiteConfigSnapshot:
    scenario = load_scenario(SCENARIOS_DIR / "walker_uwb.yaml")
    return build_site_config_from_scenario(scenario, adapter_id="sim-1", version="sha256:v1")


def _build(
    *,
    clock: Callable[[], datetime],
    publisher: InMemoryTagFixPublisher | None = None,
    snapshot: SiteConfigSnapshot | None = None,
    epoch_window_s: float = 0.25,
    epoch_flush_interval_s: float | None = None,
) -> tuple[RfIngressRuntime, SimulatorAdapter, InMemoryTagFixPublisher, SiteConfigSnapshot]:
    snap = snapshot if snapshot is not None else _snapshot()
    provider = InMemorySiteConfigProvider(snap)
    scenario = load_scenario(SCENARIOS_DIR / "walker_uwb.yaml")
    adapter = SimulatorAdapter(
        adapter_id="sim-1", deps=AdapterDeps(site_config=provider), scenario=scenario
    )
    pub = publisher if publisher is not None else InMemoryTagFixPublisher()
    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=provider,
        publisher=pub,
        epoch_window_s=epoch_window_s,
        epoch_flush_interval_s=epoch_flush_interval_s,
        clock=clock,
    )
    return runtime, adapter, pub, snap


def _range(
    node: RfNodeGeometry,
    *,
    epoch_seq: int,
    measured_at: datetime,
    bias_m: float = 0.0,
) -> RangeMeasurement:
    distance = math.sqrt(
        (_TRUE_XY[0] - node.x_m) ** 2 + (_TRUE_XY[1] - node.y_m) ** 2 + (1.0 - node.z_m) ** 2
    )
    return RangeMeasurement(
        tag_id=_TAG,
        anchor_id=node.anchor_id,
        distance_m=distance + bias_m,
        variance_m2=0.01,
        technology=RangingTechnology.UWB,
        measured_at=measured_at,
        adapter_id="sim-1",
        epoch_seq=epoch_seq,
    )


async def test_flush_emits_final_burst_epoch_without_a_successor() -> None:
    clock = _ManualClock(_T0)
    runtime, adapter, publisher, snap = _build(clock=clock)
    # Last epoch of a burst: 3 of the room's 4 anchors heard, then the watch goes quiet.
    for node in snap.nodes[:3]:
        out = await runtime.handle_range_measurement(
            adapter, _range(node, epoch_seq=12, measured_at=_T0), arrival_at=clock()
        )
        assert out == []

    clock.advance(0.20)
    assert await runtime.flush_expired_epochs() == []
    clock.advance(0.06)
    emitted = await runtime.flush_expired_epochs()
    assert len(emitted) == 1
    assert emitted[0].kind == FixKind.POSITION
    assert emitted[0].epoch_seq == 12
    assert emitted[0].anchors_used == 3
    assert len(publisher.fixes) == 1
    await adapter.close()


async def test_flush_tick_between_measurements_of_one_batch_does_not_split_it() -> None:
    """A micro-batch arrives seconds after capture; a tick inside it must not split it."""
    clock = _ManualClock(_T0)
    runtime, adapter, publisher, snap = _build(clock=clock)
    captured_at = _T0 - timedelta(seconds=2.5)
    nodes = list(snap.nodes)
    await runtime.handle_range_measurement(
        adapter, _range(nodes[0], epoch_seq=5, measured_at=captured_at), arrival_at=clock()
    )
    await runtime.handle_range_measurement(
        adapter, _range(nodes[1], epoch_seq=5, measured_at=captured_at), arrival_at=clock()
    )
    clock.advance(0.01)
    assert await runtime.flush_expired_epochs() == []
    await runtime.handle_range_measurement(
        adapter, _range(nodes[2], epoch_seq=5, measured_at=captured_at), arrival_at=clock()
    )
    emitted = await runtime.handle_range_measurement(
        adapter, _range(nodes[3], epoch_seq=5, measured_at=captured_at), arrival_at=clock()
    )
    assert len(emitted) == 1
    assert emitted[0].anchors_used == 4
    assert [p.fix.epoch_seq for p in publisher.fixes] == [5]
    await adapter.close()


async def test_position_fix_stamps_inlier_room_with_probability() -> None:
    clock = _ManualClock(_T0)
    runtime, adapter, publisher, snap = _build(clock=clock)
    for node in snap.nodes:
        await runtime.handle_range_measurement(
            adapter, _range(node, epoch_seq=1, measured_at=_T0), arrival_at=clock()
        )
    assert len(publisher.fixes) == 1
    fix = publisher.fixes[0].fix
    assert fix.kind == FixKind.POSITION
    assert fix.room_id == "1"
    assert fix.room_probability is not None
    assert 0.5 < fix.room_probability < 1.0
    assert publisher.fixes[0].proto.room_id == "1"
    assert publisher.fixes[0].proto.has_room_probability
    await adapter.close()


async def test_position_fix_without_node_rooms_keeps_room_empty() -> None:
    clock = _ManualClock(_T0)
    base = _snapshot()
    snap = replace(base, nodes=tuple(replace(n, room_id=None) for n in base.nodes))
    runtime, adapter, publisher, _ = _build(clock=clock, snapshot=snap)
    for node in snap.nodes:
        await runtime.handle_range_measurement(
            adapter, _range(node, epoch_seq=1, measured_at=_T0), arrival_at=clock()
        )
    assert len(publisher.fixes) == 1
    fix = publisher.fixes[0].fix
    assert fix.kind == FixKind.POSITION
    assert fix.room_id == ""
    assert fix.room_probability is None
    assert not publisher.fixes[0].proto.has_room_probability
    await adapter.close()


async def test_flush_survives_a_publisher_failure() -> None:
    clock = _ManualClock(_T0)
    runtime, adapter, publisher, snap = _build(clock=clock, publisher=_FailingOncePublisher())
    before = FIXES_DROPPED_TOTAL.labels(reason=DropReason.PUBLISH_FAILED)._value.get()
    for seq in (1, 2):
        for node in snap.nodes[:3]:
            await runtime.handle_range_measurement(
                adapter,
                _range(node, epoch_seq=seq, measured_at=clock()),
                arrival_at=clock(),
            )
        clock.advance(0.3)
        await runtime.flush_expired_epochs()
    after = FIXES_DROPPED_TOTAL.labels(reason=DropReason.PUBLISH_FAILED)._value.get()
    assert after == before + 1
    assert [p.fix.epoch_seq for p in publisher.fixes] == [2]
    await adapter.close()


async def test_late_straggler_is_counted_and_not_published() -> None:
    clock = _ManualClock(_T0)
    runtime, adapter, publisher, snap = _build(clock=clock)
    before = FIXES_DROPPED_TOTAL.labels(reason=DropReason.LATE_EPOCH_MEASUREMENT)._value.get()
    for node in snap.nodes[:3]:
        await runtime.handle_range_measurement(
            adapter, _range(node, epoch_seq=3, measured_at=_T0), arrival_at=clock()
        )
    clock.advance(0.3)
    assert len(await runtime.flush_expired_epochs()) == 1
    clock.advance(0.05)
    out = await runtime.handle_range_measurement(
        adapter, _range(snap.nodes[3], epoch_seq=3, measured_at=_T0), arrival_at=clock()
    )
    clock.advance(0.3)
    assert out == []
    assert await runtime.flush_expired_epochs() == []
    after = FIXES_DROPPED_TOTAL.labels(reason=DropReason.LATE_EPOCH_MEASUREMENT)._value.get()
    assert after == before + 1
    assert len(publisher.fixes) == 1
    await adapter.close()


async def test_started_runtime_flushes_the_last_epoch_in_the_background() -> None:
    """The flush loop started by ``start()`` publishes a stranded epoch on its own."""
    base = _snapshot()
    # Wall-clock time so the background loop sees time pass; bind the tag from an hour ago.
    live_snap = replace(
        base,
        bindings=tuple(
            replace(b, valid_from=datetime.now(UTC) - timedelta(hours=1)) for b in base.bindings
        ),
    )
    runtime, adapter, publisher, snap = _build(
        clock=lambda: datetime.now(UTC),
        snapshot=live_snap,
        epoch_window_s=0.05,
        epoch_flush_interval_s=0.01,
    )
    await runtime.start()
    try:
        now = datetime.now(UTC)
        for node in snap.nodes[:3]:
            await runtime.handle_range_measurement(
                adapter, _range(node, epoch_seq=1, measured_at=now), arrival_at=now
            )
        for _ in range(100):
            if publisher.fixes:
                break
            await asyncio.sleep(0.01)
        assert len(publisher.fixes) == 1
        assert publisher.fixes[0].fix.kind == FixKind.POSITION
    finally:
        await runtime.stop()


def test_flush_interval_defaults_and_validation() -> None:
    clock = _ManualClock(_T0)
    runtime, _, _, _ = _build(clock=clock, epoch_window_s=1.0)
    assert runtime._epoch_flush_interval_s == pytest.approx(0.05)
    runtime_short, _, _, _ = _build(clock=clock, epoch_window_s=0.1)
    assert runtime_short._epoch_flush_interval_s == pytest.approx(0.025)
    with pytest.raises(ValueError, match="epoch_flush_interval_s"):
        _build(clock=clock, epoch_flush_interval_s=0.0)
