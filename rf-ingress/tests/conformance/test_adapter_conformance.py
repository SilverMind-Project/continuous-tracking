"""Adapter conformance suite: every rf-ingress adapter must pass unchanged.

Asserts strictly through the public adapter protocols (:class:`RangeAdapter`,
:class:`PositionAdapter`, :class:`RoomAdapter`) that:
1. Emitted reports carry only registered ``tag_id`` values; unregistered devices
   are dropped and counted.
2. Timestamps are timezone-aware and monotonic per tag within tolerance.
3. Every range carries a strictly positive ``variance_m2`` and a valid closed
   ``RangingTechnology``.
4. Malformed inputs are counted and never raise out of the adapter.
5. Declared :class:`AdapterCapabilityDescriptor` matches emitted reports
   (fix kinds, covariance, wear state, motion state).
6. ``close()`` shuts down cleanly and marks the adapter disconnected.
"""

from __future__ import annotations

import math
from datetime import datetime

from cts_contracts.wearables import FixKind, MotionState, RangingTechnology, WearState

from rf_ingress.adapters.base import PositionAdapter, RangeAdapter, RoomAdapter
from rf_ingress.domain import PositionReport, RangeMeasurement, RoomReport
from rf_ingress.observability import FIXES_DROPPED_TOTAL, UNREGISTERED_TAG_TOTAL, DropReason
from tests.conformance.conftest import AdapterConformanceContext

# Allow minor batch reordering tolerance (seconds) when comparing timestamps per tag.
_MONOTONIC_TOLERANCE_S = 1.0


async def _collect_all(
    ctx: AdapterConformanceContext,
) -> tuple[list[RangeMeasurement], list[PositionReport], list[RoomReport]]:
    adapter = ctx.adapter
    ranges: list[RangeMeasurement] = []
    positions: list[PositionReport] = []
    rooms: list[RoomReport] = []
    if isinstance(adapter, RangeAdapter):
        async for r in adapter.ranges():
            ranges.append(r)
    if isinstance(adapter, PositionAdapter):
        async for p in adapter.positions():
            positions.append(p)
    if isinstance(adapter, RoomAdapter):
        async for rm in adapter.rooms():
            rooms.append(rm)
    return ranges, positions, rooms


async def test_conformance_emits_only_registered_tags_and_counts_unregistered(
    conformance_ctx: AdapterConformanceContext,
) -> None:
    before_unreg = UNREGISTERED_TAG_TOTAL.labels(
        adapter_id=conformance_ctx.adapter.adapter_id
    )._value.get()
    ranges, positions, rooms = await _collect_all(conformance_ctx)
    all_reports: list[RangeMeasurement | PositionReport | RoomReport] = [
        *ranges,
        *positions,
        *rooms,
    ]
    assert len(all_reports) > 0, "Conformance adapter must emit at least one valid report"

    emitted_tags = {r.tag_id for r in all_reports}
    assert emitted_tags.issubset(conformance_ctx.expected_registered_tags)
    assert emitted_tags.isdisjoint(conformance_ctx.expected_unregistered_tags)

    if conformance_ctx.expected_unregistered_tags:
        after_unreg = UNREGISTERED_TAG_TOTAL.labels(
            adapter_id=conformance_ctx.adapter.adapter_id
        )._value.get()
        assert after_unreg > before_unreg
        assert conformance_ctx.adapter.health().unregistered_reports > 0


async def test_conformance_timestamps_timezone_aware_and_monotonic_per_tag(
    conformance_ctx: AdapterConformanceContext,
) -> None:
    ranges, positions, rooms = await _collect_all(conformance_ctx)
    for stream in (ranges, positions, rooms):
        last_by_tag: dict[str, datetime] = {}
        for report in stream:
            assert report.measured_at.tzinfo is not None
            assert report.measured_at.utcoffset() is not None
            prev = last_by_tag.get(report.tag_id)
            if prev is not None:
                delta_s = (report.measured_at - prev).total_seconds()
                assert delta_s >= -_MONOTONIC_TOLERANCE_S, (
                    f"Non-monotonic timestamp for tag {report.tag_id}: delta={delta_s}s"
                )
            last_by_tag[report.tag_id] = report.measured_at


async def test_conformance_ranges_have_positive_variance_and_closed_technology(
    conformance_ctx: AdapterConformanceContext,
) -> None:
    ranges, _, _ = await _collect_all(conformance_ctx)
    for report in ranges:
        assert math.isfinite(report.variance_m2) and report.variance_m2 > 0.0
        assert math.isfinite(report.distance_m) and report.distance_m >= 0.0
        assert isinstance(report.technology, RangingTechnology)


async def test_conformance_malformed_input_counted_and_never_raises(
    conformance_ctx: AdapterConformanceContext,
) -> None:
    before_malformed = FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT)._value.get()
    ranges, positions, rooms = await _collect_all(conformance_ctx)
    assert len(ranges) + len(positions) + len(rooms) > 0
    if conformance_ctx.expects_malformed_drop:
        after_malformed = FIXES_DROPPED_TOTAL.labels(reason=DropReason.MALFORMED_INPUT)._value.get()
        assert after_malformed > before_malformed
        assert conformance_ctx.adapter.health().malformed_reports > 0


async def test_conformance_declared_capability_matches_emitted_reports(
    conformance_ctx: AdapterConformanceContext,
) -> None:
    cap = conformance_ctx.adapter.capability
    ranges, positions, rooms = await _collect_all(conformance_ctx)
    assert cap.adapter_id == conformance_ctx.adapter.adapter_id
    assert cap.nominal_rate_hz > 0.0
    assert len(cap.fix_kinds) > 0

    if ranges or positions:
        assert FixKind.POSITION in cap.fix_kinds
        assert cap.reports_covariance is True
    if rooms:
        assert FixKind.ROOM in cap.fix_kinds
    for pos in positions:
        if cap.reports_covariance:
            assert pos.cov_native_m2 is not None

    all_reports: list[RangeMeasurement | PositionReport | RoomReport] = [
        *ranges,
        *positions,
        *rooms,
    ]
    if not cap.reports_wear_state:
        assert all(r.wear_state == WearState.UNSPECIFIED for r in all_reports)
    if not cap.reports_motion_state:
        assert all(r.motion_state == MotionState.UNSPECIFIED for r in all_reports)


async def test_conformance_clean_shutdown(
    conformance_ctx: AdapterConformanceContext,
) -> None:
    adapter = conformance_ctx.adapter
    assert adapter.health().connected is True
    await adapter.close()
    assert adapter.health().connected is False
    # Iterating after close yields cleanly without raising.
    ranges, positions, rooms = await _collect_all(conformance_ctx)
    assert ranges == []
    assert positions == []
    assert rooms == []
