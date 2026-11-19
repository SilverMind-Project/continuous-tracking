"""Unit tests for protobuf encoding, presence flags, MINID calculation, and publishers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from cts_contracts.wearables import AdapterKind, FixKind, MotionState, WearState

from rf_ingress.domain import AdapterCapabilityDescriptor, EmittedFix
from rf_ingress.observability import OFF_BODY_FIXES_TOTAL
from rf_ingress.proto.continuoustracking.v1 import wearable_pb2
from rf_ingress.publish import (
    InMemoryTagFixPublisher,
    TagFixContractError,
    build_capability_proto,
    build_tag_fix_proto,
    compute_minid_ms,
)

_T0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)


def test_build_tag_fix_proto_preserves_origin_zero_and_presence_flags() -> None:
    fix = EmittedFix(
        tag_id="tag-1",
        measured_at=_T0,
        kind=FixKind.POSITION,
        epoch_seq=4,
        adapter_id="sim-1",
        adapter_kind=AdapterKind.SIMULATOR,
        adapter_version="0.1.0",
        floor_plan_id="household:1",
        x_mm=0,
        y_mm=0,
        z_mm=1000,
        cov_xx_m2=0.04,
        cov_xy_m2=0.01,
        cov_yy_m2=0.09,
        wear_state=WearState.OFF_BODY,
        motion_state=MotionState.STILL,
        quality=0.82,
        anchors_used=4,
        residual_rms_m=0.06,
        battery_fraction=0.75,
    )
    proto = build_tag_fix_proto(fix)
    raw = proto.SerializeToString()

    decoded = wearable_pb2.TagFix()
    decoded.ParseFromString(raw)
    assert decoded.tag_id == "tag-1"
    assert decoded.has_position is True
    assert decoded.x_mm == 0 and decoded.y_mm == 0
    assert decoded.has_z is True and decoded.z_mm == 1000
    assert decoded.has_room_probability is False
    assert decoded.has_distance is False
    assert decoded.has_quality is True
    assert decoded.has_residual is True
    assert decoded.has_battery is True
    assert decoded.wear_state == wearable_pb2.WEAR_STATE_OFF_BODY
    assert decoded.motion_state == wearable_pb2.MOTION_STATE_STILL


def test_validate_rejects_non_positive_definite_covariance() -> None:
    bad_fix = EmittedFix(
        tag_id="tag-1",
        measured_at=_T0,
        kind=FixKind.POSITION,
        epoch_seq=1,
        adapter_id="sim-1",
        adapter_kind=AdapterKind.SIMULATOR,
        adapter_version="0.1.0",
        x_mm=100,
        y_mm=200,
        cov_xx_m2=0.04,
        cov_xy_m2=0.04,
        cov_yy_m2=0.04,  # det == 0
    )
    with pytest.raises(TagFixContractError, match="positive-definite"):
        build_tag_fix_proto(bad_fix)


def test_compute_minid_ms_matches_72_hour_cutoff() -> None:
    expected_ms = int((_T0 - timedelta(hours=72)).timestamp() * 1000)
    assert compute_minid_ms(_T0, 72.0) == f"{expected_ms}-0"


async def test_in_memory_publisher_records_minid_off_body_metric_and_capability() -> None:
    pub = InMemoryTagFixPublisher(clock=lambda: _T0)
    before = OFF_BODY_FIXES_TOTAL.labels(adapter_id="sim-1")._value.get()
    fix = EmittedFix(
        tag_id="tag-1",
        measured_at=_T0,
        kind=FixKind.ROOM,
        epoch_seq=2,
        adapter_id="sim-1",
        adapter_kind=AdapterKind.SIMULATOR,
        adapter_version="0.1.0",
        room_id="1",
        room_probability=0.85,
        wear_state=WearState.OFF_BODY,
        motion_state=MotionState.STILL,
    )
    await pub.publish(fix)
    assert len(pub.fixes) == 1
    assert pub.fixes[0].minid == compute_minid_ms(_T0, 72.0)
    assert OFF_BODY_FIXES_TOTAL.labels(adapter_id="sim-1")._value.get() == before + 1

    cap = AdapterCapabilityDescriptor(
        adapter_id="sim-1",
        adapter_kind=AdapterKind.SIMULATOR,
        adapter_version="0.1.0",
        fix_kinds=(FixKind.POSITION, FixKind.ROOM),
        nominal_rate_hz=1.0,
        reports_covariance=True,
        reports_wear_state=True,
        reports_z=True,
        reports_motion_state=True,
    )
    await pub.publish_capability(cap, reported_at=_T0)
    assert len(pub.capabilities) == 1
    decoded_cap = build_capability_proto(cap, reported_at=_T0)
    assert decoded_cap.adapter_id == "sim-1"
    assert decoded_cap.reports_motion_state is True
