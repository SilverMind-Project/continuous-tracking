"""Proto contract round-trip tests for identity_snapshots and signal metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from cts_contracts import (
    AdapterKind,
    FixKind,
    MotionState,
    RangingTechnology,
    RfNodeRole,
    WearState,
)

from app.domain import IdentityEvidence, IdentityRevision
from app.proto.continuoustracking.v1 import (
    position_sensor_pb2,
    signals_pb2,
    tracking_pb2,
    wearable_pb2,
)
from app.transport.revision_publisher import _to_proto
from app.transport.wearable_contract import WearableContractError, validate_tag_fix


class TestIdentitySnapshotRoundTrip:
    """IdentitySnapshot proto encode/decode."""

    def test_encode_decode_identity_snapshot(self) -> None:
        event = tracking_pb2.TrackingEvent(
            camera_id="cam-1",
            event_id="ev-1",
            room_name="living_room",
        )

        snap = event.identity_snapshots.add()
        snap.ph_id = "ph-001"  # renamed from ph_id
        snap.identity_id = "alice"
        snap.top_probability = 0.94
        snap.second_probability = 0.03
        snap.posterior_entropy = 0.32
        snap.direct_face_evidence = True
        snap.evidence_json = json.dumps(
            {
                "sources": {"direct_face": 1},
                "direct_face_confidence": 0.95,
            }
        )

        # Round-trip.
        data = event.SerializeToString()
        parsed = tracking_pb2.TrackingEvent.FromString(data)

        assert len(parsed.identity_snapshots) == 1
        s = parsed.identity_snapshots[0]
        assert s.ph_id == "ph-001"  # renamed from ph_id
        assert s.identity_id == "alice"
        assert s.top_probability == pytest.approx(0.94)
        assert s.second_probability == pytest.approx(0.03)
        assert s.posterior_entropy == pytest.approx(0.32)
        assert s.direct_face_evidence is True
        assert "direct_face" in s.evidence_json

    def test_multiple_identity_snapshots(self) -> None:
        event = tracking_pb2.TrackingEvent(camera_id="cam-1", event_id="ev-1")

        for ph_id, identity_id in [("ph-1", "alice"), ("ph-2", "bob"), ("ph-3", "")]:
            snap = event.identity_snapshots.add()
            snap.ph_id = ph_id  # renamed from ph_id
            snap.identity_id = identity_id

        data = event.SerializeToString()
        parsed = tracking_pb2.TrackingEvent.FromString(data)

        assert len(parsed.identity_snapshots) == 3
        assert parsed.identity_snapshots[0].identity_id == "alice"
        assert parsed.identity_snapshots[2].identity_id == ""

    def test_legacy_identity_revisions_still_present(self) -> None:
        """Field 5 (identity_revisions) uses ph_id (N0 rename from ph_id)."""
        event = tracking_pb2.TrackingEvent(camera_id="cam-1", event_id="ev-1")

        rev = event.identity_revisions.add()
        rev.ph_id = "ph-1"
        rev.map_identity_id = "alice"
        rev.candidates.add(identity_id="alice", probability=0.95)

        data = event.SerializeToString()
        parsed = tracking_pb2.TrackingEvent.FromString(data)

        assert len(parsed.identity_revisions) == 1
        assert parsed.identity_revisions[0].ph_id == "ph-1"
        assert parsed.identity_revisions[0].map_identity_id == "alice"

    def test_inferred_backfill_revision_round_trip(self) -> None:
        """M04: an inferred_backfill revision carries its range through the wire.

        Fields 18-25 (revision_kind, range_start/end_unix_ns, range_authority,
        revision_range_id, correction_id, required_projections,
        revision_schema_version) must all survive serialize/parse.
        """
        range_start = datetime(2026, 7, 20, 8, 0, 0, tzinfo=UTC)
        range_end = datetime(2026, 7, 20, 11, 0, 0, tzinfo=UTC)
        revision = IdentityRevision(
            revision_id="rev-1",
            ph_id="ph-1",
            previous_identity_id=None,
            new_identity_id="alice",
            actor="system",
            reason="unknown_backfill",
            applied_at=range_end,
            rewritten_rows=0,
            evidence=IdentityEvidence(evidence_sources=["direct_face"]),
            revision_kind="inferred_backfill",
            range_start=range_start,
            range_end=range_end,
            range_authority="inferred",
            revision_range_id="rev-1",
            required_projections=("cc",),
            revision_schema_version="1",
        )

        pb = _to_proto(revision)
        data = pb.SerializeToString()
        parsed = tracking_pb2.IdentityRevision.FromString(data)

        assert parsed.ph_id == "ph-1"
        assert parsed.previous_identity_id == ""
        assert parsed.new_identity_id == "alice"
        assert parsed.revision_kind == "inferred_backfill"
        assert parsed.range_start_unix_ns == int(range_start.timestamp() * 1e9)
        assert parsed.range_end_unix_ns == int(range_end.timestamp() * 1e9)
        assert parsed.range_authority == "inferred"
        assert parsed.revision_range_id == "rev-1"
        assert list(parsed.required_projections) == ["cc"]
        assert parsed.revision_schema_version == "1"


class TestDetectionFloorPointRoundTrip:
    """Floor points must round-trip through proto encoding."""

    def test_calibrated_floor_point_round_trip(self) -> None:
        event = tracking_pb2.TrackingEvent(camera_id="cam-1", event_id="ev-1")

        det = event.detections.add(detection_id="d-1")
        det.floor_point.x_mm = 3500
        det.floor_point.y_mm = -1200
        det.floor_point.calibrated = True
        det.floor_x = 3.5
        det.floor_y = -1.2

        data = event.SerializeToString()
        parsed = tracking_pb2.TrackingEvent.FromString(data)

        assert len(parsed.detections) == 1
        d = parsed.detections[0]
        assert d.floor_point.x_mm == 3500
        assert d.floor_point.y_mm == -1200
        assert d.floor_point.calibrated is True
        assert d.floor_x == pytest.approx(3.5)
        assert d.floor_y == pytest.approx(-1.2)

    def test_uncalibrated_floor_point_round_trip(self) -> None:
        """Uncalibrated floor points should be all zeros."""
        event = tracking_pb2.TrackingEvent(camera_id="cam-1", event_id="ev-1")

        det = event.detections.add(detection_id="d-1")
        det.floor_point.x_mm = 0
        det.floor_point.y_mm = 0
        det.floor_point.calibrated = False

        data = event.SerializeToString()
        parsed = tracking_pb2.TrackingEvent.FromString(data)

        d = parsed.detections[0]
        assert d.floor_point.calibrated is False
        assert d.floor_point.x_mm == 0
        assert d.floor_x == 0.0


class TestSignalAlgorithmMetadata:
    """Algorithm metadata must round-trip through signal proto."""

    def test_signal_algorithm_metadata_round_trip(self) -> None:
        sig = signals_pb2.DementiaSignal()
        sig.signal_id = "sig-1"
        sig.identity_id = "alice"
        sig.algorithm_version = 2
        sig.algorithm_name = "stillness-v2"
        sig.evidence_grade = "clinical_review"
        sig.algorithm_spec_json = json.dumps(
            {
                "stillness_threshold_minutes": 60,
                "motion_floor": 0.02,
            }
        )

        data = sig.SerializeToString()
        parsed = signals_pb2.DementiaSignal.FromString(data)

        assert parsed.algorithm_version == 2
        assert parsed.algorithm_name == "stillness-v2"
        assert parsed.evidence_grade == "clinical_review"
        assert "stillness_threshold_minutes" in parsed.algorithm_spec_json

    def test_signal_backward_compat_empty_metadata(self) -> None:
        """Old signals without algorithm metadata should decode correctly."""
        sig = signals_pb2.DementiaSignal()
        sig.signal_id = "sig-old"
        sig.identity_id = "bob"
        sig.algorithm_version = 1

        data = sig.SerializeToString()
        parsed = signals_pb2.DementiaSignal.FromString(data)

        assert parsed.algorithm_name == ""
        assert parsed.evidence_grade == ""
        assert parsed.algorithm_spec_json == ""


def _proto_enum_wire_values(
    enum_desc: object, *, prefix: str, include_unspecified: bool = False
) -> list[str]:
    values = []
    for val in enum_desc.values:  # type: ignore[attr-defined]
        assert val.name.startswith(prefix), val.name
        stripped = val.name[len(prefix) :].lower()
        if not include_unspecified and val.number == 0:
            continue
        values.append(stripped)
    return values


class TestWearableProtoContracts:
    """TagFix, AdapterCapability, AnonymousPositionTrack, and enum parity."""

    def test_cts_contracts_enums_match_wearable_proto(self) -> None:
        assert [m.value for m in FixKind] == _proto_enum_wire_values(
            wearable_pb2.FixKind.DESCRIPTOR, prefix="FIX_KIND_"
        )
        assert [m.value for m in AdapterKind] == _proto_enum_wire_values(
            wearable_pb2.AdapterKind.DESCRIPTOR, prefix="ADAPTER_KIND_"
        )
        assert [m.value for m in WearState] == _proto_enum_wire_values(
            wearable_pb2.WearState.DESCRIPTOR, prefix="WEAR_STATE_", include_unspecified=True
        )
        assert [m.value for m in MotionState] == _proto_enum_wire_values(
            wearable_pb2.MotionState.DESCRIPTOR, prefix="MOTION_STATE_", include_unspecified=True
        )
        assert [m.value for m in RfNodeRole] == _proto_enum_wire_values(
            wearable_pb2.RfNodeRole.DESCRIPTOR, prefix="RF_NODE_ROLE_"
        )
        assert [m.value for m in RangingTechnology] == _proto_enum_wire_values(
            wearable_pb2.RangingTechnology.DESCRIPTOR, prefix="RANGING_TECHNOLOGY_"
        )

    def test_tag_fix_position_at_origin_and_optional_scalars_round_trip(self) -> None:
        # (0, 0) is a real floor coordinate distinguished by has_position=True.
        fix = wearable_pb2.TagFix(
            tag_id="tag-watch-1",
            measured_at_unix_ns=1_794_600_000_000_000_000,
            kind=wearable_pb2.FIX_KIND_POSITION,
            epoch_seq=42,
            has_position=True,
            x_mm=0,
            y_mm=0,
            has_z=True,
            z_mm=1050,
            cov_xx_m2=0.04,
            cov_xy_m2=0.01,
            cov_yy_m2=0.09,
            floor_plan_id="floor-plans/main.png",
            room_id="1",
            has_room_probability=True,
            room_probability=0.92,
            anchor_id="node-lr-1",
            has_distance=True,
            distance_m=1.45,
            wear_state=wearable_pb2.WEAR_STATE_ON_BODY,
            has_quality=True,
            quality=0.88,
            anchors_used=4,
            has_residual=True,
            residual_rms_m=0.12,
            adapter_id="uwb-watch",
            adapter_kind=wearable_pb2.ADAPTER_KIND_DEVICE_RANGING,
            adapter_version="1.0.0",
            motion_state=wearable_pb2.MOTION_STATE_MOVING,
            has_battery=True,
            battery_fraction=0.73,
        )
        validate_tag_fix(fix)

        parsed = wearable_pb2.TagFix.FromString(fix.SerializeToString())
        assert parsed.tag_id == "tag-watch-1"
        assert parsed.measured_at_unix_ns == 1_794_600_000_000_000_000
        assert parsed.kind == wearable_pb2.FIX_KIND_POSITION
        assert parsed.epoch_seq == 42
        assert parsed.has_position is True
        assert parsed.x_mm == 0
        assert parsed.y_mm == 0
        assert parsed.has_z is True
        assert parsed.z_mm == 1050
        assert parsed.cov_xx_m2 == pytest.approx(0.04)
        assert parsed.cov_xy_m2 == pytest.approx(0.01)
        assert parsed.cov_yy_m2 == pytest.approx(0.09)
        assert parsed.floor_plan_id == "floor-plans/main.png"
        assert parsed.room_id == "1"
        assert parsed.has_room_probability is True
        assert parsed.room_probability == pytest.approx(0.92)
        assert parsed.anchor_id == "node-lr-1"
        assert parsed.has_distance is True
        assert parsed.distance_m == pytest.approx(1.45)
        assert parsed.wear_state == wearable_pb2.WEAR_STATE_ON_BODY
        assert parsed.has_quality is True
        assert parsed.quality == pytest.approx(0.88)
        assert parsed.anchors_used == 4
        assert parsed.has_residual is True
        assert parsed.residual_rms_m == pytest.approx(0.12)
        assert parsed.adapter_id == "uwb-watch"
        assert parsed.adapter_kind == wearable_pb2.ADAPTER_KIND_DEVICE_RANGING
        assert parsed.adapter_version == "1.0.0"
        assert parsed.motion_state == wearable_pb2.MOTION_STATE_MOVING
        assert parsed.has_battery is True
        assert parsed.battery_fraction == pytest.approx(0.73)

    def test_tag_fix_position_rejects_missing_or_non_pd_covariance(self) -> None:
        base = wearable_pb2.TagFix(
            tag_id="tag-1",
            measured_at_unix_ns=1_794_600_000_000_000_000,
            kind=wearable_pb2.FIX_KIND_POSITION,
            has_position=True,
            x_mm=1500,
            y_mm=2000,
        )
        with pytest.raises(WearableContractError, match="covariance"):
            validate_tag_fix(base)

        non_pd = wearable_pb2.TagFix(
            tag_id="tag-1",
            measured_at_unix_ns=1_794_600_000_000_000_000,
            kind=wearable_pb2.FIX_KIND_POSITION,
            has_position=True,
            x_mm=1500,
            y_mm=2000,
            cov_xx_m2=1.0,
            cov_xy_m2=1.0,
            cov_yy_m2=1.0,
        )
        with pytest.raises(WearableContractError, match="positive-definite"):
            validate_tag_fix(non_pd)

        no_pos = wearable_pb2.TagFix(
            tag_id="tag-1",
            measured_at_unix_ns=1_794_600_000_000_000_000,
            kind=wearable_pb2.FIX_KIND_POSITION,
            has_position=False,
            cov_xx_m2=1.0,
            cov_yy_m2=1.0,
        )
        with pytest.raises(WearableContractError, match="has_position"):
            validate_tag_fix(no_pos)

    def test_tag_fix_room_allows_unspecified_wear_and_motion(self) -> None:
        fix = wearable_pb2.TagFix(
            tag_id="beacon-1",
            measured_at_unix_ns=1_794_600_000_000_000_000,
            kind=wearable_pb2.FIX_KIND_ROOM,
            room_id="kitchen",
            wear_state=wearable_pb2.WEAR_STATE_UNSPECIFIED,
            motion_state=wearable_pb2.MOTION_STATE_UNSPECIFIED,
        )
        validate_tag_fix(fix)
        parsed = wearable_pb2.TagFix.FromString(fix.SerializeToString())
        assert parsed.has_position is False
        assert parsed.has_z is False
        assert parsed.has_battery is False
        assert parsed.wear_state == wearable_pb2.WEAR_STATE_UNSPECIFIED
        assert parsed.motion_state == wearable_pb2.MOTION_STATE_UNSPECIFIED

    def test_adapter_capability_round_trip(self) -> None:
        cap = wearable_pb2.AdapterCapability(
            adapter_id="ble-home",
            adapter_kind=wearable_pb2.ADAPTER_KIND_ROOM_PRESENCE,
            fix_kinds=[wearable_pb2.FIX_KIND_ROOM, wearable_pb2.FIX_KIND_POSITION],
            nominal_rate_hz=1.0,
            reports_covariance=True,
            reports_wear_state=False,
            reports_z=False,
            reported_at_unix_ns=1_794_600_000_000_000_000,
            reports_motion_state=True,
        )
        parsed = wearable_pb2.AdapterCapability.FromString(cap.SerializeToString())
        assert parsed.adapter_id == "ble-home"
        assert parsed.adapter_kind == wearable_pb2.ADAPTER_KIND_ROOM_PRESENCE
        assert list(parsed.fix_kinds) == [
            wearable_pb2.FIX_KIND_ROOM,
            wearable_pb2.FIX_KIND_POSITION,
        ]
        assert parsed.nominal_rate_hz == pytest.approx(1.0)
        assert parsed.reports_covariance is True
        assert parsed.reports_wear_state is False
        assert parsed.reports_z is False
        assert parsed.reports_motion_state is True

    def test_anonymous_position_track_round_trip(self) -> None:
        track = position_sensor_pb2.AnonymousPositionTrack(
            sensor_id="radar-bath-1",
            sensor_track_id="t-7",
            measured_at_unix_ns=1_794_600_000_000_000_000,
            x_mm=2100,
            y_mm=3400,
            cov_xx_m2=0.16,
            cov_xy_m2=0.0,
            cov_yy_m2=0.16,
            has_speed=True,
            speed_m_s=0.45,
            floor_plan_id="floor-plans/main.png",
        )
        parsed = position_sensor_pb2.AnonymousPositionTrack.FromString(track.SerializeToString())
        assert parsed.sensor_id == "radar-bath-1"
        assert parsed.sensor_track_id == "t-7"
        assert parsed.x_mm == 2100
        assert parsed.y_mm == 3400
        assert parsed.cov_xx_m2 == pytest.approx(0.16)
        assert parsed.has_speed is True
        assert parsed.speed_m_s == pytest.approx(0.45)
        assert parsed.floor_plan_id == "floor-plans/main.png"
