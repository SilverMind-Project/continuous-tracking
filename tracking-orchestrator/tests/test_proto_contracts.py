"""M6: Proto contract round-trip tests for identity_snapshots and signal metadata."""

from __future__ import annotations

import json

import pytest

from app.proto.continuoustracking.v1 import signals_pb2, tracking_pb2


class TestIdentitySnapshotRoundTrip:
    """IdentitySnapshot proto encode/decode."""

    def test_encode_decode_identity_snapshot(self) -> None:
        event = tracking_pb2.TrackingEvent(
            camera_id="cam-1",
            event_id="ev-1",
            room_name="living_room",
        )

        snap = event.identity_snapshots.add()
        snap.global_track_id = "gt-001"
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
        assert s.global_track_id == "gt-001"
        assert s.identity_id == "alice"
        assert s.top_probability == pytest.approx(0.94)
        assert s.second_probability == pytest.approx(0.03)
        assert s.posterior_entropy == pytest.approx(0.32)
        assert s.direct_face_evidence is True
        assert "direct_face" in s.evidence_json

    def test_multiple_identity_snapshots(self) -> None:
        event = tracking_pb2.TrackingEvent(camera_id="cam-1", event_id="ev-1")

        for gt_id, identity_id in [("gt-1", "alice"), ("gt-2", "bob"), ("gt-3", "")]:
            snap = event.identity_snapshots.add()
            snap.global_track_id = gt_id
            snap.identity_id = identity_id

        data = event.SerializeToString()
        parsed = tracking_pb2.TrackingEvent.FromString(data)

        assert len(parsed.identity_snapshots) == 3
        assert parsed.identity_snapshots[0].identity_id == "alice"
        assert parsed.identity_snapshots[2].identity_id == ""

    def test_legacy_identity_revisions_still_present(self) -> None:
        """Field 5 (identity_revisions) uses ph_id (N0 rename from global_track_id)."""
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
