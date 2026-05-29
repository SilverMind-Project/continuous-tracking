"""Tests for the proto-only Redis-Streams codec."""

from __future__ import annotations

import pytest

from app.proto.continuoustracking.v1 import frame_pb2, tracking_pb2
from app.transport.codec import decode, encode


def _sample_event() -> tracking_pb2.TrackingEvent:
    ev = tracking_pb2.TrackingEvent(
        camera_id="cam-1",
        event_id="evt-1",
        event_time_unix_ns=1000,
        room_name="Kitchen",
    )
    det = ev.detections.add(
        detection_id="d1",
        confidence=0.9,
        ph_id="gt-1",
    )
    det.bbox.x_min, det.bbox.y_min = 10, 20
    det.bbox.x_max, det.bbox.y_max = 30, 40
    det.floor_point.x_mm = 1234
    det.floor_point.y_mm = 5678
    det.floor_point.calibrated = True
    return ev


def test_encode_packs_payload_under_named_field() -> None:
    fields = encode(_sample_event(), field="event")
    assert set(fields.keys()) == {"event"}
    assert isinstance(fields["event"], bytes)


def test_decode_round_trips_proto() -> None:
    fields = encode(_sample_event(), field="event")
    parsed = decode(fields, tracking_pb2.TrackingEvent, field="event")
    assert parsed.camera_id == "cam-1"
    assert parsed.event_id == "evt-1"
    assert parsed.room_name == "Kitchen"
    assert parsed.detections[0].floor_point.x_mm == 1234


def test_decode_accepts_bytes_keys() -> None:
    fields = encode(_sample_event(), field="event")
    bytes_fields = {k.encode(): v for k, v in fields.items()}
    parsed = decode(bytes_fields, tracking_pb2.TrackingEvent, field="event")
    assert parsed.camera_id == "cam-1"


def test_decode_raises_when_field_missing() -> None:
    with pytest.raises(ValueError, match="missing 'event' field"):
        decode({"other": b""}, tracking_pb2.TrackingEvent, field="event")


def test_codec_works_for_arbitrary_message_types() -> None:
    frame = frame_pb2.FrameReady(
        camera_id="cam-2",
        minio_key="frames/cam-2/1.jpg",
        frame_index=7,
        capture_time_unix_ns=1000,
        received_time_unix_ns=2000,
        width=640,
        height=480,
        sample_fps=2.0,
    )
    fields = encode(frame, field="frame")
    parsed = decode(fields, frame_pb2.FrameReady, field="frame")
    assert parsed.frame_index == 7
    assert parsed.sample_fps == pytest.approx(2.0)
