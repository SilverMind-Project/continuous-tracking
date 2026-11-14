"""Unit tests for the Redis Streams transport layer."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.proto.continuoustracking.v1 import frame_pb2
from app.storage.postgres.gallery_repo import (
    _embedding_to_pgvector,
    _pgvector_to_list,
)
from app.transport.redis_streams import (
    FrameReady,
    RedisStreamsTransport,
    TransportConfig,
)

# ---------------------------------------------------------------------------
# FrameReady deserialization tests
# ---------------------------------------------------------------------------


class TestFrameReady:
    def test_defaults(self) -> None:
        frame = FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/42.jpg",
            frame_index=42,
            capture_time_unix_ns=1700000000000000000,
            received_time_unix_ns=1700000000100000000,
            width=640,
            height=480,
        )
        assert frame.camera_id == "cam-1"
        assert frame.sample_fps == 0.0

    def test_with_fps(self) -> None:
        frame = FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/1.jpg",
            frame_index=1,
            capture_time_unix_ns=1000000000,
            received_time_unix_ns=1000000000,
            width=640,
            height=480,
            sample_fps=5.0,
        )
        assert frame.sample_fps == 5.0


# ---------------------------------------------------------------------------
# Transport deserialization tests
# ---------------------------------------------------------------------------


class TestTransportDeserialization:
    def _proto_frame_fields(self, **overrides) -> dict[bytes, bytes]:
        from app.proto.continuoustracking.v1 import frame_pb2

        defaults = {
            "camera_id": "cam-1",
            "minio_key": "frames/cam-1/5.jpg",
            "frame_index": 5,
            "capture_time_unix_ns": 1000000000,
            "received_time_unix_ns": 1000000001,
            "width": 640,
            "height": 480,
            "sample_fps": 5.0,
        }
        defaults.update(overrides)
        msg = frame_pb2.FrameReady(**defaults)
        return {b"frame": msg.SerializeToString()}

    def test_valid_fields(self) -> None:
        from app.transport.codec import decode as proto_decode

        frame = proto_decode(self._proto_frame_fields(), frame_pb2.FrameReady, field="frame")
        assert frame is not None
        assert frame.camera_id == "cam-1"
        assert frame.frame_index == 5
        assert frame.width == 640

    def test_missing_payload_returns_none(self) -> None:
        import pytest as _pytest

        from app.transport.codec import decode as proto_decode

        with _pytest.raises(ValueError, match="missing"):
            proto_decode({}, frame_pb2.FrameReady, field="frame")

    def test_invalid_proto_returns_none(self) -> None:
        import pytest as _pytest

        from app.transport.codec import decode as proto_decode

        with _pytest.raises(Exception):  # noqa: B017
            payload = {b"frame": b"not-valid-protobuf-\xff\x01"}
            proto_decode(payload, frame_pb2.FrameReady, field="frame")


# ---------------------------------------------------------------------------
# Transport config tests
# ---------------------------------------------------------------------------


class TestTransportConfig:
    def test_defaults(self) -> None:
        config = TransportConfig()
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.consumer_group == "cts-orchestrator"
        assert config.consumer_name == "orchestrator-1"
        assert config.frames_stream == "frames.ready"
        assert config.events_stream == "tracking.events"
        assert config.batch_max_wait_ms == 100
        assert config.batch_max_size == 8


# ---------------------------------------------------------------------------
# Transport lifecycle tests (no actual Redis)
# ---------------------------------------------------------------------------


class TestTransportLifecycle:
    def test_not_connected(self) -> None:
        transport = RedisStreamsTransport()
        assert not transport.is_connected

    def test_disconnect_without_connect(self) -> None:
        transport = RedisStreamsTransport()
        # Should not raise
        import asyncio

        asyncio.run(transport.disconnect())
        assert not transport.is_connected


# ---------------------------------------------------------------------------
# Embedding conversion tests (pgvector format)
# ---------------------------------------------------------------------------


class TestEmbeddingConversion:
    def test_to_pgvector(self) -> None:
        # This tests the helper function that was accidentally in the transport module
        # In production, it lives in storage/postgres/gallery_repo.py
        embedding = [0.1, -0.2, 0.3, 0.0]
        result = _embedding_to_pgvector(embedding)
        assert result.startswith("[")
        assert result.endswith("]")
        assert "0.10000000" in result
        assert "-0.20000000" in result

    def test_from_pgvector(self) -> None:
        result = _pgvector_to_list("[0.1,-0.2,0.3,0.0]")
        assert len(result) == 4
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(-0.2)

    def test_from_empty_pgvector(self) -> None:
        assert _pgvector_to_list("[]") == []
        assert _pgvector_to_list("") == []


# ---------------------------------------------------------------------------
# Transport data-path tests (mocked Redis)
# ---------------------------------------------------------------------------


class TestTransportDataPath:
    """Test consume_frames, ack_frame, publish_event with mocked Redis."""

    def _mock_transport(self) -> tuple[RedisStreamsTransport, AsyncMock]:
        transport = RedisStreamsTransport()
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.ResponseError = Exception
        transport._redis = mock_redis
        transport._group_created = True
        return transport, mock_redis

    @pytest.mark.asyncio
    async def test_consume_frames_no_redis(self) -> None:
        transport = RedisStreamsTransport()
        frames = []
        async for frame in transport.consume_frames(count=1):
            frames.append(frame)
        assert frames == []

    @pytest.mark.asyncio
    async def test_consume_frames_yields_frames(self) -> None:
        from app.proto.continuoustracking.v1 import frame_pb2

        transport, mock_redis = self._mock_transport()

        def _frame(idx: int) -> dict[bytes, bytes]:
            msg = frame_pb2.FrameReady(
                camera_id="cam-1",
                minio_key=f"frames/cam-1/{idx}.jpg",
                frame_index=idx,
                capture_time_unix_ns=1_000_000_000 + idx,
                received_time_unix_ns=1_000_000_001 + idx,
                width=640,
                height=480,
            )
            return {b"frame": msg.SerializeToString()}

        mock_redis.xreadgroup = AsyncMock(
            return_value=[
                (
                    b"frames.ready",
                    [
                        (b"1700000000000-0", _frame(1)),
                        (b"1700000000001-0", _frame(2)),
                    ],
                ),
            ]
        )

        frames = []
        async for frame in transport.consume_frames(count=2):
            frames.append(frame)
        assert len(frames) == 2
        assert frames[0].camera_id == "cam-1"
        assert frames[0].frame_index == 1
        assert frames[1].frame_index == 2

    @pytest.mark.asyncio
    async def test_consume_frames_empty_stream(self) -> None:
        transport, mock_redis = self._mock_transport()
        mock_redis.xreadgroup = AsyncMock(return_value=[])

        frames = []
        async for frame in transport.consume_frames(count=1):
            frames.append(frame)
        assert frames == []

    @pytest.mark.asyncio
    async def test_ack_frame_with_pending(self) -> None:
        transport, mock_redis = self._mock_transport()
        # Simulate a frame that was consumed (message ID stored in pending_acks).
        frame = FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/1.jpg",
            frame_index=1,
            capture_time_unix_ns=1000000000,
            received_time_unix_ns=1000000001,
            width=640,
            height=480,
        )
        key = (frame.camera_id, frame.frame_index, frame.capture_time_unix_ns)
        transport._pending_acks[key] = (
            "1700000000000-0",
            0.0,
        )

        await transport.ack_frame(frame)
        mock_redis.xack.assert_called_once_with(
            "frames.ready",
            "cts-orchestrator",
            "1700000000000-0",
        )

    @pytest.mark.asyncio
    async def test_ack_frame_no_pending_id(self) -> None:
        transport, mock_redis = self._mock_transport()
        frame = FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/1.jpg",
            frame_index=1,
            capture_time_unix_ns=1000000000,
            received_time_unix_ns=1000000001,
            width=640,
            height=480,
        )
        # No message ID stored → should not call xack.
        await transport.ack_frame(frame)
        mock_redis.xack.assert_not_called()

    @pytest.mark.asyncio
    async def test_ack_frame_not_connected(self) -> None:
        transport = RedisStreamsTransport()
        frame = FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/1.jpg",
            frame_index=1,
            capture_time_unix_ns=1000000000,
            received_time_unix_ns=1000000001,
            width=640,
            height=480,
        )
        # Should not raise.
        await transport.ack_frame(frame)

    @pytest.mark.asyncio
    async def test_publish_event(self) -> None:
        transport, mock_redis = self._mock_transport()
        mock_redis.xadd = AsyncMock(return_value="event-msg-1")

        msg_id = await transport.publish_event(
            camera_id="cam-1",
            event_time=datetime.now(UTC),
            frame_index=42,
            detection_count=3,
        )
        assert msg_id == "event-msg-1"
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "tracking.events"


class TestPublishEventProto:
    """tracking.events publishes a proto-only envelope."""

    def _mock_transport(self) -> tuple[RedisStreamsTransport, AsyncMock]:
        transport = RedisStreamsTransport()
        mock_redis = AsyncMock()
        transport._redis = mock_redis
        return transport, mock_redis

    def _sample_detections(self) -> list:
        from app.domain import BoundingBox, Detection, FloorPoint

        return [
            Detection(
                detection_id="d1",
                camera_id="cam-1",
                bbox=BoundingBox(10, 20, 30, 40),
                embedding=[0.0] * 4,
                capture_time=datetime.now(UTC),
                event_time=datetime.now(UTC),
                confidence=0.92,
                ph_id="gt-1",
                floor_point=FloorPoint(1234, 5678, calibrated=True),
            ),
        ]

    @pytest.mark.asyncio
    async def test_payload_is_single_proto_field(self) -> None:
        transport, mock_redis = self._mock_transport()
        mock_redis.xadd = AsyncMock(return_value=b"msg-1")

        await transport.publish_event(
            camera_id="cam-1",
            event_time=datetime.now(UTC),
            frame_index=42,
            detection_count=1,
            detections=self._sample_detections(),
            minio_key="frames/cam-1/42.jpg",
            room_name="Kitchen",
        )

        payload = mock_redis.xadd.call_args[0][1]
        assert set(payload.keys()) == {"event"}
        assert isinstance(payload["event"], bytes)

    @pytest.mark.asyncio
    async def test_proto_carries_identity_and_floor_point(self) -> None:
        from app.proto.continuoustracking.v1 import tracking_pb2

        transport, mock_redis = self._mock_transport()
        mock_redis.xadd = AsyncMock(return_value=b"msg-1")

        await transport.publish_event(
            camera_id="cam-1",
            event_time=datetime.now(UTC),
            frame_index=99,
            detection_count=1,
            detections=self._sample_detections(),
            minio_key="frames/cam-1/99.jpg",
            room_name="Bedroom",
        )

        payload = mock_redis.xadd.call_args[0][1]
        parsed = tracking_pb2.TrackingEvent.FromString(payload["event"])
        assert parsed.camera_id == "cam-1"
        assert parsed.room_name == "Bedroom"
        assert parsed.event_id  # populated with a fresh UUID
        assert parsed.frame_ref.frame_index == 99
        assert parsed.detections[0].floor_point.x_mm == 1234
        assert parsed.detections[0].floor_point.y_mm == 5678
        assert parsed.detections[0].floor_point.calibrated is True
        assert len(parsed.identity_revisions) == 0


class TestFrameReadyConsume:
    """The transport consumes proto-encoded FrameReady messages from ingress."""

    @pytest.mark.asyncio
    async def test_consume_proto_frame(self) -> None:
        from app.proto.continuoustracking.v1 import frame_pb2

        transport = RedisStreamsTransport()
        mock_redis = AsyncMock()
        transport._redis = mock_redis

        proto_frame = frame_pb2.FrameReady(
            camera_id="cam-2",
            minio_key="frames/cam-2/7.jpg",
            frame_index=7,
            capture_time_unix_ns=2000000000,
            received_time_unix_ns=2000000010,
            width=1280,
            height=720,
            sample_fps=2.5,
        )
        fields = {b"frame": proto_frame.SerializeToString()}
        mock_redis.xreadgroup = AsyncMock(
            return_value=[(b"frames.ready", [(b"0-1", fields)])],
        )

        frames = []
        async for frame in transport.consume_frames(count=1):
            frames.append(frame)

        assert len(frames) == 1
        assert frames[0].camera_id == "cam-2"
        assert frames[0].frame_index == 7
        assert frames[0].sample_fps == pytest.approx(2.5)


class TestPendingAcksEviction:
    """Tests for the _pending_acks TTL-based eviction (review issue #7)."""

    def test_cleanup_removes_stale_acks(self) -> None:
        """Verify that _cleanup_stale_acks removes entries older than the
        configured TTL."""
        import time

        config = TransportConfig(ack_ttl_seconds=10)
        transport = RedisStreamsTransport(config)

        # Manually inject a stale entry (timestamp = 20 seconds ago).
        stale_time = time.monotonic() - 20
        frame = FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/1.jpg",
            frame_index=1,
            capture_time_unix_ns=1000000000,
            received_time_unix_ns=1000000001,
            width=640,
            height=480,
        )
        key = (frame.camera_id, frame.frame_index, frame.capture_time_unix_ns)
        transport._pending_acks[key] = (
            "msg-1",
            stale_time,
        )

        # Also add a fresh entry.
        transport._pending_acks[("cam-1", 2, 2000000000)] = ("msg-2", time.monotonic())

        assert len(transport._pending_acks) == 2
        transport._cleanup_stale_acks()
        # Stale entry should be removed, fresh one remains.
        assert len(transport._pending_acks) == 1

    def test_cleanup_keeps_fresh_acks(self) -> None:
        """Verify that _cleanup_stale_acks does not remove entries within
        the TTL window."""
        import time

        config = TransportConfig(ack_ttl_seconds=300)
        transport = RedisStreamsTransport(config)

        frame = FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/1.jpg",
            frame_index=1,
            capture_time_unix_ns=1000000000,
            received_time_unix_ns=1000000001,
            width=640,
            height=480,
        )
        key = (frame.camera_id, frame.frame_index, frame.capture_time_unix_ns)
        transport._pending_acks[key] = (
            "msg-1",
            time.monotonic(),
        )

        transport._cleanup_stale_acks()
        assert len(transport._pending_acks) == 1

    @pytest.mark.asyncio
    async def test_pending_acks_key_does_not_collide_on_object_reuse(self) -> None:
        """Two distinct frames use value-based keys and never overwrite each other."""
        from app.proto.continuoustracking.v1 import frame_pb2

        transport = RedisStreamsTransport()
        mock_redis = AsyncMock()
        transport._redis = mock_redis

        pb1 = frame_pb2.FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/1.jpg",
            frame_index=1,
            capture_time_unix_ns=1000,
            received_time_unix_ns=1001,
            width=640,
            height=480,
        )
        pb2 = frame_pb2.FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/2.jpg",
            frame_index=2,
            capture_time_unix_ns=2000,
            received_time_unix_ns=2001,
            width=640,
            height=480,
        )
        mock_redis.xreadgroup = AsyncMock(
            return_value=[
                (
                    b"frames.ready",
                    [
                        (b"1-0", {b"frame": pb1.SerializeToString()}),
                        (b"2-0", {b"frame": pb2.SerializeToString()}),
                    ],
                )
            ]
        )

        consumed: list[FrameReady] = []
        async for f in transport.consume_frames(count=2):
            consumed.append(f)

        assert len(consumed) == 2
        assert len(transport._pending_acks) == 2

        # Re-create an equal value FrameReady instance at a new Python id() and ACK it.
        clone_f1 = FrameReady(
            camera_id="cam-1",
            minio_key="frames/cam-1/1.jpg",
            frame_index=1,
            capture_time_unix_ns=1000,
            received_time_unix_ns=1001,
            width=640,
            height=480,
        )
        assert id(clone_f1) != id(consumed[0])
        await transport.ack_frame(clone_f1)
        await transport.ack_frame(consumed[1])
        assert mock_redis.xack.await_count == 2
        assert len(transport._pending_acks) == 0

    @pytest.mark.asyncio
    async def test_connect_reclaims_stale_pending_entries(self, monkeypatch) -> None:
        """Startup connect() reclaims and XACKs stale pending entries via XAUTOCLAIM."""
        from prometheus_client import CollectorRegistry

        from app.observability import metrics as metrics_pkg
        from app.transport import redis_streams

        fresh = metrics_pkg.build_metrics(registry=CollectorRegistry())
        monkeypatch.setattr(metrics_pkg, "metrics", fresh)
        monkeypatch.setattr(redis_streams.metrics, "metrics", fresh)

        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock(return_value=True)
        mock_redis.xautoclaim = AsyncMock(
            side_effect=[
                (b"1-0", [], [b"0-1", b"0-2"]),
                (b"3-0", [(b"1-0", {b"frame": b"x"}), (b"2-0", {b"frame": b"y"})], []),
                (b"0-0", [(b"3-0", {b"frame": b"z"})], []),
            ]
        )
        mock_redis.xack = AsyncMock(return_value=2)

        with monkeypatch.context() as m:
            m.setattr(redis_streams.redis, "from_url", lambda *a, **kw: mock_redis)
            transport = RedisStreamsTransport(TransportConfig(pending_reclaim_min_idle_s=300))
            await transport.connect()

        assert mock_redis.xautoclaim.await_count == 3
        assert mock_redis.xack.await_count == 2
        assert fresh.frames_abandoned_total._value.get() == 5.0

    @pytest.mark.asyncio
    async def test_pending_count_reads_pending_and_lag_from_xinfo_groups(self) -> None:
        """pending_count() returns (pending, lag) from XINFO GROUPS and handles lag=None."""
        transport = RedisStreamsTransport(TransportConfig(consumer_group="cts-orchestrator"))
        mock_redis = AsyncMock()
        mock_redis.xinfo_groups = AsyncMock(
            return_value=[
                {b"name": b"other-group", b"pending": 99, b"lag": 50},
                {b"name": b"cts-orchestrator", b"pending": 12, b"lag": 34},
            ]
        )
        transport._redis = mock_redis

        pending, lag = await transport.pending_count()
        assert (pending, lag) == (12, 34)

        mock_redis.xinfo_groups = AsyncMock(
            return_value=[{b"name": b"cts-orchestrator", b"pending": 5, b"lag": None}]
        )
        assert await transport.pending_count() == (5, 0)
