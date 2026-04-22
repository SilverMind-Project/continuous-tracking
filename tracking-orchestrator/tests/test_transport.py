"""Unit tests for the Redis Streams transport layer."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain import Detection
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
    def test_valid_fields(self) -> None:
        transport = RedisStreamsTransport()
        fields = {
            "camera_id": "cam-1",
            "minio_key": "frames/cam-1/5.jpg",
            "frame_index": "5",
            "capture_time_unix_ns": "1000000000",
            "received_time_unix_ns": "1000000001",
            "width": "640",
            "height": "480",
            "sample_fps": "5.0",
        }
        frame = transport._deserialize_frame(fields)
        assert frame is not None
        assert frame.camera_id == "cam-1"
        assert frame.frame_index == 5
        assert frame.width == 640

    def test_missing_fields(self) -> None:
        transport = RedisStreamsTransport()
        fields: dict[str, str] = {}
        frame = transport._deserialize_frame(fields)
        assert frame is not None  # Should still work with defaults
        assert frame.camera_id == ""
        assert frame.frame_index == 0

    def test_invalid_frame_index(self) -> None:
        transport = RedisStreamsTransport()
        fields = {
            "camera_id": "cam-1",
            "minio_key": "frames/cam-1/x.jpg",
            "frame_index": "not_a_number",
            "capture_time_unix_ns": "1000000000",
            "received_time_unix_ns": "1000000001",
            "width": "640",
            "height": "480",
        }
        frame = transport._deserialize_frame(fields)
        assert frame is None  # Should return None on deserialization error


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
        transport, mock_redis = self._mock_transport()
        mock_redis.xreadgroup = AsyncMock(
            return_value=[
                (
                    b"frames.ready",
                    [
                        (
                            "1700000000000-0",
                            {
                                "camera_id": "cam-1",
                                "minio_key": "frames/cam-1/1.jpg",
                                "frame_index": "1",
                                "capture_time_unix_ns": "1000000000",
                                "received_time_unix_ns": "1000000001",
                                "width": "640",
                                "height": "480",
                            },
                        ),
                        (
                            "1700000000001-0",
                            {
                                "camera_id": "cam-1",
                                "minio_key": "frames/cam-1/2.jpg",
                                "frame_index": "2",
                                "capture_time_unix_ns": "1000000002",
                                "received_time_unix_ns": "1000000003",
                                "width": "640",
                                "height": "480",
                            },
                        ),
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
        transport._pending_acks[id(frame)] = "1700000000000-0"

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
