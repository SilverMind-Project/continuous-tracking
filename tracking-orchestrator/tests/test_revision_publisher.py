"""Tests for RevisionPublisher and BasePublisher."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain import (
    IdentityCandidate,
    IdentityRevision,
)
from app.transport.base_publisher import BasePublisher
from app.transport.revision_publisher import RevisionPublisher


def _make_revision(
    global_track_id: str = "gt-1",
    identity_id: str = "alice",
) -> IdentityRevision:
    return IdentityRevision(
        revision_id="rev-1",
        global_track_id=global_track_id,
        tracklet_ids=["t1"],
        candidates=[IdentityCandidate(identity_id, "Alice", 0.9)],
        map_identity_id=identity_id,
        posterior_entropy=0.3,
        revision_time=datetime.now(UTC),
    )


class TestBasePublisher:
    """Test the shared BasePublisher lifecycle."""

    @pytest.fixture
    def publisher(self) -> BasePublisher:
        pub = BasePublisher(redis_url="redis://localhost:6379/0", stream="test.stream")
        return pub

    def test_not_connected_by_default(self, publisher: BasePublisher) -> None:
        assert not publisher.is_connected
        assert publisher._redis is None

    @pytest.mark.asyncio
    async def test_connect_creates_redis(self, publisher: BasePublisher) -> None:
        mock_redis_instance = AsyncMock()
        mock_redis_module = MagicMock()
        mock_redis_module.from_url = MagicMock(return_value=mock_redis_instance)
        with patch("app.transport.base_publisher.redis", mock_redis_module):
            await publisher.connect()
            assert publisher.is_connected
            mock_redis_module.from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, publisher: BasePublisher) -> None:
        mock_redis_instance = AsyncMock()
        mock_redis_module = MagicMock()
        mock_redis_module.from_url = MagicMock(return_value=mock_redis_instance)
        with patch("app.transport.base_publisher.redis", mock_redis_module):
            await publisher.connect()
            await publisher.connect()  # second call is no-op
            assert mock_redis_module.from_url.call_count == 1

    @pytest.mark.asyncio
    async def test_disconnect(self, publisher: BasePublisher) -> None:
        mock_redis_instance = AsyncMock()
        publisher._redis = mock_redis_instance
        await publisher.disconnect()
        assert not publisher.is_connected
        mock_redis_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, publisher: BasePublisher) -> None:
        await publisher.disconnect()  # should not raise
        assert not publisher.is_connected

    @pytest.mark.asyncio
    async def test_connect_does_not_create_consumer_group(self, publisher: BasePublisher) -> None:
        """Producers must never create consumer groups (TD-003)."""
        mock_redis_instance = AsyncMock()
        mock_redis_module = MagicMock()
        mock_redis_module.from_url = MagicMock(return_value=mock_redis_instance)
        with patch("app.transport.base_publisher.redis", mock_redis_module):
            await publisher.connect()
            mock_redis_instance.xgroup_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_xadd_not_connected(self, publisher: BasePublisher) -> None:
        result = await publisher._xadd({"key": "value"})
        assert result == ""

    @pytest.mark.asyncio
    async def test_xadd_success(self, publisher: BasePublisher) -> None:
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value="1234567890-0")
        publisher._redis = mock_redis
        result = await publisher._xadd({"key": "value"})
        assert result == "1234567890-0"
        mock_redis.xadd.assert_awaited_once()


class TestRevisionPublisher:
    """Test the RevisionPublisher Redis Streams publisher."""

    @pytest.fixture
    def publisher(self) -> RevisionPublisher:
        return RevisionPublisher(redis_url="redis://localhost:6379/0")

    def test_create_publisher(self, publisher: RevisionPublisher) -> None:
        assert publisher._redis is None
        assert publisher._stream == "tracking.revisions"

    def test_inherits_base_publisher(self, publisher: RevisionPublisher) -> None:
        assert isinstance(publisher, BasePublisher)

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, publisher: RevisionPublisher) -> None:
        """Test connect and disconnect with mocked Redis."""
        mock_redis_instance = AsyncMock()
        mock_redis_module = MagicMock()
        mock_redis_module.from_url = MagicMock(return_value=mock_redis_instance)
        with patch(
            "app.transport.base_publisher.redis",
            mock_redis_module,
        ):
            await publisher.connect()
            assert publisher._redis is not None
            # Verify no consumer group is created (TD-003)
            mock_redis_instance.xgroup_create.assert_not_called()
            await publisher.disconnect()
            assert publisher._redis is None

    @pytest.mark.asyncio
    async def test_publish_not_connected(self, publisher: RevisionPublisher) -> None:
        result = await publisher.publish(_make_revision())
        assert result == ""

    @pytest.mark.asyncio
    async def test_publish_many_empty(self, publisher: RevisionPublisher) -> None:
        result = await publisher.publish_many([])
        assert result == []

    @pytest.mark.asyncio
    async def test_publish_many_not_connected(self, publisher: RevisionPublisher) -> None:
        result = await publisher.publish_many([_make_revision()])
        assert result == []

    @pytest.mark.asyncio
    async def test_publish_many_success(self, publisher: RevisionPublisher) -> None:
        """publish_many should call Redis pipeline and return message IDs."""
        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock()
        mock_pipe = AsyncMock()
        mock_pipe.xadd = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=["msg-1", "msg-2"])
        mock_redis.pipeline.return_value = mock_pipe
        publisher._redis = mock_redis

        revisions = [
            _make_revision("gt-1", "alice"),
            _make_revision("gt-2", "bob"),
        ]
        result = await publisher.publish_many(revisions)
        assert len(result) == 2
        assert result[0] == "msg-1"
        assert result[1] == "msg-2"
        assert mock_pipe.xadd.call_count == 2

    @pytest.mark.asyncio
    async def test_publish_success(self, publisher: RevisionPublisher) -> None:
        """publish should call _xadd with serialized payload."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value="msg-42")
        publisher._redis = mock_redis

        revision = _make_revision()
        result = await publisher.publish(revision)
        assert result == "msg-42"
        mock_redis.xadd.assert_awaited_once()
