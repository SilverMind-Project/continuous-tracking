"""Tests for RevisionPublisher."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain import (
    IdentityCandidate,
    IdentityRevision,
)
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


class TestRevisionPublisher:
    """Test the RevisionPublisher Redis Streams publisher."""

    @pytest.fixture
    def publisher(self) -> RevisionPublisher:
        return RevisionPublisher(redis_url="redis://localhost:6379/0")

    def test_create_publisher(self, publisher: RevisionPublisher) -> None:
        assert publisher._redis is None
        assert publisher._stream == "tracking.revisions"

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, publisher: RevisionPublisher) -> None:
        """Test connect and disconnect with mocked Redis."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.xgroup_create = AsyncMock()
        mock_redis_module = MagicMock()
        mock_redis_module.from_url = MagicMock(return_value=mock_redis_instance)
        mock_redis_module.ResponseError = Exception
        with patch(
            "app.transport.revision_publisher.redis",
            mock_redis_module,
        ):
            await publisher.connect()
            assert publisher._redis is not None
            await publisher.disconnect()
            assert publisher._redis is None

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
