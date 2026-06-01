"""PresencePublisher unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.transport.presence_publisher import PresencePublisher


@pytest.mark.asyncio
async def test_publish_appeared_builds_correct_proto():
    """publish_appeared constructs a PresenceEvent proto with appeared type."""
    pub = PresencePublisher(redis_url="redis://localhost:6379/0")
    pub._redis = MagicMock()
    pub._redis.xadd = AsyncMock(return_value=b"test-msg-id")

    msg_id = await pub.publish_appeared(
        ph_id="ph-1",
        identity_id="alice",
        room_name="kitchen",
        event_time_unix_ns=1700000000000000000,
    )

    assert msg_id == "test-msg-id"
    pub._redis.xadd.assert_called_once()
    call_args = pub._redis.xadd.call_args
    # First positional arg is stream name, second is the payload dict.
    payload_dict = call_args[0][1]
    assert "presence" in payload_dict


@pytest.mark.asyncio
async def test_publish_disappeared_builds_correct_proto():
    """publish_disappeared constructs a PresenceEvent proto with disappeared type."""
    pub = PresencePublisher(redis_url="redis://localhost:6379/0")
    pub._redis = MagicMock()
    pub._redis.xadd = AsyncMock(return_value=b"test-msg-id")

    msg_id = await pub.publish_disappeared(
        ph_id="ph-2",
        identity_id=None,
        room_name="",
        event_time_unix_ns=1700000000000000000,
    )

    assert msg_id == "test-msg-id"
    pub._redis.xadd.assert_called_once()


@pytest.mark.asyncio
async def test_publish_returns_none_when_not_connected():
    """When not connected, publish returns None."""
    pub = PresencePublisher(redis_url="redis://localhost:6379/0")
    # _redis is None by default (not connected).

    msg_id = await pub.publish_appeared(
        ph_id="ph-1",
        identity_id="alice",
        room_name="kitchen",
        event_time_unix_ns=1700000000000000000,
    )

    assert msg_id is None


@pytest.mark.asyncio
async def test_publish_handles_redis_error():
    """Redis errors are caught and logged, returning None."""
    pub = PresencePublisher(redis_url="redis://localhost:6379/0")
    pub._redis = MagicMock()
    pub._redis.xadd = AsyncMock(side_effect=ConnectionError("redis down"))

    msg_id = await pub.publish_appeared(
        ph_id="ph-1",
        identity_id="alice",
        room_name="kitchen",
        event_time_unix_ns=1700000000000000000,
    )

    assert msg_id is None
