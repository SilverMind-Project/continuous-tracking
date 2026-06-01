"""DwellPublisher unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.transport.dwell_publisher import DwellPublisher


@pytest.mark.asyncio
async def test_publish_started_builds_correct_proto():
    """publish_started constructs a DwellEvent proto with started type."""
    pub = DwellPublisher(redis_url="redis://localhost:6379/0")
    pub._redis = MagicMock()
    pub._redis.xadd = AsyncMock(return_value=b"test-msg-id")

    msg_id = await pub.publish_started(
        ph_id="ph-1",
        identity_id="alice",
        room_name="kitchen",
        event_time_unix_ns=1700000000000000000,
    )

    assert msg_id == "test-msg-id"
    pub._redis.xadd.assert_called_once()
    payload_dict = pub._redis.xadd.call_args[0][1]
    assert "dwell" in payload_dict


@pytest.mark.asyncio
async def test_publish_ended_builds_correct_proto():
    """publish_ended constructs a DwellEvent proto with duration_s."""
    pub = DwellPublisher(redis_url="redis://localhost:6379/0")
    pub._redis = MagicMock()
    pub._redis.xadd = AsyncMock(return_value=b"test-msg-id")

    msg_id = await pub.publish_ended(
        ph_id="ph-2",
        identity_id="bob",
        room_name="bathroom",
        event_time_unix_ns=1700000000000000000,
        duration_s=45,
    )

    assert msg_id == "test-msg-id"
    pub._redis.xadd.assert_called_once()


@pytest.mark.asyncio
async def test_publish_returns_none_when_not_connected():
    """When not connected, publish returns None."""
    pub = DwellPublisher(redis_url="redis://localhost:6379/0")

    msg_id = await pub.publish_started(
        ph_id="ph-1",
        identity_id="alice",
        room_name="kitchen",
        event_time_unix_ns=1700000000000000000,
    )

    assert msg_id is None


@pytest.mark.asyncio
async def test_publish_handles_redis_error():
    """Redis errors are caught and logged, returning None."""
    pub = DwellPublisher(redis_url="redis://localhost:6379/0")
    pub._redis = MagicMock()
    pub._redis.xadd = AsyncMock(side_effect=ConnectionError("redis down"))

    msg_id = await pub.publish_started(
        ph_id="ph-1",
        identity_id="alice",
        room_name="kitchen",
        event_time_unix_ns=1700000000000000000,
    )

    assert msg_id is None
