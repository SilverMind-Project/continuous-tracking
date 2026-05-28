"""WTR2: CCIdentityAssertionSubscriber tests.

Tests the assertion cache, field decoding, expiry, and malformed message
handling without requiring Redis testcontainers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.cc_identity_assertion_subscriber import (
    ASSERTION_TTL_S,
    CCIdentityAssertionSubscriber,
    IdentityAssertionCache,
)

# ---------------------------------------------------------------------------
# IdentityAssertionCache tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_expires_old_assertions():
    """Assertions older than TTL are pruned from get_recent()."""
    cache = IdentityAssertionCache()
    now = datetime.now(UTC)

    old = {
        "person_id": "alice",
        "confidence": 0.9,
        "camera_id": "cam-1",
        "captured_at": now - timedelta(seconds=60),
        "floor_x_m": 1.0,
        "floor_y_m": 2.0,
        "_received_at": now - timedelta(seconds=60),
    }
    await cache.add(old)

    recent = {
        "person_id": "bob",
        "confidence": 0.85,
        "camera_id": "cam-1",
        "captured_at": now,
        "floor_x_m": 1.5,
        "floor_y_m": 2.5,
        "_received_at": now,
    }
    await cache.add(recent)

    result = await cache.get_recent(max_age_s=ASSERTION_TTL_S)
    assert len(result) == 1
    assert result[0]["person_id"] == "bob"


@pytest.mark.asyncio
async def test_cache_stores_floor_coordinates():
    """Assertions with floor coordinates preserve them for spatial matching."""
    cache = IdentityAssertionCache()
    now = datetime.now(UTC)

    assertion = {
        "person_id": "carol",
        "confidence": 0.8,
        "camera_id": "cam-2",
        "captured_at": now,
        "floor_x_m": 3.5,
        "floor_y_m": 4.2,
        "_received_at": now,
    }
    await cache.add(assertion)

    result = await cache.get_recent()
    assert len(result) == 1
    assert result[0]["floor_x_m"] == 3.5
    assert result[0]["floor_y_m"] == 4.2


@pytest.mark.asyncio
async def test_cache_handles_empty():
    """get_recent() on an empty cache returns empty list."""
    cache = IdentityAssertionCache()
    result = await cache.get_recent()
    assert result == []


# ---------------------------------------------------------------------------
# Subscriber field decoding (unit tests on _handle)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscriber_decodes_all_required_fields():
    """The _handle method decodes person_id, confidence, camera_id, captured_at,
    floor_x_m, and floor_y_m from Redis byte fields."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )

    now = datetime.now(UTC)
    fields: dict[bytes, bytes] = {
        b"person_id": b"dave",
        b"confidence": b"0.92",
        b"camera_id": b"cam-3",
        b"captured_at": now.isoformat().encode(),
        b"floor_x_m": b"5.0",
        b"floor_y_m": b"6.0",
    }

    await subscriber._handle(b"msg-1", fields)
    result = await cache.get_recent()
    assert len(result) == 1
    a = result[0]
    assert a["person_id"] == "dave"
    assert a["confidence"] == 0.92
    assert a["camera_id"] == "cam-3"
    assert a["floor_x_m"] == 5.0
    assert a["floor_y_m"] == 6.0


@pytest.mark.asyncio
async def test_subscriber_drops_empty_person_id():
    """A message with no person_id must not be cached."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )

    now = datetime.now(UTC)
    fields: dict[bytes, bytes] = {
        b"person_id": b"",
        b"confidence": b"0.5",
        b"camera_id": b"cam-1",
        b"captured_at": now.isoformat().encode(),
    }

    await subscriber._handle(b"msg-2", fields)
    result = await cache.get_recent()
    assert len(result) == 0


@pytest.mark.asyncio
async def test_subscriber_handles_malformed_captured_at():
    """A message with unparseable captured_at defaults to now without crashing."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )

    fields: dict[bytes, bytes] = {
        b"person_id": b"eve",
        b"confidence": b"0.7",
        b"camera_id": b"cam-1",
        b"captured_at": b"not-a-datetime",
        b"floor_x_m": b"0",
        b"floor_y_m": b"0",
    }

    # Must not raise.
    await subscriber._handle(b"msg-3", fields)
    result = await cache.get_recent()
    assert len(result) == 1
    assert result[0]["person_id"] == "eve"


@pytest.mark.asyncio
async def test_subscriber_handles_malformed_floor_coordinates():
    """Unparseable floor coordinates default to 0.0 without crashing."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )

    now = datetime.now(UTC)
    fields: dict[bytes, bytes] = {
        b"person_id": b"frank",
        b"confidence": b"0.7",
        b"camera_id": b"cam-1",
        b"captured_at": now.isoformat().encode(),
        b"floor_x_m": b"not-a-float",
        b"floor_y_m": b"also-not-float",
    }

    await subscriber._handle(b"msg-4", fields)
    result = await cache.get_recent()
    assert len(result) == 1
    assert result[0]["floor_x_m"] == 0.0
    assert result[0]["floor_y_m"] == 0.0
