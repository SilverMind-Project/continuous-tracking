"""CCIdentityAssertionSubscriber tests.

Tests the typed assertion cache, field decoding, expiry, and malformed message
handling without requiring Redis testcontainers.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import ExternalIdentityAssertion
from app.proto.continuoustracking.v1.tracking_pb2 import CCIdentityAssertion
from app.services.cc_identity_assertion_subscriber import (
    ASSERTION_TTL_S,
    CCIdentityAssertionSubscriber,
    IdentityAssertionCache,
)

# ---------------------------------------------------------------------------
# IdentityAssertionCache tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_expires_old_assertions() -> None:
    """Assertions older than TTL are pruned from recent()."""
    cache = IdentityAssertionCache()
    now = datetime.now(UTC)

    old = ExternalIdentityAssertion(
        person_id="alice",
        calibrated_confidence=0.9,
        camera_id="cam-1",
        captured_at=now - timedelta(seconds=60),
        floor_x_m=1.0,
        floor_y_m=2.0,
        received_at=now - timedelta(seconds=60),
    )
    await cache.add(old)

    recent_item = ExternalIdentityAssertion(
        person_id="bob",
        calibrated_confidence=0.85,
        camera_id="cam-1",
        captured_at=now,
        floor_x_m=1.5,
        floor_y_m=2.5,
        received_at=now,
    )
    await cache.add(recent_item)

    result = await cache.recent(max_age_s=ASSERTION_TTL_S)
    assert len(result) == 1
    assert result[0].person_id == "bob"


@pytest.mark.asyncio
async def test_cache_stores_floor_coordinates() -> None:
    """Assertions with floor coordinates preserve them for spatial matching."""
    cache = IdentityAssertionCache()
    now = datetime.now(UTC)

    assertion = ExternalIdentityAssertion(
        person_id="carol",
        calibrated_confidence=0.8,
        camera_id="cam-2",
        captured_at=now,
        floor_x_m=3.5,
        floor_y_m=4.2,
        received_at=now,
    )
    await cache.add(assertion)

    result = await cache.recent()
    assert len(result) == 1
    assert result[0].floor_x_m == 3.5
    assert result[0].floor_y_m == 4.2


@pytest.mark.asyncio
async def test_cache_handles_empty() -> None:
    """recent() on an empty cache returns empty sequence."""
    cache = IdentityAssertionCache()
    result = await cache.recent()
    assert list(result) == []


# ---------------------------------------------------------------------------
# Subscriber field decoding (unit tests on _handle)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscriber_decodes_all_required_fields() -> None:
    """The _handle method decodes person_id, calibrated_confidence, camera_id, captured_at,
    floor_x_m, and floor_y_m from Redis byte fields when has_floor_point is set."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )

    now = datetime.now(UTC)
    msg = CCIdentityAssertion(
        person_id="dave",
        camera_id="cam-3",
        captured_at_unix_ns=int(now.timestamp() * 1e9),
        floor_x_m=5.0,
        floor_y_m=6.0,
        has_floor_point=True,
    )
    msg.calibrated_confidence = 0.92

    fields: dict[bytes, bytes] = {b"assertion": msg.SerializeToString()}

    await subscriber._handle(b"msg-1", fields)
    result = await cache.recent()
    assert len(result) == 1
    assertion = result[0]
    assert assertion.person_id == "dave"
    assert assertion.calibrated_confidence is not None
    assert math.isclose(assertion.calibrated_confidence, 0.92, abs_tol=1e-5)
    assert assertion.camera_id == "cam-3"
    assert assertion.floor_x_m == 5.0
    assert assertion.floor_y_m == 6.0


@pytest.mark.asyncio
async def test_subscriber_decode_round_trip_with_all_presence_flags() -> None:
    """Room, yaw, and quality decode when their has_* flags are set."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )
    now = datetime.now(UTC)
    msg = CCIdentityAssertion(
        person_id="grace",
        camera_id="recamera_kitchen",
        captured_at_unix_ns=int(now.timestamp() * 1e9),
        room_name="Kitchen",
        yaw_deg=25.0,
        has_yaw=True,
        quality=0.7,
        has_quality=True,
    )
    msg.calibrated_confidence = 0.88

    await subscriber._handle(b"msg-flags", {b"assertion": msg.SerializeToString()})
    result = await cache.recent()
    assert len(result) == 1
    a = result[0]
    assert a.room_name == "Kitchen"
    assert a.yaw_deg == 25.0
    assert a.quality is not None
    assert math.isclose(a.quality, 0.7, abs_tol=1e-5)
    assert a.floor_x_m is None
    assert a.floor_y_m is None


@pytest.mark.asyncio
async def test_subscriber_decode_round_trip_without_presence_flags() -> None:
    """Absent has_* flags decode to None, never a fabricated value."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )
    now = datetime.now(UTC)
    msg = CCIdentityAssertion(
        person_id="henry",
        camera_id="cam-1",
        captured_at_unix_ns=int(now.timestamp() * 1e9),
    )
    msg.calibrated_confidence = 0.85

    await subscriber._handle(b"msg-no-flags", {b"assertion": msg.SerializeToString()})
    result = await cache.recent()
    assert len(result) == 1
    a = result[0]
    assert a.room_name is None
    assert a.yaw_deg is None
    assert a.quality is None
    assert a.floor_x_m is None
    assert a.floor_y_m is None


@pytest.mark.asyncio
async def test_subscriber_no_calibration_cached_with_none_confidence() -> None:
    """No calibrated_confidence on the wire caches calibrated_confidence=None."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )
    now = datetime.now(UTC)
    msg = CCIdentityAssertion(
        person_id="ivy",
        camera_id="cam-1",
        captured_at_unix_ns=int(now.timestamp() * 1e9),
        raw_similarity=0.9,
    )

    await subscriber._handle(b"msg-uncalibrated", {b"assertion": msg.SerializeToString()})
    result = await cache.recent()
    assert len(result) == 1
    assert result[0].calibrated_confidence is None


@pytest.mark.asyncio
async def test_subscriber_drops_empty_person_id() -> None:
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
    result = await cache.recent()
    assert len(result) == 0


@pytest.mark.asyncio
async def test_subscriber_handles_malformed_captured_at() -> None:
    """A message with missing captured_at defaults to now without crashing."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )

    msg = CCIdentityAssertion(
        person_id="eve",
        camera_id="cam-1",
        floor_x_m=0.0,
        floor_y_m=0.0,
    )

    fields: dict[bytes, bytes] = {b"assertion": msg.SerializeToString()}

    await subscriber._handle(b"msg-3", fields)
    result = await cache.recent()
    assert len(result) == 1
    assert result[0].captured_at is not None


@pytest.mark.asyncio
async def test_subscriber_handles_malformed_floor_coordinates() -> None:
    """Missing floor coordinates decode to None, never fabricated (0, 0)."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )

    now = datetime.now(UTC)
    msg = CCIdentityAssertion(
        person_id="frank",
        camera_id="cam-1",
        captured_at_unix_ns=int(now.timestamp() * 1e9),
    )

    fields: dict[bytes, bytes] = {b"assertion": msg.SerializeToString()}

    await subscriber._handle(b"msg-4", fields)
    result = await cache.recent()
    assert len(result) == 1
    assert result[0].floor_x_m is None
    assert result[0].floor_y_m is None


@pytest.mark.asyncio
async def test_subscriber_zero_zero_without_flag_is_not_a_position() -> None:
    """A proto message with floor_x_m/floor_y_m literally 0.0 and has_floor_point unset
    must decode to None, not the real position (0, 0)."""
    cache = IdentityAssertionCache()
    subscriber = CCIdentityAssertionSubscriber(
        redis_client=object(),  # type: ignore[arg-type]
        cache=cache,
    )
    now = datetime.now(UTC)
    msg = CCIdentityAssertion(
        person_id="jack",
        camera_id="cam-1",
        captured_at_unix_ns=int(now.timestamp() * 1e9),
        floor_x_m=0.0,
        floor_y_m=0.0,
    )

    await subscriber._handle(b"msg-zero-zero", {b"assertion": msg.SerializeToString()})
    result = await cache.recent()
    assert len(result) == 1
    assert result[0].floor_x_m is None
    assert result[0].floor_y_m is None
