"""Redis Streams subscriber for ``cc.household_presence`` protobuf messages.

Cognitive Companion publishes household-wide presence snapshots and roster
completeness on ``cc.household_presence`` (field ``"presence"``). This module
decodes those messages, holds the latest snapshot in memory behind
:class:`HouseholdPresenceSource`, and dead-letters malformed entries (XACK +
warning + counter, never silent fallback).
"""

from __future__ import annotations

import asyncio
import math
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from cts_contracts.household import PresenceStatus
from google.protobuf.message import DecodeError
from structlog import get_logger

from app.domain import (
    HouseholdPresenceSnapshot,
    MemberPresenceSnapshot,
)
from app.observability import metrics as _metrics
from app.observability.metrics import DeadLetterStream
from app.proto.continuoustracking.v1 import household_pb2

logger = get_logger(__name__)

STREAM = "cc.household_presence"
FIELD_PRESENCE = b"presence"
GROUP = "tracking-orchestrator"
CONSUMER = "orchestrator-1"

_PROTO_TO_PRESENCE_STATUS: dict[int, PresenceStatus] = {
    household_pb2.PRESENCE_STATUS_PRESENT_ROOM: PresenceStatus.PRESENT_ROOM,
    household_pb2.PRESENCE_STATUS_PRESENT_HOME: PresenceStatus.PRESENT_HOME,
    household_pb2.PRESENCE_STATUS_AWAY: PresenceStatus.AWAY,
    household_pb2.PRESENCE_STATUS_ASLEEP: PresenceStatus.ASLEEP,
    household_pb2.PRESENCE_STATUS_UNKNOWN: PresenceStatus.UNKNOWN,
    household_pb2.PRESENCE_STATUS_STALE: PresenceStatus.STALE,
}


def decode_household_presence(
    raw_bytes: bytes,
    *,
    received_at: datetime,
) -> HouseholdPresenceSnapshot | None:
    """Decode and validate one serialized ``HouseholdPresence`` protobuf payload."""
    try:
        msg = household_pb2.HouseholdPresence()
        msg.ParseFromString(raw_bytes)
    except (DecodeError, TypeError, ValueError):
        return None

    if msg.published_at_unix_ns == 0:
        return None
    try:
        published_at = datetime.fromtimestamp(msg.published_at_unix_ns / 1e9, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None

    members: list[MemberPresenceSnapshot] = []
    for member in msg.members:
        if not member.member_id:
            return None
        status = _PROTO_TO_PRESENCE_STATUS.get(member.status)
        if status is None:
            return None
        conf = float(member.confidence)
        if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            return None
        if member.inferred_at_unix_ns == 0:
            inferred_at = published_at
        else:
            try:
                inferred_at = datetime.fromtimestamp(member.inferred_at_unix_ns / 1e9, tz=UTC)
            except (OverflowError, OSError, ValueError):
                return None
        members.append(
            MemberPresenceSnapshot(
                member_id=member.member_id,
                status=status,
                confidence=conf,
                room_id=member.room_id or None,
                inferred_at=inferred_at,
            )
        )

    return HouseholdPresenceSnapshot(
        members=tuple(members),
        roster_complete=bool(msg.roster_complete),
        visitors_present=bool(msg.visitors_present),
        published_at=published_at,
        received_at=received_at,
    )


class HouseholdPresenceCache:
    """In-memory latest-snapshot holder implementing :class:`HouseholdPresenceSource`."""

    def __init__(self) -> None:
        self._latest: HouseholdPresenceSnapshot | None = None

    def update(self, snapshot: HouseholdPresenceSnapshot) -> None:
        """Store *snapshot* if it is not older than the currently cached snapshot."""
        if self._latest is not None and snapshot.published_at < self._latest.published_at:
            return
        self._latest = snapshot

    def latest(self) -> HouseholdPresenceSnapshot | None:
        """Return the latest received household presence snapshot, or ``None``."""
        return self._latest

    def purge_identity(self, identity_id: str) -> int:
        """Drop *identity_id* from the cached member presence tuple if present."""
        snap = self._latest
        if snap is None:
            return 0
        kept = tuple(m for m in snap.members if m.member_id != identity_id)
        removed = len(snap.members) - len(kept)
        if removed > 0:
            self._latest = replace(snap, members=kept)
        return removed

    def clear(self) -> None:
        """Clear the cached snapshot."""
        self._latest = None


class HouseholdPresenceSubscriber:
    """Background consumer of the ``cc.household_presence`` Redis stream."""

    def __init__(
        self,
        *,
        redis_client: Any,
        cache: HouseholdPresenceCache,
        stream: str = STREAM,
        group: str = GROUP,
        consumer: str = CONSUMER,
    ) -> None:
        self._redis = redis_client
        self._cache = cache
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Begin consuming ``cc.household_presence`` in the background."""
        self._running = True
        self._task = asyncio.create_task(
            self._consume_loop(),
            name="household-presence-subscriber",
        )
        logger.info("household_presence_subscriber_started", stream=self._stream, group=self._group)

    async def stop(self) -> None:
        """Stop background consumption cleanly."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("household_presence_subscriber_stopped", stream=self._stream)

    async def _consume_loop(self) -> None:
        with suppress(Exception):
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)

        while self._running:
            try:
                streams = await self._redis.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams={self._stream: ">"},
                    count=16,
                    block=2000,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("household_presence_stream_read_failed", stream=self._stream)
                await asyncio.sleep(1.0)
                continue

            for _stream_name, messages in streams or []:
                for message_id, fields in messages or []:
                    try:
                        self.handle_fields(fields, now=datetime.now(UTC))
                    except Exception:
                        _metrics.metrics.cts_messages_dead_lettered_total.labels(
                            stream=str(DeadLetterStream.HOUSEHOLD_PRESENCE)
                        ).inc()
                        logger.warning(
                            "household_presence_message_dead_lettered",
                            stream=self._stream,
                            message_id=str(message_id),
                            exc_info=True,
                        )
                    finally:
                        with suppress(Exception):
                            await self._redis.xack(self._stream, self._group, message_id)

    def handle_fields(
        self,
        fields: dict[bytes, bytes] | dict[str, bytes],
        *,
        now: datetime | None = None,
    ) -> bool:
        """Decode one Redis stream entry and update the cache; return True on success."""
        ts = now or datetime.now(UTC)
        raw_bytes: bytes | None = None
        if FIELD_PRESENCE in fields:
            raw_bytes = fields[FIELD_PRESENCE]  # type: ignore[index]
        elif "presence" in fields:
            raw_bytes = fields["presence"]  # type: ignore[index]

        if not raw_bytes:
            _metrics.metrics.cts_messages_dead_lettered_total.labels(
                stream=str(DeadLetterStream.HOUSEHOLD_PRESENCE)
            ).inc()
            logger.warning("household_presence_missing_payload", stream=self._stream)
            return False

        snapshot = decode_household_presence(raw_bytes, received_at=ts)
        if snapshot is None:
            _metrics.metrics.cts_messages_dead_lettered_total.labels(
                stream=str(DeadLetterStream.HOUSEHOLD_PRESENCE)
            ).inc()
            logger.warning("household_presence_malformed", stream=self._stream)
            return False

        self._cache.update(snapshot)
        logger.debug(
            "household_presence_updated",
            member_count=len(snapshot.members),
            roster_complete=snapshot.roster_complete,
            visitors_present=snapshot.visitors_present,
        )
        return True
