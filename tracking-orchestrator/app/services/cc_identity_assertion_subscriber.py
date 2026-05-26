"""Subscribes to ``cc.identity_assertions`` Redis stream (M4 bidirectional flow).

Cognitive-companion publishes face-anchor-equivalent assertions from the
recamera VLM path.  This subscriber consumes them and caches the most
recent assertions for the orchestrator's FaceIdentityStage to inject as
evidence into the Bayesian identity resolver.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from structlog import get_logger

logger = get_logger(__name__)

# Stream constants.
STREAM = "cc.identity_assertions"
GROUP = "tracking-orchestrator"
CONSUMER = "orchestrator-1"

# How long to keep cached assertions before expiry.
ASSERTION_TTL_S = 30.0


class IdentityAssertionCache:
    """Thread-safe cache of recent CC identity assertions.

    The FaceIdentityStage reads this cache to find assertions that
    match a PH within the spatial/temporal window.
    """

    def __init__(self) -> None:
        self._assertions: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add(self, assertion: dict[str, Any]) -> None:
        async with self._lock:
            self._assertions.append(assertion)

    async def get_recent(self, max_age_s: float = ASSERTION_TTL_S) -> list[dict[str, Any]]:
        """Return assertions not older than *max_age_s*."""
        now = datetime.now(UTC)
        async with self._lock:
            # Prune expired.
            self._assertions = [
                a
                for a in self._assertions
                if (now - a["_received_at"]).total_seconds() <= max_age_s
            ]
            return list(self._assertions)


class CCIdentityAssertionSubscriber:
    """Background consumer of the cc.identity_assertions Redis stream."""

    def __init__(
        self,
        redis_client: Any,  # redis[hiredis] async client
        cache: IdentityAssertionCache,
    ) -> None:
        self._redis = redis_client
        self._cache = cache
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Begin consuming assertions in the background."""
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())
        logger.info("cc_identity_assertion_subscriber_started", stream=STREAM)

    async def stop(self) -> None:
        """Stop consuming."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        logger.info("cc_identity_assertion_subscriber_stopped")

    async def _consume_loop(self) -> None:
        """Poll cc.identity_assertions stream in a loop."""
        # Create consumer group if not exists.
        with suppress(Exception):
            await self._redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)

        while self._running:
            try:
                streams = await self._redis.xreadgroup(
                    groupname=GROUP,
                    consumername=CONSUMER,
                    streams={STREAM: ">"},
                    count=10,
                    block=5000,
                )
            except Exception:
                logger.exception("cc_identity_assertion_read_failed")
                await asyncio.sleep(1)
                continue

            for _stream_name, messages in streams or []:
                for message_id, fields in messages or []:
                    try:
                        await self._handle(message_id, fields)
                        await self._redis.xack(STREAM, GROUP, message_id)
                    except Exception:
                        logger.exception(
                            "cc_identity_assertion_handle_failed",
                            message_id=message_id,
                        )

    async def _handle(self, message_id: bytes, fields: dict[bytes, bytes]) -> None:
        """Parse a single assertion and add to the cache."""
        person_id = fields.get(b"person_id", b"").decode()
        if not person_id:
            return

        confidence = float(fields.get(b"confidence", b"0.7").decode())
        camera_id = fields.get(b"camera_id", b"").decode()

        captured_at_str = fields.get(b"captured_at", b"").decode()
        try:
            captured_at = datetime.fromisoformat(captured_at_str)
        except (ValueError, TypeError):
            captured_at = datetime.now(UTC)

        assertion = {
            "person_id": person_id,
            "confidence": confidence,
            "camera_id": camera_id,
            "captured_at": captured_at,
            "_received_at": datetime.now(UTC),
        }
        await self._cache.add(assertion)
        logger.debug(
            "cc_identity_assertion_received",
            person_id=person_id,
            confidence=round(confidence, 3),
            camera_id=camera_id,
        )
