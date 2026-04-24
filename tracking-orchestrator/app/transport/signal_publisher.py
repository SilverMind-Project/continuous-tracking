"""SignalPublisher: publishes dementia signals to Redis Streams.

This module provides the transport layer for emitting ``DementiaSignal``
messages to the ``tracking.signals`` Redis Stream.  The
``DementiaSignalWorker`` calls ``publish_signal()`` after persisting
each signal to the repository.

The stream is consumed by Cognitive Companion's
``DementiaSignalSubscriber`` (which persists to the CC-side cache and
fires events into the rule engine).
"""

from __future__ import annotations

import json

import redis.asyncio as redis
from structlog import get_logger

from ..domain import DementiaSignal

logger = get_logger(__name__)


class SignalPublisher:
    """Publishes dementia signals to the ``tracking.signals`` Redis Stream.

    Usage::

        publisher = SignalPublisher(redis_url="redis://localhost:6379/0")
        await publisher.connect()
        await publisher.publish_signal(signal)
        await publisher.disconnect()
    """

    STREAM = "tracking.signals"

    def __init__(self, redis_url: str = "redis://localhost:6379/0", maxlen: int = 50000) -> None:
        self._redis_url = redis_url
        self._maxlen = maxlen
        self._redis: redis.Redis | None = None
        self._group_created = False

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        """Connect to Redis and create the consumer group if needed."""
        if self._redis is not None:
            return

        self._redis = redis.from_url(
            self._redis_url,
            decode_responses=False,  # We serialize JSON ourselves
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

        # Create consumer group (ignores BUSYGROUP error)
        try:
            await self._redis.xgroup_create(
                self.STREAM,
                "cognitive-companion-signals",
                id="$",
                mkstream=True,
            )
            self._group_created = True
            logger.info("Created signal consumer group", stream=self.STREAM)
        except redis.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                self._group_created = True
                logger.info("Signal consumer group already exists", stream=self.STREAM)
            else:
                raise

        logger.info("Connected to Redis for signal publishing", url=self._redis_url)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis for signal publishing")

    async def publish_signal(self, signal: DementiaSignal) -> str:
        """Publish a dementia signal to the Redis Stream.

        Args:
            signal: the computed dementia signal.

        Returns:
            The Redis message ID of the published signal.
        """
        if self._redis is None:
            logger.error("Cannot publish signal: not connected to Redis")
            return ""

        payload = self._serialize(signal)
        message_id = await self._redis.xadd(
            self.STREAM,
            {"signal": json.dumps(payload).encode("utf-8")},
            maxlen=self._maxlen,
            approximate=True,
        )

        logger.info(
            "Published dementia signal",
            signal_id=signal.signal_id,
            signal_kind=signal.signal_kind,
            identity_id=signal.identity_id,
            severity=signal.severity,
            message_id=message_id,
        )
        return str(message_id)

    async def publish_batch(self, signals: list[DementiaSignal]) -> list[str]:
        """Publish multiple signals in a single pipeline.

        Args:
            signals: list of dementia signals.

        Returns:
            List of Redis message IDs.
        """
        if self._redis is None:
            logger.error("Cannot publish batch: not connected to Redis")
            return []

        pipe = self._redis.pipeline(transaction=False)
        for signal in signals:
            payload = self._serialize(signal)
            pipe.xadd(
                self.STREAM,
                {"signal": json.dumps(payload).encode("utf-8")},
                maxlen=self._maxlen,
                approximate=True,
            )

        message_ids = await pipe.execute()
        logger.info(
            "Published batch of dementia signals",
            count=len(signals),
        )
        return [str(mid) for mid in message_ids]

    def _serialize(self, signal: DementiaSignal) -> dict[str, object]:
        """Convert a DementiaSignal to a JSON-serialisable dict."""
        return {
            "signal_id": signal.signal_id,
            "identity_id": signal.identity_id,
            "signal_kind": signal.signal_kind,
            "severity": signal.severity,
            "value": signal.value,
            "baseline": signal.baseline,
            "z_score": signal.z_score,
            "window_start": signal.window_start.isoformat(),
            "window_end": signal.window_end.isoformat(),
            "context": signal.context,
            "emitted_at": signal.emitted_at.isoformat(),
        }
