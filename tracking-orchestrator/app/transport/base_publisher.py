"""Shared base class for Redis Streams publishers.

All CTS publishers (revisions, signals, scene samples) follow the same
lifecycle pattern: connect to Redis, XADD a single proto-bytes field,
disconnect.  This base class eliminates the duplicated connection
management; subclasses implement only the proto build + the public
``publish`` method.

Connections run with ``decode_responses=False`` so binary proto bodies
round-trip unchanged through ``XADD``.

Design note: publishers never create consumer groups -- ``XADD`` creates
the stream when ``mkstream`` is implied by Redis defaults.  Consumer
groups belong to the consuming side (see ``StreamConsumer._ensure_group``
on the CC backend).
"""

from __future__ import annotations

import redis.asyncio as redis
from structlog import get_logger

logger = get_logger(__name__)


class BasePublisher:
    """Abstract base for Redis Streams publishers.

    Subclasses inherit connection management and override ``_stream_name``
    and ``_default_maxlen`` as class attributes.

    Usage::

        class MyPublisher(BasePublisher):
            _stream_name = "my.stream"

            async def publish(self, msg: MyMessage) -> str:
                return await self._xadd({b"my_field": msg.SerializeToString()})

        pub = MyPublisher(redis_url="redis://localhost:6379/0")
        await pub.connect()
        await pub.publish(msg)
        await pub.disconnect()
    """

    _stream_name: str = ""
    _default_maxlen: int = 50000

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        stream: str | None = None,
        maxlen: int | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._stream = stream or self._stream_name
        self._maxlen = maxlen if maxlen is not None else self._default_maxlen
        self._redis: redis.Redis | None = None

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        """Connect to Redis.  Idempotent: a second call is a no-op."""
        if self._redis is not None:
            return

        self._redis = redis.from_url(
            self._redis_url,
            decode_responses=False,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        logger.info(
            "Publisher connected to Redis",
            stream=self._stream,
            url=self._redis_url,
        )

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("Publisher disconnected from Redis", stream=self._stream)

    async def _xadd(self, payload: dict[bytes, bytes]) -> str:
        """Append *payload* to the stream.  Returns the message ID or ``""``."""
        if self._redis is None:
            logger.error("Cannot publish: not connected to Redis", stream=self._stream)
            return ""

        message_id = await self._redis.xadd(
            self._stream,
            payload,  # type: ignore[arg-type]
            maxlen=self._maxlen,
            approximate=True,
        )
        return message_id.decode("ascii") if isinstance(message_id, bytes) else str(message_id)
