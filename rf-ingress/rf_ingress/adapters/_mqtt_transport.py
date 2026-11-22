"""Shared injectable MQTT message stream transport for rf-ingress MQTT adapters.

Keeps broker I/O behind a small protocol so unit and conformance tests run
deterministically without a live MQTT broker while production adapters use
``aiomqtt.Client``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol
from urllib.parse import urlparse

import aiomqtt


class MqttMessageStream(Protocol):
    """Injectable MQTT subscription stream so CI and conformance need no broker."""

    def messages(self) -> AsyncIterator[tuple[str, bytes]]: ...

    async def close(self) -> None: ...


class InMemoryMqttMessageStream:
    """Deterministic in-memory :class:`MqttMessageStream` for unit and conformance tests."""

    def __init__(self, items: Sequence[tuple[str, bytes]]) -> None:
        self._items: list[tuple[str, bytes]] = list(items)
        self._closed: bool = False

    async def messages(self) -> AsyncIterator[tuple[str, bytes]]:
        for topic, payload in self._items:
            if self._closed:
                break
            yield (topic, payload)

    async def close(self) -> None:
        self._closed = True


class AiomqttMessageStream:
    """Production MQTT message stream backed by ``aiomqtt.Client``."""

    def __init__(
        self,
        *,
        broker_url: str,
        topic: str,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        parsed = urlparse(broker_url)
        if parsed.scheme not in ("mqtt", "mqtts") or not parsed.hostname:
            raise ValueError(f"Invalid MQTT broker_url: {broker_url!r}")
        self._host: str = parsed.hostname
        self._port: int = parsed.port or (8883 if parsed.scheme == "mqtts" else 1883)
        self._topic: str = topic
        self._username: str | None = username
        self._password: str | None = password
        self._closed: bool = False

    async def messages(self) -> AsyncIterator[tuple[str, bytes]]:
        async with aiomqtt.Client(
            hostname=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
        ) as client:
            await client.subscribe(self._topic)
            async for message in client.messages:
                if self._closed:
                    break
                raw = message.payload
                if isinstance(raw, bytes):
                    payload_bytes = raw
                elif isinstance(raw, bytearray):
                    payload_bytes = bytes(raw)
                elif isinstance(raw, str):
                    payload_bytes = raw.encode("utf-8")
                else:
                    payload_bytes = b""
                yield (str(message.topic), payload_bytes)

    async def close(self) -> None:
        self._closed = True
