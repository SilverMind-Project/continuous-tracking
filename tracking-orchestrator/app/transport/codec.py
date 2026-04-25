"""Wire codec for CTS Redis Streams.

Every CTS Redis Stream carries exactly one protobuf message type per
stream. Each message is stored as a single Redis-Streams field whose
value is the raw ``Message.SerializeToString()`` output. Stream
consumers know what type to expect, so no codec discriminator is
needed -- the entire envelope is::

    {b"<field>": <raw protobuf bytes>}

Field names mirror the message they carry so ``XRANGE`` output is
self-explanatory:

* ``frames.ready`` -> field ``"frame"`` carrying ``FrameReady``
* ``tracking.events`` -> field ``"event"`` carrying ``TrackingEvent``
* ``tracking.revisions`` -> field ``"revision"`` carrying ``IdentityRevision``
* ``tracking.signals`` -> field ``"signal"`` carrying ``DementiaSignal``
* ``scene.samples`` -> field ``"sample"`` carrying ``SceneSample``

Publishers should set ``decode_responses=False`` on the Redis client so
binary payloads round-trip unchanged.
"""

from __future__ import annotations

from typing import Any

from google.protobuf.message import Message


def encode(message: Message, *, field: str) -> dict[str, bytes]:
    """Serialise *message* into a Redis-Streams field dict."""
    return {field: message.SerializeToString()}


def decode[T: Message](fields: dict[Any, Any], message_type: type[T], *, field: str) -> T:
    """Parse a Redis-Streams field dict into *message_type*.

    Accepts either ``str`` or ``bytes`` keys (Redis returns whichever
    based on ``decode_responses``).
    """
    payload: Any = fields.get(field) if field in fields else fields.get(field.encode())
    if payload is None:
        raise ValueError(f"missing '{field}' field in stream message")
    if isinstance(payload, str):
        payload = payload.encode("latin-1")
    instance: T = message_type()
    instance.ParseFromString(payload)
    return instance


__all__ = ["decode", "encode"]
