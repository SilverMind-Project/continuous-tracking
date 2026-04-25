"""Stream message codec — future protobuf wire-format support.

This module is a forward-looking skeleton for the M10 migration from
JSON-encoded Redis Streams payloads to protobuf-encoded payloads (TD-004).

Current state (M9): all streams use JSON-flat encoding.
Target state (M10): all streams use protobuf with a dual-codec decoder
that accepts either format during the rollout window.

The planned migration strategy is:

1. Producers start emitting messages with a ``codec`` field set to
   ``"proto:continuoustracking.v1.<MessageType>"``.  The payload body
   is stored in a ``proto`` field as base64-encoded ``SerializeToString()``
   output.  Legacy fields remain present during the dual-write phase.
2. Consumers switch to :func:`decode_message`, which inspects ``codec``
   and dispatches to proto or JSON decoding accordingly.
3. Once all consumers have been updated, the legacy JSON fields are
   removed from the producer.

Codec field conventions
-----------------------
- ``codec="json"`` (or absent): current JSON-flat encoding.
- ``codec="proto:continuoustracking.v1.TrackingEvent"``: protobuf
  payload in the ``proto`` field (base64).

Usage (M10)::

    # Encoding (producer)
    payload = encode_message(tracking_event_pb2, codec=Codec.PROTO)

    # Decoding (consumer)
    msg = decode_message(raw_fields, registry={
        "continuoustracking.v1.TrackingEvent": TrackingEvent,
    })
"""

from __future__ import annotations

import enum
from typing import Any


class Codec(enum.Enum):
    """Wire codec for Redis Streams message payloads."""

    JSON = "json"
    PROTO = "proto"


def _field_get(fields: dict[Any, Any], key: str) -> Any:
    """Look up *key* in a Redis fields dict that may have str or bytes keys."""
    val = fields.get(key)
    if val is not None:
        return val
    return fields.get(key.encode())


def detect_codec(fields: dict[Any, Any]) -> Codec:
    """Detect the codec used by a Redis Streams message.

    Args:
        fields: raw Redis Streams field dict (bytes or str keys/values).

    Returns:
        The detected codec type.
    """
    codec_field = _field_get(fields, "codec")
    if codec_field is None:
        return Codec.JSON
    codec_str = (
        codec_field.decode("utf-8")
        if isinstance(codec_field, bytes | bytearray)
        else str(codec_field)
    )
    if codec_str.startswith("proto:"):
        return Codec.PROTO
    return Codec.JSON


def decode_message(
    fields: dict[Any, Any],
    *,
    json_decoder: Any = None,
    proto_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode a Redis Streams message using the appropriate codec.

    This is the dual-codec decoder entry point.  During the M10 rollout
    window, it accepts both JSON and protobuf messages.

    Args:
        fields: raw Redis Streams field dict.
        json_decoder: callable ``(fields) -> dict`` for JSON decoding.
            Falls back to identity if not provided.
        proto_registry: mapping of fully-qualified proto message names
            to their generated Python classes.  Required for proto
            decoding.

    Returns:
        Decoded message as a Python dict.

    Raises:
        ValueError: if proto codec is detected but no registry is
            provided or the message type is not found in the registry.
    """
    codec = detect_codec(fields)

    if codec == Codec.JSON:
        if json_decoder is not None:
            return json_decoder(fields)  # type: ignore[no-any-return]
        # Identity fallback: return string-decoded fields.
        return {_to_str(k): _to_str(v) for k, v in fields.items()}

    # Proto path — skeleton for M10.
    if proto_registry is None:
        msg = "Proto codec detected but no proto_registry provided"
        raise ValueError(msg)

    codec_field = _to_str(_field_get(fields, "codec") or "")
    # Extract message type from "proto:continuoustracking.v1.TrackingEvent"
    _, _, message_type = codec_field.partition("proto:")
    if message_type not in proto_registry:
        msg = f"Unknown proto message type: {message_type}"
        raise ValueError(msg)

    # Placeholder: actual proto deserialization goes here in M10.
    # proto_class = proto_registry[message_type]
    # raw_bytes = base64.b64decode(_field_get(fields, "proto") or b"")
    # msg_instance = proto_class()
    # msg_instance.ParseFromString(raw_bytes)
    # return MessageToDict(msg_instance)
    msg = f"Proto deserialization not yet implemented for {message_type}"
    raise NotImplementedError(msg)


def _to_str(value: Any) -> str:
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8", errors="replace")
    return str(value)
