"""Redis Streams subscriber for hardware-neutral ``TagFix`` messages on ``wearable.fixes``.

Decodes protobuf ``TagFix`` payloads strictly respecting presence flags, binds each
``tag_id`` to a household member via ``WearableSiteConfig.member_at(tag_id, measured_at)``,
drops ``OFF_BODY`` and stale fixes with counters, and dead-letters malformed messages
(XACK + warning + dead-letter counter, never silent fallback or infinite pending).
"""

from __future__ import annotations

import asyncio
import math
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from cts_contracts import FixKind, MotionState, WearState
from google.protobuf.message import DecodeError
from structlog import get_logger

from app.domain import WearableFix, WearableSiteConfig
from app.observability import metrics as _metrics
from app.observability.metrics import (
    DeadLetterStream,
    WearableFixIngestResult,
    WearableFixKindLabel,
)
from app.proto.continuoustracking.v1 import wearable_pb2
from app.services.wearable_fix_buffer import WearableFixBuffer
from app.services.wearable_site_config_map import WearableSiteConfigMap

logger = get_logger(__name__)

FIELD_WEARABLE_FIX = b"fix"

_PROTO_TO_FIX_KIND: dict[int, tuple[FixKind, WearableFixKindLabel]] = {
    wearable_pb2.FIX_KIND_POSITION: (FixKind.POSITION, WearableFixKindLabel.POSITION),
    wearable_pb2.FIX_KIND_ROOM: (FixKind.ROOM, WearableFixKindLabel.ROOM),
    wearable_pb2.FIX_KIND_PROXIMITY: (FixKind.PROXIMITY, WearableFixKindLabel.PROXIMITY),
}

_PROTO_TO_WEAR_STATE: dict[int, WearState] = {
    wearable_pb2.WEAR_STATE_UNSPECIFIED: WearState.UNSPECIFIED,
    wearable_pb2.WEAR_STATE_ON_BODY: WearState.ON_BODY,
    wearable_pb2.WEAR_STATE_OFF_BODY: WearState.OFF_BODY,
}

_PROTO_TO_MOTION_STATE: dict[int, MotionState] = {
    wearable_pb2.MOTION_STATE_UNSPECIFIED: MotionState.UNSPECIFIED,
    wearable_pb2.MOTION_STATE_MOVING: MotionState.MOVING,
    wearable_pb2.MOTION_STATE_STILL: MotionState.STILL,
}


def decode_tag_fix(
    raw_bytes: bytes,
    *,
    site_config: WearableSiteConfig,
    now: datetime,
    max_fix_age_s: float,
) -> tuple[WearableFix | None, WearableFixKindLabel, WearableFixIngestResult]:
    """Decode and validate one serialized ``TagFix`` protobuf payload.

    Returns ``(fix_or_none, kind_label, ingest_result)``. Presence flags on the
    protobuf wire message are strictly enforced: a scalar without its presence
    flag set is treated as ``None``.
    """
    try:
        msg = wearable_pb2.TagFix()
        msg.ParseFromString(raw_bytes)
    except (DecodeError, TypeError, ValueError):
        return None, WearableFixKindLabel.UNSPECIFIED, WearableFixIngestResult.DROPPED_MALFORMED

    kind_entry = _PROTO_TO_FIX_KIND.get(msg.kind)
    if kind_entry is None:
        return None, WearableFixKindLabel.UNSPECIFIED, WearableFixIngestResult.DROPPED_MALFORMED
    fix_kind, kind_label = kind_entry

    if not msg.tag_id or msg.measured_at_unix_ns == 0:
        return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED

    wear_state = _PROTO_TO_WEAR_STATE.get(msg.wear_state)
    motion_state = _PROTO_TO_MOTION_STATE.get(msg.motion_state)
    if wear_state is None or motion_state is None:
        return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED

    # Enforce presence flags and validate optional numeric domains.
    x_m: float | None = None
    y_m: float | None = None
    cov_m2: tuple[float, float, float, float] | None = None
    if msg.has_position:
        xx = float(msg.cov_xx_m2)
        xy = float(msg.cov_xy_m2)
        yy = float(msg.cov_yy_m2)
        if not (math.isfinite(xx) and math.isfinite(xy) and math.isfinite(yy)):
            return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED
        det = xx * yy - xy * xy
        if xx <= 0.0 or yy <= 0.0 or det <= 0.0:
            return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED
        x_m = msg.x_mm / 1000.0
        y_m = msg.y_mm / 1000.0
        cov_m2 = (xx, xy, xy, yy)
    elif fix_kind == FixKind.POSITION:
        return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED

    room_id = msg.room_id or None
    room_probability: float | None = None
    if msg.has_room_probability:
        rp = float(msg.room_probability)
        if not math.isfinite(rp) or not 0.0 <= rp <= 1.0:
            return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED
        room_probability = rp
    if fix_kind == FixKind.ROOM and room_id is None:
        return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED

    anchor_id = msg.anchor_id or None
    distance_m: float | None = None
    if msg.has_distance:
        dist = float(msg.distance_m)
        if not math.isfinite(dist) or dist < 0.0:
            return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED
        distance_m = dist

    quality: float | None = None
    if msg.has_quality:
        q = float(msg.quality)
        if not math.isfinite(q) or not 0.0 <= q <= 1.0:
            return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED
        quality = q

    residual_rms_m: float | None = None
    if msg.has_residual:
        res = float(msg.residual_rms_m)
        if not math.isfinite(res) or res < 0.0:
            return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED
        residual_rms_m = res

    try:
        measured_at = datetime.fromtimestamp(msg.measured_at_unix_ns / 1e9, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None, kind_label, WearableFixIngestResult.DROPPED_MALFORMED

    # Explicit off-body tags never enter association or the buffer.
    if wear_state == WearState.OFF_BODY:
        return None, kind_label, WearableFixIngestResult.DROPPED_OFF_BODY

    # Drop fixes older than the configured staleness horizon.
    age_s = (now - measured_at).total_seconds()
    if age_s > max_fix_age_s:
        return None, kind_label, WearableFixIngestResult.DROPPED_STALE

    member_id = site_config.member_at(msg.tag_id, measured_at)
    if member_id is None:
        return None, kind_label, WearableFixIngestResult.DROPPED_UNBOUND

    fix = WearableFix(
        tag_id=msg.tag_id,
        member_id=member_id,
        measured_at=measured_at,
        kind=fix_kind,
        x_m=x_m,
        y_m=y_m,
        cov_m2=cov_m2,
        floor_plan_id=msg.floor_plan_id,
        room_id=room_id,
        room_probability=room_probability,
        anchor_id=anchor_id,
        distance_m=distance_m,
        wear_state=wear_state,
        motion_state=motion_state,
        quality=quality,
        anchors_used=int(msg.anchors_used),
        residual_rms_m=residual_rms_m,
        adapter_id=msg.adapter_id,
        epoch_seq=int(msg.epoch_seq),
        physical_device_id=site_config.physical_device_of(msg.tag_id),
        received_at=now,
    )
    return fix, kind_label, WearableFixIngestResult.ACCEPTED


class WearableFixSubscriber:
    """Background consumer of the ``wearable.fixes`` Redis stream."""

    def __init__(
        self,
        *,
        redis_client: Any,
        buffer: WearableFixBuffer,
        site_config_map: WearableSiteConfigMap,
        stream: str = "wearable.fixes",
        group: str = "cts-orchestrator",
        consumer: str = "orchestrator-1",
        max_fix_age_s: float = 15.0,
    ) -> None:
        self._redis = redis_client
        self._buffer = buffer
        self._site_config_map = site_config_map
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._max_fix_age_s = max_fix_age_s
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start consuming ``wearable.fixes`` in the background."""
        self._running = True
        self._task = asyncio.create_task(self._consume_loop(), name="wearable-fix-subscriber")
        logger.info("wearable_fix_subscriber_started", stream=self._stream, group=self._group)

    async def stop(self) -> None:
        """Stop background consumption cleanly."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("wearable_fix_subscriber_stopped", stream=self._stream)

    async def _consume_loop(self) -> None:
        with suppress(Exception):
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)

        while self._running:
            try:
                streams = await self._redis.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams={self._stream: ">"},
                    count=32,
                    block=2000,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("wearable_fix_stream_read_failed", stream=self._stream)
                await asyncio.sleep(1.0)
                continue

            for _stream_name, messages in streams or []:
                for message_id, fields in messages or []:
                    try:
                        self.handle_fields(fields, now=datetime.now(UTC))
                    except Exception:
                        _metrics.metrics.cts_messages_dead_lettered_total.labels(
                            stream=str(DeadLetterStream.WEARABLE_FIXES)
                        ).inc()
                        logger.warning(
                            "wearable_fix_message_dead_lettered",
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
    ) -> WearableFixIngestResult:
        """Decode one Redis stream entry, update metrics, and insert into the buffer."""
        ts = now or datetime.now(UTC)
        raw_bytes: bytes | None = None
        if FIELD_WEARABLE_FIX in fields:
            raw_bytes = fields[FIELD_WEARABLE_FIX]  # type: ignore[index]
        elif "fix" in fields:
            raw_bytes = fields["fix"]  # type: ignore[index]

        if not raw_bytes:
            _metrics.metrics.cts_wearable_fixes_total.labels(
                kind=str(WearableFixKindLabel.UNSPECIFIED),
                result=str(WearableFixIngestResult.DROPPED_MALFORMED),
            ).inc()
            _metrics.metrics.cts_messages_dead_lettered_total.labels(
                stream=str(DeadLetterStream.WEARABLE_FIXES)
            ).inc()
            logger.warning("wearable_fix_missing_payload", stream=self._stream)
            return WearableFixIngestResult.DROPPED_MALFORMED

        site_config = self._site_config_map.get_sync()
        fix, kind_label, result = decode_tag_fix(
            raw_bytes,
            site_config=site_config,
            now=ts,
            max_fix_age_s=self._max_fix_age_s,
        )
        _metrics.metrics.cts_wearable_fixes_total.labels(
            kind=str(kind_label),
            result=str(result),
        ).inc()

        if result == WearableFixIngestResult.DROPPED_MALFORMED:
            _metrics.metrics.cts_messages_dead_lettered_total.labels(
                stream=str(DeadLetterStream.WEARABLE_FIXES)
            ).inc()
            logger.warning("wearable_fix_malformed", stream=self._stream, kind=str(kind_label))
            return result

        if fix is not None:
            self._buffer.add(fix)
            logger.debug(
                "wearable_fix_buffered",
                tag_id=fix.tag_id,
                kind=str(kind_label),
                epoch_seq=fix.epoch_seq,
            )
        return result
