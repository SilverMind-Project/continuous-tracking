"""Publishes DwellEvent to tracking.dwell."""

from __future__ import annotations

from structlog import get_logger

from ..observability import metrics
from ..proto.continuoustracking.v1 import tracking_pb2
from .base_publisher import BasePublisher
from .codec import encode

logger = get_logger(__name__)

FIELD = "dwell"


class DwellPublisher(BasePublisher):
    """Publishes dwell state-change events to the tracking.dwell stream.

    One event per threshold crossing (started / ended); never per frame.
    """

    _stream_name = "tracking.dwell"
    _default_maxlen = 50000

    async def publish_started(
        self,
        ph_id: str,
        identity_id: str | None,
        room_name: str,
        event_time_unix_ns: int,
    ) -> str | None:
        """Publish a dwell-started event.

        Returns the Redis message ID, or None if not connected.
        """
        return await self._publish(
            ph_id=ph_id,
            identity_id=identity_id,
            event_type=tracking_pb2.DWELL_EVENT_TYPE_STARTED,
            room_name=room_name,
            event_time_unix_ns=event_time_unix_ns,
            duration_s=0,
        )

    async def publish_ended(
        self,
        ph_id: str,
        identity_id: str | None,
        room_name: str,
        event_time_unix_ns: int,
        duration_s: int,
    ) -> str | None:
        """Publish a dwell-ended event.

        Returns the Redis message ID, or None if not connected.
        """
        return await self._publish(
            ph_id=ph_id,
            identity_id=identity_id,
            event_type=tracking_pb2.DWELL_EVENT_TYPE_ENDED,
            room_name=room_name,
            event_time_unix_ns=event_time_unix_ns,
            duration_s=duration_s,
        )

    async def _publish(
        self,
        *,
        ph_id: str,
        identity_id: str | None,
        event_type: int,
        room_name: str,
        event_time_unix_ns: int,
        duration_s: int = 0,
    ) -> str | None:
        if not self.is_connected:
            logger.warning("dwell_publisher_not_connected")
            return None

        msg = tracking_pb2.DwellEvent(
            ph_id=ph_id,
            identity_id=identity_id or "",
            event_type=event_type,  # type: ignore[arg-type]
            room_name=room_name,
            event_time_unix_ns=event_time_unix_ns,
            duration_s=duration_s,
        )
        try:
            msg_id = await self._xadd(encode(msg, field=FIELD))  # type: ignore[arg-type]
            event_type_str = (
                "started" if event_type == tracking_pb2.DWELL_EVENT_TYPE_STARTED else "ended"
            )
            metrics.metrics.cts_dwell_events_published_total.labels(event_type=event_type_str).inc()
            logger.debug(
                "dwell_event_published",
                ph_id=ph_id,
                identity_id=identity_id,
                event_type=event_type_str,
                room_name=room_name,
            )
            return msg_id
        except Exception:
            logger.exception("dwell_publish_failed", ph_id=ph_id)
            return None
