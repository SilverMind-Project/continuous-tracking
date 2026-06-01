"""Publishes PresenceEvent to tracking.presence."""

from __future__ import annotations

from structlog import get_logger

from ..observability import metrics
from ..proto.continuoustracking.v1 import tracking_pb2
from .base_publisher import BasePublisher
from .codec import encode

logger = get_logger(__name__)

FIELD = "presence"


class PresencePublisher(BasePublisher):
    """Publishes presence state-change events to the tracking.presence stream.

    One event per state transition (appeared / disappeared); never per frame.
    """

    _stream_name = "tracking.presence"
    _default_maxlen = 50000

    async def publish_appeared(
        self,
        ph_id: str,
        identity_id: str | None,
        room_name: str,
        event_time_unix_ns: int,
    ) -> str | None:
        """Publish an appeared event.

        Returns the Redis message ID, or None if not connected.
        """
        return await self._publish(
            ph_id=ph_id,
            identity_id=identity_id,
            event_type=tracking_pb2.PRESENCE_EVENT_TYPE_APPEARED,
            room_name=room_name,
            event_time_unix_ns=event_time_unix_ns,
        )

    async def publish_disappeared(
        self,
        ph_id: str,
        identity_id: str | None,
        room_name: str,
        event_time_unix_ns: int,
    ) -> str | None:
        """Publish a disappeared event.

        Returns the Redis message ID, or None if not connected.
        """
        return await self._publish(
            ph_id=ph_id,
            identity_id=identity_id,
            event_type=tracking_pb2.PRESENCE_EVENT_TYPE_DISAPPEARED,
            room_name=room_name,
            event_time_unix_ns=event_time_unix_ns,
        )

    async def _publish(
        self,
        *,
        ph_id: str,
        identity_id: str | None,
        event_type: int,
        room_name: str,
        event_time_unix_ns: int,
    ) -> str | None:
        if not self.is_connected:
            logger.warning("presence_publisher_not_connected")
            return None

        msg = tracking_pb2.PresenceEvent(
            ph_id=ph_id,
            identity_id=identity_id or "",
            event_type=event_type,  # type: ignore[arg-type]
            room_name=room_name,
            event_time_unix_ns=event_time_unix_ns,
        )
        try:
            msg_id = await self._xadd(encode(msg, field=FIELD))  # type: ignore[arg-type]
            event_type_str = (
                "appeared"
                if event_type == tracking_pb2.PRESENCE_EVENT_TYPE_APPEARED
                else "disappeared"
            )
            metrics.metrics.cts_presence_events_published_total.labels(
                event_type=event_type_str
            ).inc()
            logger.debug(
                "presence_event_published",
                ph_id=ph_id,
                identity_id=identity_id,
                event_type=event_type_str,
                room_name=room_name,
            )
            return msg_id
        except Exception:
            logger.exception("presence_publish_failed", ph_id=ph_id)
            return None
