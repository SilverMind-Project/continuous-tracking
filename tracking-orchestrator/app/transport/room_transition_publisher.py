"""Publishes RoomTransitionEvent to tracking.room_transitions (M2)."""

from __future__ import annotations

from structlog import get_logger

from ..domain import RoomTransitionEvent
from .base_publisher import BasePublisher

logger = get_logger(__name__)


class RoomTransitionPublisher(BasePublisher):
    """Publishes room transition events to the tracking.room_transitions stream."""

    def __init__(self, redis_url: str, maxlen: int = 10000) -> None:
        super().__init__(redis_url=redis_url, stream="tracking.room_transitions", maxlen=maxlen)

    async def publish(
        self, event: RoomTransitionEvent, identity_id: str | None = None
    ) -> str | None:
        """Publish a single room transition event.

        Returns the Redis message ID, or None if the publisher is not connected.
        """
        if not self.is_connected:
            logger.warning("room_transition_publisher_not_connected")
            return None

        payload: dict[bytes, bytes] = {
            b"ph_id": event.ph_id.encode(),
            b"transit_zone_id": event.transit_zone_id.encode(),
            b"direction": event.direction.encode(),
            b"inside_room_id": event.inside_room_id.encode(),
            b"outside_room_id": event.outside_room_id.encode(),
            b"floor_x_m": str(event.floor_x_m).encode(),
            b"floor_y_m": str(event.floor_y_m).encode(),
            b"event_time": event.event_time.isoformat().encode(),
        }
        if identity_id:
            payload[b"identity_id"] = identity_id.encode()
        try:
            msg_id = await self._xadd(payload)
            logger.debug(
                "room_transition_published",
                ph_id=event.ph_id,
                direction=event.direction,
                transit_zone_id=event.transit_zone_id,
            )
            return msg_id
        except Exception:
            logger.exception("room_transition_publish_failed")
            return None
