"""Publishes PHContinuationCandidate events to tracking.continuations."""

from __future__ import annotations

from structlog import get_logger

from ..domain import PHContinuationCandidate
from .base_publisher import BasePublisher

logger = get_logger(__name__)


class PHContinuationPublisher(BasePublisher):
    """Publishes PH continuation candidates consumed by Cognitive Companion."""

    _stream_name = "tracking.continuations"
    _default_maxlen = 50000

    async def publish(self, candidate: PHContinuationCandidate) -> str:
        """Publish a single PH continuation candidate."""
        payload: dict[bytes, bytes] = {
            b"predecessor_ph_id": candidate.predecessor_ph_id.encode(),
            b"successor_ph_id": candidate.successor_ph_id.encode(),
            b"predecessor_closed_at": candidate.predecessor_closed_at.isoformat().encode(),
            b"successor_born_at": candidate.successor_born_at.isoformat().encode(),
            b"distance_m": str(candidate.distance_m).encode(),
            b"seconds_elapsed": str(candidate.seconds_elapsed).encode(),
            b"predicted_drift_m": str(candidate.predicted_drift_m).encode(),
            b"predecessor_identity_id": (candidate.predecessor_identity_id or "").encode(),
        }
        message_id = await self._xadd(payload)
        logger.debug(
            "ph_continuation_published",
            predecessor_ph_id=candidate.predecessor_ph_id,
            successor_ph_id=candidate.successor_ph_id,
            predecessor_identity_id=candidate.predecessor_identity_id,
            message_id=message_id,
        )
        return message_id
