"""Revision publisher: emits IdentityRevision messages to Redis Streams.

Publishes identity revisions to the ``tracking.revisions`` stream so that
downstream consumers (Cognitive Companion's identity_rewriter, UI, etc.)
can react to retroactive identity changes.

Each revision is serialized as a JSON payload with the following fields:
- revision_id: unique revision identifier
- global_track_id: the GlobalTrack being revised
- tracklet_ids: list of tracklet IDs affected
- previous_identity_id: prior identity (or null)
- new_identity_id: new identity (or null for UNKNOWN)
- map_identity_id: the MAP (max a posteriori) identity
- posterior_entropy: entropy of the posterior distribution
- reason: human-readable reason for the revision
- evidence: structured evidence (top_probability, margin, etc.)
- revision_time: ISO 8601 timestamp

This module is dependency-free: it operates on domain types and uses
the Redis transport for publishing.
"""

from __future__ import annotations

import json

from structlog import get_logger

from ..domain import IdentityRevision
from .base_publisher import BasePublisher

logger = get_logger(__name__)


class RevisionPublisher(BasePublisher):
    """Publishes IdentityRevision messages to Redis Streams.

    Usage::

        publisher = RevisionPublisher(redis_url="redis://localhost:6379/0")
        await publisher.connect()

        revision = IdentityRevision(...)
        await publisher.publish(revision)

        await publisher.disconnect()
    """

    _stream_name = "tracking.revisions"
    _default_maxlen = 50000

    async def publish(self, revision: IdentityRevision) -> str:
        """Publish an IdentityRevision to the revisions stream.

        Args:
            revision: the identity revision to publish.

        Returns:
            The Redis message ID.
        """
        payload = _serialize(revision)
        message_id = await self._xadd(payload)

        logger.info(
            "Published identity revision",
            revision_id=revision.revision_id,
            global_track_id=revision.global_track_id,
            previous_identity_id=revision.previous_identity_id,
            new_identity_id=revision.new_identity_id,
            message_id=message_id,
        )

        return message_id

    async def publish_many(self, revisions: list[IdentityRevision]) -> list[str]:
        """Publish multiple revisions in a single pipeline.

        Args:
            revisions: list of identity revisions to publish.

        Returns:
            List of Redis message IDs.
        """
        if not revisions:
            return []

        if self._redis is None:
            logger.error("Cannot publish revisions: not connected")
            return []

        pipe = self._redis.pipeline(transaction=False)

        for revision in revisions:
            payload = _serialize(revision)
            pipe.xadd(
                self._stream,
                payload,  # type: ignore[arg-type]
                maxlen=self._maxlen,
                approximate=True,
            )

        results = await pipe.execute()
        logger.info(
            "Published batch of revisions",
            count=len(revisions),
            message_ids=results,
        )
        return [str(r) for r in results]


def _serialize(revision: IdentityRevision) -> dict[str, str]:
    """Convert an IdentityRevision to a flat Redis-fields dict."""
    return {
        "revision_id": revision.revision_id,
        "global_track_id": revision.global_track_id,
        "tracklet_ids": json.dumps(revision.tracklet_ids),
        "previous_identity_id": revision.previous_identity_id or "",
        "new_identity_id": revision.new_identity_id or "",
        "map_identity_id": revision.map_identity_id,
        "posterior_entropy": str(revision.posterior_entropy),
        "reason": revision.reason,
        "evidence": json.dumps(revision.evidence),
        "revision_time": revision.revision_time.isoformat(),
    }
