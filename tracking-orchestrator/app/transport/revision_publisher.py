"""Revision publisher: emits IdentityRevision messages to Redis Streams.

Publishes identity revisions to the `tracking.revisions` stream so that
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

import redis.asyncio as redis
from structlog import get_logger

from ..domain import IdentityRevision

logger = get_logger(__name__)

# Default stream name for identity revisions.
DEFAULT_REVISIONS_STREAM = "tracking.revisions"
# Default max length for the stream (auto-trim).
DEFAULT_MAXLEN = 50000


class RevisionPublisher:
    """Publishes IdentityRevision messages to Redis Streams.

    Usage::

        publisher = RevisionPublisher(redis_url="redis://localhost:6379/0")
        await publisher.connect()

        revision = IdentityRevision(...)
        await publisher.publish(revision)

        await publisher.disconnect()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        stream: str = DEFAULT_REVISIONS_STREAM,
        maxlen: int = DEFAULT_MAXLEN,
    ) -> None:
        self._redis_url = redis_url
        self._stream = stream
        self._maxlen = maxlen
        self._redis: redis.Redis | None = None
        self._group_created = False

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        """Connect to Redis and create the consumer group if needed."""
        if self._redis is not None:
            return

        self._redis = redis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

        try:
            await self._redis.xgroup_create(
                self._stream,
                "cts-orchestrator",
                id="0",
                mkstream=True,
            )
            self._group_created = True
            logger.info("Created revision consumer group", stream=self._stream)
        except redis.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                self._group_created = True
                logger.info("Revision consumer group already exists", stream=self._stream)
            else:
                raise

        logger.info("Connected to Redis for revisions", url=self._redis_url)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis (revisions)")

    async def publish(self, revision: IdentityRevision) -> str:
        """Publish an IdentityRevision to the revisions stream.

        Args:
            revision: the identity revision to publish.

        Returns:
            The Redis message ID.
        """
        if self._redis is None:
            logger.error("Cannot publish revision: not connected")
            return ""

        payload: dict[str, str] = {
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

        message_id = str(
            await self._redis.xadd(
                self._stream,
                payload,  # type: ignore[arg-type]
                maxlen=self._maxlen,
                approximate=True,
            )
        )

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
        message_ids: list[str] = []

        for revision in revisions:
            payload: dict[str, str] = {
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
            message_ids.append(
                pipe.xadd(
                    self._stream,
                    payload,  # type: ignore[arg-type]
                    maxlen=self._maxlen,
                    approximate=True,
                )
            )

        results = await pipe.execute()
        logger.info(
            "Published batch of revisions",
            count=len(revisions),
            message_ids=results,
        )
        return [str(r) for r in results]
