"""RevisionPublisher: emits IdentityRevision proto messages to Redis Streams.

Publishes identity revisions to the ``tracking.revisions`` stream so
Cognitive Companion's ``IdentityRevisionSubscriber`` can rewrite
``PersonLocationHistory`` rows under the new identity.

Wire format: each Redis Streams message is a single field ``revision``
carrying the raw protobuf body of a
``continuoustracking.v1.IdentityRevision``.
"""

from __future__ import annotations

import json

from structlog import get_logger

from ..domain import IdentityRevision
from ..observability import metrics
from ..proto.continuoustracking.v1 import tracking_pb2
from .base_publisher import BasePublisher

logger = get_logger(__name__)

FIELD = b"revision"


class RevisionPublisher(BasePublisher):
    """Publishes IdentityRevision proto messages to Redis Streams."""

    _stream_name = "tracking.revisions"
    _default_maxlen = 50000

    async def publish(self, revision: IdentityRevision) -> str:
        """Publish a single IdentityRevision."""
        message = _to_proto(revision)
        message_id = await self._xadd({FIELD: message.SerializeToString()})

        metrics.metrics.tracking_revisions_published_total.labels(
            reason=revision.reason or "unknown"
        ).inc()

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
        """Publish multiple revisions in a single Redis pipeline."""
        if not revisions:
            return []
        if self._redis is None:
            logger.error("Cannot publish revisions: not connected")
            return []

        pipe = self._redis.pipeline(transaction=False)
        for revision in revisions:
            message = _to_proto(revision)
            pipe.xadd(
                self._stream,
                {FIELD: message.SerializeToString()},
                maxlen=self._maxlen,
                approximate=True,
            )

        results = await pipe.execute()
        for revision in revisions:
            metrics.metrics.tracking_revisions_published_total.labels(
                reason=revision.reason or "unknown"
            ).inc()
        logger.info(
            "Published batch of revisions",
            count=len(revisions),
        )
        return [r.decode("ascii") if isinstance(r, bytes) else str(r) for r in results]


def _to_proto(revision: IdentityRevision) -> tracking_pb2.IdentityRevision:
    """Convert a domain IdentityRevision to its proto wire form."""
    pb = tracking_pb2.IdentityRevision(
        revision_id=revision.revision_id,
        global_track_id=revision.global_track_id,
        tracklet_ids=list(revision.tracklet_ids),
        map_identity_id=revision.map_identity_id,
        posterior_entropy=revision.posterior_entropy,
        previous_identity_id=revision.previous_identity_id or "",
        new_identity_id=revision.new_identity_id or "",
        reason=revision.reason,
        evidence_json=json.dumps(revision.evidence, default=str),
        revision_time_unix_ns=int(revision.revision_time.timestamp() * 1e9),
    )
    for candidate in revision.candidates:
        pb.candidates.add(
            identity_id=candidate.identity_id,
            display_name=getattr(candidate, "display_name", "") or "",
            probability=float(candidate.probability),
        )
    return pb
