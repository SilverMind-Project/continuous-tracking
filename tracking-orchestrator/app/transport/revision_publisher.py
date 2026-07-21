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

from ..domain import IdentityEvidence, IdentityRevision
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
            ph_id=revision.ph_id,
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
    """Convert a domain IdentityRevision to its proto wire form.

    Fields renamed from global_track_id/tracklet_ids to ph_id.
    The proto message still carries the legacy field numbers for
    wire compatibility; CC subscriber decodes accordingly.
    """
    evidence = revision.evidence or IdentityEvidence()
    pb = tracking_pb2.IdentityRevision(
        revision_id=revision.revision_id,
        ph_id=revision.ph_id,
        previous_identity_id=revision.previous_identity_id or "",
        new_identity_id=revision.new_identity_id or "",
        reason=revision.reason,
        revision_time_unix_ns=int(revision.applied_at.timestamp() * 1e9),
        evidence_json=json.dumps(
            {
                "top_identity_id": evidence.top_identity_id,
                "top_probability": evidence.top_probability,
                "second_probability": evidence.second_probability,
                "posterior_entropy": evidence.posterior_entropy,
                "evidence_sources": evidence.evidence_sources,
                "observation_count": evidence.observation_count,
                "actor": revision.actor,
                "rewritten_rows": revision.rewritten_rows,
            },
            default=str,
        ),
    )
    # -- typed revision-range / projection fields (18-25) --
    if revision.revision_kind:
        pb.revision_kind = revision.revision_kind
    if revision.range_start is not None:
        pb.range_start_unix_ns = int(revision.range_start.timestamp() * 1e9)
    if revision.range_end is not None:
        pb.range_end_unix_ns = int(revision.range_end.timestamp() * 1e9)
    if revision.range_authority:
        pb.range_authority = revision.range_authority
    if revision.revision_range_id:
        pb.revision_range_id = revision.revision_range_id
    if revision.correction_id:
        pb.correction_id = revision.correction_id
    if revision.required_projections:
        pb.required_projections.extend(revision.required_projections)
    if revision.revision_schema_version:
        pb.revision_schema_version = revision.revision_schema_version
    # tracklet_ids and map_identity_id left at proto defaults (empty / "")
    return pb
