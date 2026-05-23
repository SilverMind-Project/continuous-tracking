"""SignalPublisher: publishes DementiaSignal proto messages to Redis Streams.

Consumed by Cognitive Companion's :class:`DementiaSignalSubscriber`,
which persists the signal to the CC cache and fires a context-filter
event into the rule engine.

Wire format: each Redis Streams message is a single field ``signal``
carrying the raw protobuf body of a
``continuoustracking.v1.DementiaSignal``.
"""

from __future__ import annotations

import json

from structlog import get_logger

from ..domain import DementiaSignal
from ..observability import metrics
from ..proto.continuoustracking.v1 import signals_pb2
from .base_publisher import BasePublisher

logger = get_logger(__name__)

FIELD = b"signal"


# Domain Literal -> proto enum mappings.  The orchestrator's domain layer
# uses string Literals; the wire format uses proto enums for validation
# and forwards-compat.

_KIND_TO_PROTO: dict[str, int] = {
    "pacing": signals_pb2.DEMENTIA_SIGNAL_KIND_PACING,
    "sundowning_index": signals_pb2.DEMENTIA_SIGNAL_KIND_SUNDOWNING_INDEX,
    "bathroom_dwell_anomaly": signals_pb2.DEMENTIA_SIGNAL_KIND_BATHROOM_DWELL_ANOMALY,
    "nighttime_movement": signals_pb2.DEMENTIA_SIGNAL_KIND_NIGHTTIME_MOVEMENT,
    "stillness_anomaly": signals_pb2.DEMENTIA_SIGNAL_KIND_STILLNESS_ANOMALY,
    "absence": signals_pb2.DEMENTIA_SIGNAL_KIND_ABSENCE,
}

_SEVERITY_TO_PROTO: dict[str, int] = {
    "info": signals_pb2.DEMENTIA_SIGNAL_SEVERITY_INFO,
    "warning": signals_pb2.DEMENTIA_SIGNAL_SEVERITY_WARNING,
    "emergency": signals_pb2.DEMENTIA_SIGNAL_SEVERITY_EMERGENCY,
}


class SignalPublisher(BasePublisher):
    """Publishes DementiaSignal proto messages to ``tracking.signals``."""

    _stream_name = "tracking.signals"
    _default_maxlen = 50000

    async def publish_signal(self, signal: DementiaSignal) -> str:
        """Publish a single DementiaSignal."""
        message = _to_proto(signal)
        message_id = await self._xadd({FIELD: message.SerializeToString()})

        metrics.metrics.dementia_signals_published_total.labels(
            signal_kind=signal.signal_kind,
            severity=signal.severity,
        ).inc()

        logger.info(
            "Published dementia signal",
            signal_id=signal.signal_id,
            signal_kind=signal.signal_kind,
            identity_id=signal.identity_id,
            severity=signal.severity,
            message_id=message_id,
        )
        return message_id

    async def publish_batch(self, signals: list[DementiaSignal]) -> list[str]:
        """Publish multiple signals in a single Redis pipeline."""
        if not signals:
            return []
        if self._redis is None:
            logger.error("Cannot publish batch: not connected to Redis")
            return []

        pipe = self._redis.pipeline(transaction=False)
        for signal in signals:
            message = _to_proto(signal)
            pipe.xadd(
                self._stream,
                {FIELD: message.SerializeToString()},
                maxlen=self._maxlen,
                approximate=True,
            )

        message_ids = await pipe.execute()
        for signal in signals:
            metrics.metrics.dementia_signals_published_total.labels(
                signal_kind=signal.signal_kind,
                severity=signal.severity,
            ).inc()
        logger.info(
            "Published batch of dementia signals",
            count=len(signals),
        )
        return [mid.decode("ascii") if isinstance(mid, bytes) else str(mid) for mid in message_ids]


def _to_proto(signal: DementiaSignal) -> signals_pb2.DementiaSignal:
    """Convert a domain DementiaSignal to its proto wire form."""
    pb = signals_pb2.DementiaSignal()
    pb.signal_id = signal.signal_id
    pb.identity_id = signal.identity_id
    # Proto enum values are plain ints at runtime; the generated stubs
    # type the attribute as the enum class which mypy treats as
    # incompatible with ``int``. Bypass via setattr.
    setattr(  # noqa: B010
        pb,
        "kind",
        _KIND_TO_PROTO.get(signal.signal_kind, signals_pb2.DEMENTIA_SIGNAL_KIND_UNSPECIFIED),
    )
    setattr(  # noqa: B010
        pb,
        "severity",
        _SEVERITY_TO_PROTO.get(signal.severity, signals_pb2.DEMENTIA_SIGNAL_SEVERITY_UNSPECIFIED),
    )
    pb.value = float(signal.value)
    pb.has_baseline = signal.baseline is not None
    pb.baseline = float(signal.baseline) if signal.baseline is not None else 0.0
    pb.has_z_score = signal.z_score is not None
    pb.z_score = float(signal.z_score) if signal.z_score is not None else 0.0
    pb.window_start_unix_ns = int(signal.window_start.timestamp() * 1e9)
    pb.window_end_unix_ns = int(signal.window_end.timestamp() * 1e9)
    pb.emitted_at_unix_ns = int(signal.emitted_at.timestamp() * 1e9)
    pb.context_json = json.dumps(signal.context, default=str)
    pb.algorithm_version = signal.algorithm_version
    pb.algorithm_name = signal.algorithm_name or ""
    pb.evidence_grade = signal.evidence_grade or ""
    pb.algorithm_spec_json = signal.algorithm_spec_json or ""
    return pb
