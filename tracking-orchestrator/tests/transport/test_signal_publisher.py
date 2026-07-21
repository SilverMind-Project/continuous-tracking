"""SignalPublisher wire-vocabulary coverage tests.

These guard the producer/consumer wire contract: the proto mapping must cover every
shared ``cts_contracts`` signal kind/severity (so a new wire kind cannot be added
without a publisher mapping and silently published as UNSPECIFIED), and the domain
vocabulary must remain a superset of the wire vocabulary.
"""

from __future__ import annotations

from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from cts_contracts import DementiaSignalKind, DementiaSignalSeverity

from app.domain import DementiaSignal
from app.domain import DementiaSignalKind as DomainSignalKind
from app.proto.continuoustracking.v1 import signals_pb2
from app.transport.signal_publisher import (
    _KIND_TO_PROTO,
    _SEVERITY_TO_PROTO,
    SignalPublisher,
    _to_proto,
)


def test_proto_maps_never_resolve_to_unspecified() -> None:
    """The maps are derived (`{m: getattr(pb, f"..._{m.name}")}`), so coverage is
    guaranteed by construction; the assertion with teeth is that no member resolved
    to UNSPECIFIED (which is what CC drops). A member whose name has no proto
    constant raises AttributeError at import instead -- also loud."""
    assert signals_pb2.DEMENTIA_SIGNAL_KIND_UNSPECIFIED not in _KIND_TO_PROTO.values()
    assert signals_pb2.DEMENTIA_SIGNAL_SEVERITY_UNSPECIFIED not in _SEVERITY_TO_PROTO.values()


def test_shared_enum_matches_proto_enum_member_set() -> None:
    """The real drift guard: the shared `cts_contracts` vocabulary and the proto wire
    enum must define exactly the same kinds/severities (excluding UNSPECIFIED). If the
    proto gains a kind the shared enum lacks (or vice versa), this fails -- catching the
    divergence as a clean test failure rather than waiting for an import-time crash."""
    # ``.keys`` is the protobuf EnumTypeWrapper API (proto enum names), not a dict.
    proto_kind_names = list(signals_pb2.DementiaSignalKind.keys())
    proto_kinds = {
        n.removeprefix("DEMENTIA_SIGNAL_KIND_")
        for n in proto_kind_names
        if n != "DEMENTIA_SIGNAL_KIND_UNSPECIFIED"
    }
    assert {m.name for m in DementiaSignalKind} == proto_kinds

    proto_severity_names = list(signals_pb2.DementiaSignalSeverity.keys())
    proto_severities = {
        n.removeprefix("DEMENTIA_SIGNAL_SEVERITY_")
        for n in proto_severity_names
        if n != "DEMENTIA_SIGNAL_SEVERITY_UNSPECIFIED"
    }
    assert {m.name for m in DementiaSignalSeverity} == proto_severities


def test_domain_vocabulary_is_superset_of_wire_vocabulary() -> None:
    """The domain Literal may carry producer-internal (non-wire) kinds, but it must
    contain every wire kind so an emittable wire kind is always a valid domain kind."""
    domain_kinds = set(get_args(DomainSignalKind))
    assert {k.value for k in DementiaSignalKind} <= domain_kinds


@pytest.mark.asyncio
async def test_previously_unmapped_kind_now_maps_to_real_proto_enum() -> None:
    """Regression: a gait_slowing signal must serialize to the real proto kind,
    not UNSPECIFIED (which CC drops)."""
    pub = SignalPublisher(redis_url="redis://localhost:6379/0")
    pub._redis = MagicMock()
    pub._redis.xadd = AsyncMock(return_value=b"msg-1")

    signal = DementiaSignal(
        signal_id="sig-1",
        identity_id="alice",
        signal_kind="gait_slowing",
        severity="warning",
        value=1.0,
    )
    proto = _to_proto(signal)
    assert proto.kind == signals_pb2.DEMENTIA_SIGNAL_KIND_GAIT_SLOWING
    assert proto.severity == signals_pb2.DEMENTIA_SIGNAL_SEVERITY_WARNING
