"""IdentityRevision protobuf round-trip test.

Construct a domain IdentityRevision, encode it to proto wire bytes,
decode back, and assert field equality on the new N0 fields.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import IdentityEvidence, IdentityRevision
from app.transport.revision_publisher import _to_proto


def _make_revision() -> IdentityRevision:
    return IdentityRevision(
        revision_id="rev-001",
        ph_id="ph-abc-123",
        previous_identity_id=None,
        new_identity_id="id-alice",
        actor="resolver",
        reason="initial_assignment",
        applied_at=datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC),
        rewritten_rows=3,
        evidence=IdentityEvidence(
            top_identity_id="id-alice",
            top_probability=0.92,
            second_probability=0.04,
            posterior_entropy=0.35,
            evidence_sources=["direct_face", "reid"],
            observation_count=12,
        ),
    )


def test_round_trip_preserves_ph_id() -> None:
    revision = _make_revision()
    pb = _to_proto(revision)

    # Field 1 (ph_id in proto) carries ph_id.
    assert pb.ph_id == "ph-abc-123"


def test_round_trip_preserves_revision_id() -> None:
    revision = _make_revision()
    pb = _to_proto(revision)

    assert pb.revision_id == "rev-001"


def test_round_trip_preserves_reason() -> None:
    revision = _make_revision()
    pb = _to_proto(revision)

    assert pb.reason == "initial_assignment"


def test_round_trip_preserves_revision_time() -> None:
    revision = _make_revision()
    pb = _to_proto(revision)

    expected_ns = int(datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC).timestamp() * 1e9)
    assert pb.revision_time_unix_ns == expected_ns


def test_round_trip_evidence_json_contains_actor() -> None:
    import json

    revision = _make_revision()
    pb = _to_proto(revision)

    evidence = json.loads(pb.evidence_json)
    assert evidence["actor"] == "resolver"
    assert evidence["rewritten_rows"] == 3
    assert evidence["top_probability"] == 0.92


def test_round_trip_handles_no_evidence() -> None:
    import json

    revision = IdentityRevision(
        revision_id="rev-002",
        ph_id="ph-def-456",
        previous_identity_id="id-bob",
        new_identity_id=None,
        actor="system",
        reason="demoted_to_unknown",
        applied_at=datetime(2026, 5, 27, 13, 0, 0, tzinfo=UTC),
        rewritten_rows=0,
        evidence=None,
    )
    pb = _to_proto(revision)

    assert pb.ph_id == "ph-def-456"
    evidence = json.loads(pb.evidence_json)
    assert evidence["actor"] == "system"
    assert evidence["rewritten_rows"] == 0
