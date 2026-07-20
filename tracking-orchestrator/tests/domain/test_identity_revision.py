"""IdentityRevision domain type regression tests.

Asserts the post-N0 shape: no tracklet_ids, ph_id present,
frozen immutability.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from app.domain import IdentityEvidence, IdentityRevision


def test_identity_revision_has_no_tracklet_ids() -> None:
    """tracklet_ids must not exist on the N0 IdentityRevision."""
    with pytest.raises(TypeError):
        IdentityRevision(
            revision_id="r",
            ph_id="p",
            previous_identity_id=None,
            new_identity_id=None,
            actor="system",
            reason="test",
            applied_at=datetime.now(UTC),
            rewritten_rows=0,
            tracklet_ids=[],  # type: ignore[call-arg]
        )


def test_identity_revision_has_no_global_track_id() -> None:
    """global_track_id must not exist on the N0 IdentityRevision."""
    with pytest.raises(TypeError):
        IdentityRevision(
            revision_id="r",
            previous_identity_id=None,
            new_identity_id=None,
            actor="system",
            reason="test",
            applied_at=datetime.now(UTC),
            rewritten_rows=0,
            global_track_id="gt-1",  # type: ignore[call-arg]
        )


def test_identity_revision_has_ph_id() -> None:
    rev = IdentityRevision(
        revision_id="rev-1",
        ph_id="ph-abc",
        previous_identity_id=None,
        new_identity_id="id-alice",
        actor="resolver",
        reason="initial_assignment",
        applied_at=datetime.now(UTC),
        rewritten_rows=0,
    )
    assert rev.ph_id == "ph-abc"


def test_identity_revision_has_actor() -> None:
    rev = IdentityRevision(
        revision_id="rev-1",
        ph_id="ph-abc",
        previous_identity_id=None,
        new_identity_id=None,
        actor="user:op-1",
        reason="manual_override",
        applied_at=datetime.now(UTC),
        rewritten_rows=1,
    )
    assert rev.actor == "user:op-1"


def test_identity_revision_has_rewritten_rows() -> None:
    rev = IdentityRevision(
        revision_id="rev-1",
        ph_id="ph-abc",
        previous_identity_id=None,
        new_identity_id="id-alice",
        actor="resolver",
        reason="initial_assignment",
        applied_at=datetime.now(UTC),
        rewritten_rows=5,
    )
    assert rev.rewritten_rows == 5


def test_identity_revision_is_frozen() -> None:
    rev = IdentityRevision(
        revision_id="rev-1",
        ph_id="ph-abc",
        previous_identity_id=None,
        new_identity_id=None,
        actor="system",
        reason="test",
        applied_at=datetime.now(UTC),
        rewritten_rows=0,
    )
    with pytest.raises(FrozenInstanceError):
        rev.ph_id = "mutated"  # type: ignore[misc]


def test_identity_evidence_defaults() -> None:
    evidence = IdentityEvidence()
    assert evidence.top_identity_id is None
    assert evidence.top_probability == 0.0
    assert evidence.evidence_sources == []


def test_dementia_signal_kind_matches_contracts() -> None:
    from typing import get_args

    import cts_contracts

    from app.domain import DementiaSignalKind  # type: ignore[attr-defined]

    assert set(get_args(DementiaSignalKind)) == {str(k) for k in cts_contracts.DementiaSignalKind}
