"""M07 (F9) repository boundary validation: bounded authority vocabulary.

``InMemoryIdentityDecisionRepository.save`` must reject any ``authority`` value
that is not a member of ``IdentityAuthority`` -- the CHECK-constraint substitute
this program uses instead of a migration. The Postgres half of this parity proof
is ``tests/integration/test_identity_decision_authority_validation_postgres.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from app.domain import IdentityProvenanceDecision
from app.storage.base import InMemoryIdentityDecisionRepository

_T0 = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)


def _decision(*, authority: str) -> IdentityProvenanceDecision:
    return IdentityProvenanceDecision(
        decision_id=str(uuid.uuid4()),
        ph_id=str(uuid.uuid4()),
        captured_at=_T0,
        authority=authority,
        decision_source="face",
        diagnostics={},
        inferred_identity_id="amma",
        effective_identity_id="amma",
    )


@pytest.mark.asyncio
async def test_save_rejects_identity_id_as_authority() -> None:
    """The F9 defect, reproduced directly: an identity id is not an authority level."""
    repo = InMemoryIdentityDecisionRepository()
    with pytest.raises(ValueError, match="amma"):
        await repo.save(_decision(authority="amma"))


@pytest.mark.asyncio
async def test_save_rejects_empty_string_authority() -> None:
    repo = InMemoryIdentityDecisionRepository()
    with pytest.raises(ValueError):
        await repo.save(_decision(authority=""))


@pytest.mark.asyncio
async def test_save_rejects_legacy_decision_source_value_as_authority() -> None:
    """``arcface_authority`` is a decision_source value, not an authority level."""
    repo = InMemoryIdentityDecisionRepository()
    with pytest.raises(ValueError):
        await repo.save(_decision(authority="arcface_authority"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authority",
    ["operator", "direct_face", "posterior", "temporal_prior", "none", "reid_gallery"],
)
async def test_save_accepts_every_vocabulary_member(authority: str) -> None:
    repo = InMemoryIdentityDecisionRepository()
    decision = _decision(authority=authority)
    await repo.save(decision)
    stored = await repo.get_decision(decision.decision_id)
    assert stored is not None
    assert stored.authority == authority
