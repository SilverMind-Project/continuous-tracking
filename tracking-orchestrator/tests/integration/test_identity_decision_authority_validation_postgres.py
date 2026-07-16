"""M07 (F9) repository boundary validation: Postgres half.

Proves ``PostgresIdentityDecisionRepository.save`` rejects the identical
out-of-vocabulary ``authority`` values as the InMemory peer
(``tests/storage/test_identity_decision_authority_validation.py``, fast gate) --
parity for the repository-boundary CHECK-constraint substitute this program uses
instead of a migration (the ``authority`` column itself stays plain TEXT).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from app.domain import IdentityProvenanceDecision
from app.storage.postgres.identity_decision_repo import PostgresIdentityDecisionRepository

pytestmark = pytest.mark.integration

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
async def test_save_rejects_identity_id_as_authority(db_pool: Any) -> None:
    repo = PostgresIdentityDecisionRepository(db_pool)
    with pytest.raises(ValueError, match="amma"):
        await repo.save(_decision(authority="amma"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authority",
    ["operator", "direct_face", "posterior", "temporal_prior", "none", "reid_gallery"],
)
async def test_save_accepts_every_vocabulary_member(db_pool: Any, authority: str) -> None:
    repo = PostgresIdentityDecisionRepository(db_pool)
    decision = _decision(authority=authority)
    await repo.save(decision)
    stored = await repo.get_decision(decision.decision_id)
    assert stored is not None
    assert stored.authority == authority
