"""PostgresPHRepository parity with the InMemory peer for the three identity-clock operations.

An evidence-backed commit stores its evidence time separately from its commit time; for a face
template the evidence is older than the commit. Both repositories must keep the two apart, and
neither prior-only maintenance nor a demotion may touch the evidence time.

Marked @pytest.mark.integration: skipped by make check, included by make ci.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from app.domain import PersonHypothesis
from app.storage.base import InMemoryPHRepository
from app.storage.postgres.ph_repo import PostgresPHRepository

_T0 = datetime(2026, 11, 14, 9, 0, 0, tzinfo=UTC)


def _ph() -> PersonHypothesis:
    return PersonHypothesis(
        ph_id="00000000-0000-0000-0000-00000000c10c",
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=tuple(float(i == j) for i in range(4) for j in range(4)),
        born_at=_T0,
        last_seen_at=_T0 + timedelta(seconds=12),
        last_seen_camera="cam-1",
        observation_count=5,
    )


def _clock(ph: PersonHypothesis | None) -> tuple[str | None, datetime | None, datetime | None]:
    assert ph is not None
    return (
        ph.current_identity_id,
        ph.current_identity_committed_at,
        ph.last_independent_identity_evidence_at,
    )


@pytest.mark.integration
async def test_identity_clock_operations_match_inmemory(db_pool: asyncpg.Pool) -> None:
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO continuous_tracking.identities (identity_id, display_name) "
            "VALUES ('alice', 'alice')"
        )
    postgres = PostgresPHRepository(db_pool)
    inmemory = InMemoryPHRepository()
    member_at = _T0
    applied_at = _T0 + timedelta(seconds=12)
    later = _T0 + timedelta(seconds=20)

    trace: list[list[tuple[str | None, datetime | None, datetime | None]]] = []
    for repo in (postgres, inmemory):
        await repo.save(_ph())
        ph_id = _ph().ph_id
        steps = []
        # Evidence older than the commit: a template applied 12 s after its newest face.
        await repo.evidence_backed_commit(
            ph_id=ph_id, identity_id="alice", evidence_at=member_at, committed_at=applied_at
        )
        steps.append(_clock(await repo.get(ph_id)))
        await repo.prior_only_update(ph_id=ph_id, identity_id="alice", committed_at=later)
        steps.append(_clock(await repo.get(ph_id)))
        await repo.clear_to_unknown(ph_id=ph_id, committed_at=later)
        steps.append(_clock(await repo.get(ph_id)))
        trace.append(steps)

    assert trace[0] == trace[1]
    assert trace[0] == [
        ("alice", applied_at, member_at),
        ("alice", later, member_at),
        (None, later, member_at),
    ]
