"""Postgres parity for the M06 identity correction repository.

Proves the asyncpg implementation matches the in-memory contract for the
effective-identity projection, range supersession, and job/ack completion.
Marked @pytest.mark.integration; CI selects this marker against a testcontainer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.domain import (
    IdentityRevisionJob,
    IdentityRevisionRange,
    IdentitySegmentCorrection,
    ProjectionAck,
)
from app.storage.postgres.correction_repo import PostgresIdentityCorrectionRepository

pytestmark = pytest.mark.integration

T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


@pytest.mark.asyncio
async def test_effective_identity_operator_wins(db_pool: Any) -> None:
    repo = PostgresIdentityCorrectionRepository(db_pool)
    ph_id = _uuid()
    rev = _uuid()
    await repo.save_range(
        IdentityRevisionRange(
            range_id=_uuid(),
            revision_id=rev,
            ph_id=ph_id,
            authority="inferred",
            range_start=T0,
            range_end=T0 + timedelta(seconds=60),
            effective_identity_id="bob",
            created_at=T0,
        )
    )
    await repo.save_range(
        IdentityRevisionRange(
            range_id=_uuid(),
            revision_id=rev,
            ph_id=ph_id,
            authority="operator",
            range_start=T0,
            range_end=T0 + timedelta(seconds=60),
            effective_identity_id="alice",
            created_at=T0 + timedelta(seconds=1),
        )
    )

    ident, authority = await repo.effective_identity(ph_id, T0 + timedelta(seconds=30))
    assert ident == "alice"
    assert authority == "operator"


@pytest.mark.asyncio
async def test_supersede_excludes_range(db_pool: Any) -> None:
    repo = PostgresIdentityCorrectionRepository(db_pool)
    ph_id = _uuid()
    old_id = _uuid()
    new_id = _uuid()
    await repo.save_range(
        IdentityRevisionRange(
            range_id=old_id,
            revision_id=_uuid(),
            ph_id=ph_id,
            authority="operator",
            range_start=T0,
            range_end=T0 + timedelta(seconds=60),
            effective_identity_id="bob",
            created_at=T0,
        )
    )
    await repo.save_range(
        IdentityRevisionRange(
            range_id=new_id,
            revision_id=_uuid(),
            ph_id=ph_id,
            authority="operator",
            range_start=T0,
            range_end=T0 + timedelta(seconds=60),
            effective_identity_id="alice",
            created_at=T0 + timedelta(seconds=2),
        )
    )
    await repo.supersede_range(old_id, by_range_id=new_id)

    live = await repo.list_ranges(ph_id, live_only=True)
    assert [r.range_id for r in live] == [new_id]
    ident, _ = await repo.effective_identity(ph_id, T0 + timedelta(seconds=10))
    assert ident == "alice"


@pytest.mark.asyncio
async def test_job_ack_completion(db_pool: Any) -> None:
    repo = PostgresIdentityCorrectionRepository(db_pool)
    rev = _uuid()
    await repo.save_job(
        IdentityRevisionJob(
            job_id=_uuid(),
            revision_id=rev,
            status="applying",
            required_projections=("cts_internal", "cc"),
        )
    )
    await repo.record_ack(
        ProjectionAck(revision_id=rev, consumer="cts_internal", schema_version="1")
    )
    assert await repo.complete_job_if_ready(rev) is False
    await repo.record_ack(ProjectionAck(revision_id=rev, consumer="cc", schema_version="1"))
    assert await repo.complete_job_if_ready(rev) is True
    job = await repo.get_job(rev)
    assert job is not None and job.status == "completed"


@pytest.mark.asyncio
async def test_correction_round_trip(db_pool: Any) -> None:
    repo = PostgresIdentityCorrectionRepository(db_pool)
    ph_id = _uuid()
    correction = IdentitySegmentCorrection(
        correction_id=_uuid(),
        ph_id=ph_id,
        actor="user:carol",
        reason_code="track_handoff",
        observation_start=T0,
        observation_end=T0 + timedelta(seconds=30),
        base_ph_version=7,
        revision_id=_uuid(),
        target_identity_id="alice",
        reviewed_bbox={"x": 1, "y": 2, "w": 3, "h": 4},
    )
    await repo.save_correction(correction)
    got = await repo.get_correction(correction.correction_id)
    assert got is not None
    assert got.actor == "user:carol"
    assert got.reason_code == "track_handoff"
    assert got.reviewed_bbox == {"x": 1, "y": 2, "w": 3, "h": 4}
    listed = await repo.list_corrections(ph_id)
    assert [c.correction_id for c in listed] == [correction.correction_id]
