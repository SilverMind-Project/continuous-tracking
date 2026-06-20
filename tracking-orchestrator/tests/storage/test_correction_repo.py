"""Unit tests for the M06 in-memory identity correction repository.

These exercise the effective-identity projection, range supersession, and the
job/ack completion machinery without a database. The Postgres parity peer is
covered under ``tests/integration/test_correction_repo_postgres.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    IdentityRevisionJob,
    IdentityRevisionRange,
    IdentitySegmentCorrection,
    ProjectionAck,
)
from app.storage.corrections import InMemoryIdentityCorrectionRepository

T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


def _range(
    range_id: str,
    *,
    ph_id: str = "ph1",
    authority: str = "inferred",
    identity: str | None = "alice",
    start: datetime = T0,
    end: datetime = T0 + timedelta(seconds=60),
    created: datetime = T0,
    revision_id: str = "rev1",
) -> IdentityRevisionRange:
    return IdentityRevisionRange(
        range_id=range_id,
        revision_id=revision_id,
        ph_id=ph_id,
        authority=authority,  # type: ignore[arg-type]
        range_start=start,
        range_end=end,
        effective_identity_id=identity,
        created_at=created,
    )


@pytest.mark.asyncio
async def test_effective_identity_operator_beats_inferred() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_range(_range("r-inf", authority="inferred", identity="bob"))
    await repo.save_range(
        _range("r-op", authority="operator", identity="alice", created=T0 + timedelta(seconds=1))
    )

    ident, authority = await repo.effective_identity("ph1", T0 + timedelta(seconds=30))
    assert ident == "alice"
    assert authority == "operator"


@pytest.mark.asyncio
async def test_effective_identity_no_range_returns_none() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_range(_range("r1", start=T0, end=T0 + timedelta(seconds=10)))
    # Query outside the only range -> no projection, caller falls back to inference.
    ident, authority = await repo.effective_identity("ph1", T0 + timedelta(seconds=999))
    assert ident is None
    assert authority is None


@pytest.mark.asyncio
async def test_effective_identity_set_unknown_range() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_range(_range("r-op", authority="operator", identity=None))
    ident, authority = await repo.effective_identity("ph1", T0 + timedelta(seconds=5))
    assert ident is None
    assert authority == "operator"


@pytest.mark.asyncio
async def test_superseded_range_excluded_from_effective() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_range(_range("r-old", authority="operator", identity="bob"))
    await repo.save_range(
        _range("r-new", authority="operator", identity="alice", created=T0 + timedelta(seconds=2))
    )
    await repo.supersede_range("r-old", by_range_id="r-new")

    live = await repo.list_ranges("ph1", live_only=True)
    assert {r.range_id for r in live} == {"r-new"}
    ident, _ = await repo.effective_identity("ph1", T0 + timedelta(seconds=10))
    assert ident == "alice"


@pytest.mark.asyncio
async def test_operator_ranges_overlapping_conflict_detection() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_range(
        _range(
            "r-op",
            authority="operator",
            start=T0,
            end=T0 + timedelta(seconds=60),
        )
    )
    # Overlapping window finds the operator range.
    hits = await repo.operator_ranges_overlapping(
        "ph1", T0 + timedelta(seconds=30), T0 + timedelta(seconds=90)
    )
    assert [r.range_id for r in hits] == ["r-op"]
    # Non-overlapping window finds nothing.
    none = await repo.operator_ranges_overlapping(
        "ph1", T0 + timedelta(seconds=120), T0 + timedelta(seconds=180)
    )
    assert none == []


@pytest.mark.asyncio
async def test_job_completes_only_after_all_required_acks() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_job(
        IdentityRevisionJob(
            job_id="j1",
            revision_id="rev1",
            status="applying",
            required_projections=("cts_internal", "cc"),
        )
    )

    await repo.record_ack(
        ProjectionAck(revision_id="rev1", consumer="cts_internal", schema_version="1")
    )
    assert await repo.complete_job_if_ready("rev1") is False
    job = await repo.get_job("rev1")
    assert job is not None and job.status == "applying"

    await repo.record_ack(ProjectionAck(revision_id="rev1", consumer="cc", schema_version="1"))
    assert await repo.complete_job_if_ready("rev1") is True
    job = await repo.get_job("rev1")
    assert job is not None and job.status == "completed"


@pytest.mark.asyncio
async def test_failed_ack_marks_job_failed_not_completed() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    await repo.save_job(
        IdentityRevisionJob(
            job_id="j1",
            revision_id="rev1",
            status="applying",
            required_projections=("cc",),
        )
    )
    await repo.record_ack(
        ProjectionAck(revision_id="rev1", consumer="cc", schema_version="1", status="failed")
    )
    assert await repo.complete_job_if_ready("rev1") is False
    job = await repo.get_job("rev1")
    assert job is not None and job.status == "failed"


@pytest.mark.asyncio
async def test_ack_idempotent_on_replay() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    await repo.record_ack(
        ProjectionAck(revision_id="rev1", consumer="cc", schema_version="1", counts={"rows": 3})
    )
    await repo.record_ack(
        ProjectionAck(revision_id="rev1", consumer="cc", schema_version="1", counts={"rows": 3})
    )
    acks = await repo.list_acks("rev1")
    assert len(acks) == 1
    assert acks[0].counts == {"rows": 3}


@pytest.mark.asyncio
async def test_correction_round_trip_and_listing() -> None:
    repo = InMemoryIdentityCorrectionRepository()
    correction = IdentitySegmentCorrection(
        correction_id="c1",
        ph_id="ph1",
        actor="user:carol",
        reason_code="wrong_person",
        observation_start=T0,
        observation_end=T0 + timedelta(seconds=30),
        base_ph_version=7,
        revision_id="rev1",
        target_identity_id="alice",
    )
    await repo.save_correction(correction)
    assert await repo.get_correction("c1") == correction
    assert await repo.list_corrections("ph1") == [correction]
    assert await repo.list_corrections("other") == []
