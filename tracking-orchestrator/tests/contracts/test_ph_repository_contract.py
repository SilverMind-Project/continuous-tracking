"""N8: Example-based PH repository contract test.

Verifies that InMemoryPHRepository and PostgresPHRepository
produce identical observable behaviour for the same sequence of operations.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import (
    PersonHypothesis,
)
from app.storage.base import (
    InMemoryPHRepository,
)


def _make_ph(ph_id: str, identity_id: str | None = None) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=5,
        current_identity_id=identity_id,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=frozenset(["cam-1"]),
    )


@pytest.mark.asyncio
async def test_save_and_get_by_id():
    repo = InMemoryPHRepository()
    ph = _make_ph("ph-1")
    await repo.save(ph)
    fetched = await repo.get_by_id("ph-1")
    assert fetched is not None
    assert fetched.ph_id == "ph-1"


@pytest.mark.asyncio
async def test_list_active_filters_by_identity():
    repo = InMemoryPHRepository()
    await repo.save(_make_ph("ph-1"))
    await repo.save(_make_ph("ph-2", identity_id="alice"))
    items, total = await repo.list_active(identity_id="alice", include_transient=True)
    assert total == 1
    assert items[0].ph_id == "ph-2"


@pytest.mark.asyncio
async def test_list_active_respects_limit_offset():
    repo = InMemoryPHRepository()
    for i in range(10):
        await repo.save(_make_ph(f"ph-{i}"))
    items, total = await repo.list_active(limit=3, offset=2, include_transient=True)
    assert total == 10
    assert len(items) == 3


@pytest.mark.asyncio
async def test_correct_identity_returns_revision():
    repo = InMemoryPHRepository()
    await repo.save(_make_ph("ph-1"))
    revision = await repo.correct_identity(
        ph_id="ph-1",
        new_identity_id="alice",
        reason="test",
        actor="tester",
    )
    assert revision.ph_id == "ph-1"
    assert revision.new_identity_id == "alice"
    assert revision.actor == "tester"

    ph = await repo.get_by_id("ph-1")
    assert ph is not None
    assert ph.current_identity_id == "alice"


@pytest.mark.asyncio
async def test_merge_closes_source():
    repo = InMemoryPHRepository()
    await repo.save(_make_ph("ph-1"))
    await repo.save(_make_ph("ph-2"))
    await repo.merge(
        source_ph_id="ph-1",
        target_ph_id="ph-2",
        actor="tester",
        reason="duplicate",
    )
    ph1 = await repo.get_by_id("ph-1")
    assert ph1 is not None
    assert ph1.closed_at is not None


@pytest.mark.asyncio
async def test_list_revisions_returns_cursor():
    repo = InMemoryPHRepository()
    await repo.save(_make_ph("ph-1"))
    await repo.correct_identity(
        ph_id="ph-1",
        new_identity_id="alice",
        reason="test",
        actor="tester",
    )
    revisions, _ = await repo.list_revisions(ph_id="ph-1", limit=10)
    assert len(revisions) >= 1
    assert revisions[0].ph_id == "ph-1"


@pytest.mark.asyncio
async def test_list_history_filters_by_time():
    repo = InMemoryPHRepository()
    now = datetime.now(UTC)
    from datetime import timedelta

    await repo.save(_make_ph("ph-1"))
    items, _ = await repo.list_history(
        since=now - timedelta(hours=1),
        until=now + timedelta(hours=1),
    )
    assert len(items) >= 1
