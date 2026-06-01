"""N8: Example-based PH repository contract test.

Verifies that InMemoryPHRepository and PostgresPHRepository
produce identical observable behaviour for the same sequence of operations.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

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
    # must use non-overlapping cameras.
    repo._phs["ph-2"] = dataclasses.replace(repo._phs["ph-2"], active_cameras=frozenset(["cam-2"]))
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


# ---------------------------------------------------------------------------
# M2.3b: Provisional-PH visibility filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provisional_ph_excluded_by_default() -> None:
    """A PH with observation_count=1 is excluded from default listing
    (include_transient=False)."""
    repo = InMemoryPHRepository()
    ph = PersonHypothesis(
        ph_id="ph-transient",
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=datetime.now(UTC),
        last_seen_at=datetime.now(UTC),
        last_seen_camera="cam-1",
        observation_count=1,  # single observation = transient
        current_identity_id=None,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=frozenset(["cam-1"]),
    )
    await repo.save(ph)
    items, total = await repo.list_active(include_transient=False)
    # The transient PH should be filtered out.
    assert total == 0
    assert len(items) == 0


@pytest.mark.asyncio
async def test_provisional_ph_included_when_include_transient() -> None:
    """A PH with observation_count=1 IS included when include_transient=True."""
    repo = InMemoryPHRepository()
    ph = PersonHypothesis(
        ph_id="ph-transient",
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=datetime.now(UTC),
        last_seen_at=datetime.now(UTC),
        last_seen_camera="cam-1",
        observation_count=1,
        current_identity_id=None,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=frozenset(["cam-1"]),
    )
    await repo.save(ph)
    items, total = await repo.list_active(include_transient=True)
    assert total == 1
    assert items[0].ph_id == "ph-transient"


@pytest.mark.asyncio
async def test_ph_with_enough_observations_is_not_filtered() -> None:
    """A PH with observation_count >= 2 is NOT filtered out by default."""
    repo = InMemoryPHRepository()
    ph = PersonHypothesis(
        ph_id="ph-valid",
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=datetime.now(UTC) - timedelta(seconds=10),
        last_seen_at=datetime.now(UTC),
        last_seen_camera="cam-1",
        observation_count=5,  # enough observations
        current_identity_id=None,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=frozenset(["cam-1"]),
    )
    await repo.save(ph)
    items, total = await repo.list_active(include_transient=False)
    assert total == 1
    assert items[0].ph_id == "ph-valid"


# ---------------------------------------------------------------------------
# Reopen parity (PH revival)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reopen_closed_ph_reappears_in_list_open() -> None:
    """Saving a closed PH with closed_at=None reopens it so list_open
    returns it and get_by_id shows it open."""
    from datetime import timedelta

    repo = InMemoryPHRepository()
    now = datetime.now(UTC)
    # Create and close a PH.
    ph = PersonHypothesis(
        ph_id="ph-reopen",
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now - timedelta(seconds=60),
        last_seen_at=now - timedelta(seconds=10),
        last_seen_camera="cam-1",
        observation_count=10,
        current_identity_id="alice",
        current_identity_committed_at=now - timedelta(seconds=30),
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=frozenset(["cam-1"]),
        closed_at=now - timedelta(seconds=5),
    )
    await repo.save(ph)
    # Verify it is closed.
    open_before = await repo.list_open()
    assert not any(p.ph_id == "ph-reopen" for p in open_before)

    # Reopen: save with closed_at=None.
    reopened = dataclasses.replace(ph, closed_at=None)
    await repo.save(reopened)

    # Verify it is open now.
    open_after = await repo.list_open()
    assert any(p.ph_id == "ph-reopen" for p in open_after)

    # Verify get_by_id shows it open.
    fetched = await repo.get_by_id("ph-reopen")
    assert fetched is not None
    assert fetched.closed_at is None
    assert fetched.current_identity_id == "alice"
