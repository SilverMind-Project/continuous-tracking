"""Tests for InMemoryDoNotFuseRepository."""

from __future__ import annotations

import pytest

from app.storage.base import InMemoryDoNotFuseRepository


@pytest.fixture()
def repo() -> InMemoryDoNotFuseRepository:
    return InMemoryDoNotFuseRepository()


@pytest.mark.asyncio
async def test_add_hint_and_is_blocked(repo: InMemoryDoNotFuseRepository) -> None:
    await repo.add_hint("tr1", "gt1")
    assert await repo.is_blocked("tr1", "gt1") is True
    assert await repo.is_blocked("tr1", "gt2") is False
    assert await repo.is_blocked("tr2", "gt1") is False


@pytest.mark.asyncio
async def test_remove_hint_allows_fusion(repo: InMemoryDoNotFuseRepository) -> None:
    await repo.add_hint("tr1", "gt1")
    await repo.remove_hint("tr1", "gt1")
    assert await repo.is_blocked("tr1", "gt1") is False


@pytest.mark.asyncio
async def test_add_hint_is_idempotent(repo: InMemoryDoNotFuseRepository) -> None:
    await repo.add_hint("tr1", "gt1")
    await repo.add_hint("tr1", "gt1")  # second call must not raise
    assert await repo.is_blocked("tr1", "gt1") is True


@pytest.mark.asyncio
async def test_get_hints_for_tracklet(repo: InMemoryDoNotFuseRepository) -> None:
    await repo.add_hint("tr1", "gt1")
    await repo.add_hint("tr1", "gt2")
    await repo.add_hint("tr2", "gt1")
    hints = await repo.get_hints_for_tracklet("tr1")
    assert sorted(hints) == sorted(["gt1", "gt2"])


@pytest.mark.asyncio
async def test_get_hints_for_unknown_tracklet(repo: InMemoryDoNotFuseRepository) -> None:
    hints = await repo.get_hints_for_tracklet("unknown")
    assert hints == []


@pytest.mark.asyncio
async def test_remove_hint_does_nothing_for_unknown(repo: InMemoryDoNotFuseRepository) -> None:
    # Must not raise.
    await repo.remove_hint("unknown_tr", "unknown_gt")
