"""Co-presence linking: unit tests for the guardrail logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import CoPresenceLink
from app.storage.base import InMemoryCoPresenceRepository

_NOW = datetime.now(UTC)


@pytest.mark.asyncio
async def test_upsert_and_list_by_identity() -> None:
    """Co-presence link written for two PHs with same identity is retrievable."""
    repo = InMemoryCoPresenceRepository()
    link = CoPresenceLink(
        id="link-1",
        group_id="g1",
        ph_id_a="ph-1",
        ph_id_b="ph-2",
        identity_id="alice",
        first_observed_at=_NOW,
        last_observed_at=_NOW,
    )
    await repo.upsert_link(link)

    links = await repo.list_by_identity("alice")
    assert len(links) == 1
    assert links[0].ph_id_a == "ph-1"
    assert links[0].ph_id_b == "ph-2"
    assert links[0].identity_id == "alice"


@pytest.mark.asyncio
async def test_get_active_link_sorted_lookup() -> None:
    """get_active_link normalises input order so either direction works."""
    repo = InMemoryCoPresenceRepository()
    link = CoPresenceLink(
        id="link-1",
        group_id="g1",
        ph_id_a="ph-a",
        ph_id_b="ph-z",
        identity_id="alice",
        first_observed_at=_NOW,
        last_observed_at=_NOW,
    )
    await repo.upsert_link(link)
    # Look up in either direction.
    found_a = await repo.get_active_link("ph-a", "ph-z")
    assert found_a is not None
    found_b = await repo.get_active_link("ph-z", "ph-a")
    assert found_b is not None


@pytest.mark.asyncio
async def test_list_by_ph_finds_both_sides() -> None:
    """list_by_ph returns links where the PH is either ph_id_a or ph_id_b."""
    repo = InMemoryCoPresenceRepository()
    link = CoPresenceLink(
        id="link-1",
        group_id="g1",
        ph_id_a="ph-1",
        ph_id_b="ph-2",
        identity_id="alice",
        first_observed_at=_NOW,
        last_observed_at=_NOW,
    )
    await repo.upsert_link(link)

    links_a = await repo.list_by_ph("ph-1")
    assert len(links_a) == 1
    links_b = await repo.list_by_ph("ph-2")
    assert len(links_b) == 1


@pytest.mark.asyncio
async def test_list_by_group_filters_correctly() -> None:
    """list_by_group returns only links for the given group."""
    repo = InMemoryCoPresenceRepository()
    await repo.upsert_link(
        CoPresenceLink(
            id="l1",
            group_id="g1",
            ph_id_a="ph-a",
            ph_id_b="ph-b",
            identity_id="alice",
            first_observed_at=_NOW,
            last_observed_at=_NOW,
        )
    )
    await repo.upsert_link(
        CoPresenceLink(
            id="l2",
            group_id="g2",
            ph_id_a="ph-c",
            ph_id_b="ph-d",
            identity_id="bob",
            first_observed_at=_NOW,
            last_observed_at=_NOW,
        )
    )

    g1_links = await repo.list_by_group("g1")
    assert len(g1_links) == 1
    assert g1_links[0].identity_id == "alice"


@pytest.mark.asyncio
async def test_different_identities_not_listed_together() -> None:
    """Two links with different identities are distinct."""
    repo = InMemoryCoPresenceRepository()
    await repo.upsert_link(
        CoPresenceLink(
            id="l1",
            group_id="g1",
            ph_id_a="ph-a",
            ph_id_b="ph-b",
            identity_id="alice",
            first_observed_at=_NOW,
            last_observed_at=_NOW,
        )
    )
    await repo.upsert_link(
        CoPresenceLink(
            id="l2",
            group_id="g1",
            ph_id_a="ph-c",
            ph_id_b="ph-d",
            identity_id="bob",
            first_observed_at=_NOW,
            last_observed_at=_NOW,
        )
    )

    alice_links = await repo.list_by_identity("alice")
    assert len(alice_links) == 1
    bob_links = await repo.list_by_identity("bob")
    assert len(bob_links) == 1
    # No identity overlap.
    assert alice_links[0].identity_id != bob_links[0].identity_id


@pytest.mark.asyncio
async def test_get_active_link_returns_none_for_missing() -> None:
    """get_active_link returns None when no link exists for the pair."""
    repo = InMemoryCoPresenceRepository()
    result = await repo.get_active_link("ph-x", "ph-y")
    assert result is None


@pytest.mark.asyncio
async def test_upsert_is_idempotent() -> None:
    """Writing the same link twice does not create duplicates."""
    repo = InMemoryCoPresenceRepository()
    link = CoPresenceLink(
        id="link-1",
        group_id="g1",
        ph_id_a="ph-1",
        ph_id_b="ph-2",
        identity_id="alice",
        first_observed_at=_NOW,
        last_observed_at=_NOW,
    )
    await repo.upsert_link(link)
    await repo.upsert_link(link)

    all_links = await repo.list_by_identity("alice")
    assert len(all_links) == 1
