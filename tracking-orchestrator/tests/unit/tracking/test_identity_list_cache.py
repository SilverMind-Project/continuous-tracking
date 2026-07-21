"""Unit tests for IdentityResolver identity-list TTL cache."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import Identity
from app.storage.base import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver


def _ts(seconds_offset: float = 0.0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=seconds_offset)


def _identity(name: str) -> Identity:
    return Identity(identity_id=name, display_name=name, enrolled_at=_ts())


@pytest.mark.asyncio
async def test_identity_list_loaded_on_first_resolve() -> None:
    """On the first resolve() call the identity list is fetched from the DB."""
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(_identity("alice"))
    resolver = IdentityResolver(gallery_repo=gallery)

    assert resolver._identities_loaded_at is None

    # resolve with zero hypotheses triggers the identity load path.
    await resolver.resolve(hypotheses=[], new_face_anchors=[], captured_at=_ts())

    assert "alice" in resolver._identities
    assert resolver._identities_loaded_at == _ts()


@pytest.mark.asyncio
async def test_identity_list_not_reloaded_within_ttl() -> None:
    """Within the TTL window, list_identities is not called again."""
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(_identity("alice"))
    resolver = IdentityResolver(gallery_repo=gallery)

    # Prime the cache.
    t0 = _ts(0.0)
    await resolver.resolve(hypotheses=[], new_face_anchors=[], captured_at=t0)
    first_load_time = resolver._identities_loaded_at

    # Add a new identity directly to the DB.
    await gallery.upsert_identity(_identity("bob"))

    # Resolve again within TTL (half the window).
    t1 = _ts(IdentityResolver._IDENTITY_LIST_TTL_S / 2)
    await resolver.resolve(hypotheses=[], new_face_anchors=[], captured_at=t1)

    # Cache should NOT have been refreshed — bob is not visible yet.
    assert resolver._identities_loaded_at == first_load_time
    assert "bob" not in resolver._identities


@pytest.mark.asyncio
async def test_identity_list_reloaded_after_ttl() -> None:
    """After the TTL elapses the identity list is refreshed from the DB."""
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(_identity("alice"))
    resolver = IdentityResolver(gallery_repo=gallery)

    t0 = _ts(0.0)
    await resolver.resolve(hypotheses=[], new_face_anchors=[], captured_at=t0)

    # Add a new identity to the DB.
    await gallery.upsert_identity(_identity("bob"))

    # Resolve after TTL has elapsed.
    t_after = _ts(IdentityResolver._IDENTITY_LIST_TTL_S + 1.0)
    await resolver.resolve(hypotheses=[], new_face_anchors=[], captured_at=t_after)

    assert "bob" in resolver._identities
    assert resolver._identities_loaded_at == t_after


@pytest.mark.asyncio
async def test_register_identity_bypasses_ttl() -> None:
    """register_identity() inserts directly and does not require a cache refresh."""
    gallery = InMemoryGalleryRepository()
    resolver = IdentityResolver(gallery_repo=gallery)

    # Prime the empty cache.
    await resolver.resolve(hypotheses=[], new_face_anchors=[], captured_at=_ts(0.0))
    assert len(resolver._identities) == 0

    resolver.register_identity(_identity("alice"))
    assert "alice" in resolver._identities
