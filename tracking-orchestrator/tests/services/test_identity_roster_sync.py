"""Tests for IdentityRosterSyncService and GalleryRepository.purge_identity."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import GalleryEmbedding, Identity
from app.pipeline.gallery_cache import GalleryCache
from app.services.identity_roster_sync import IdentityRosterSyncService
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig


class _StubMembersSource:
    def __init__(self, members: list[tuple[str, str]] | None) -> None:
        self.members = members

    async def list_enrolled_members(self) -> list[tuple[str, str]] | None:
        return self.members


@pytest.mark.asyncio
async def test_sync_once_upserts_enrolled_and_purges_stale_identities() -> None:
    repo = InMemoryGalleryRepository()
    now = datetime.now(UTC)
    await repo.upsert_identity(
        Identity(identity_id="amma", display_name="Amma", enrolled_at=now, is_active=True)
    )
    await repo.upsert_identity(
        Identity(identity_id="grandma", display_name="Grandma", enrolled_at=now, is_active=True)
    )
    await repo.upsert_identity(
        Identity(identity_id="sriram", display_name="sriram", enrolled_at=now, is_active=True)
    )
    await repo.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="ge-amma",
            identity_id="amma",
            embedding=[0.1] * 768,
            seen_at=now,
            quality=0.9,
            state="operator_verified",
        )
    )

    cache = GalleryCache(repo)
    resolver = IdentityResolver(gallery_repo=repo, config=ResolverConfig(), gallery_cache=cache)
    source = _StubMembersSource([("sriram", "sriram"), ("vasanthi-amma", "vasanthi")])
    service = IdentityRosterSyncService(
        gallery_repo=repo,
        members_source=source,
        identity_resolver=resolver,
        gallery_cache=cache,
    )

    upserted, purged = await service.sync_once()

    assert upserted == 1  # vasanthi-amma added (sriram already active with same name)
    assert purged == 2  # amma and grandma removed
    identities = await repo.list_identities(active_only=False)
    assert {i.identity_id for i in identities} == {"sriram", "vasanthi-amma"}
    assert await repo.get_gallery_entry("ge-amma") is None


@pytest.mark.asyncio
async def test_sync_once_does_not_purge_on_upstream_failure_or_empty_roster() -> None:
    repo = InMemoryGalleryRepository()
    now = datetime.now(UTC)
    await repo.upsert_identity(
        Identity(identity_id="sriram", display_name="sriram", enrolled_at=now, is_active=True)
    )

    # Upstream failure (None) preserves existing identities
    fail_service = IdentityRosterSyncService(
        gallery_repo=repo,
        members_source=_StubMembersSource(None),
    )
    assert await fail_service.sync_once() == (0, 0)
    assert len(await repo.list_identities(active_only=False)) == 1

    # Empty upstream roster preserves existing identities (safety guard)
    empty_service = IdentityRosterSyncService(
        gallery_repo=repo,
        members_source=_StubMembersSource([]),
    )
    assert await empty_service.sync_once() == (0, 0)
    assert len(await repo.list_identities(active_only=False)) == 1
