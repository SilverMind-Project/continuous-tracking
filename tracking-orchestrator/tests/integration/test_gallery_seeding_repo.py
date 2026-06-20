"""Postgres gallery repo: online-seeded entries round-trip.

Regression for the multi-view seeding write path. Online seeding constructs a
GalleryEmbedding with no originating tracklet, so origin_tracklet_id keeps its
domain default of "" (empty string). The reid_gallery.origin_tracklet_id column
is a nullable UUID, and asyncpg rejects "" as an invalid UUID. The repo must
coerce empty to NULL. The InMemory repo does not validate UUIDs, so only a real
Postgres round-trip catches this; hence an integration test.

Marked @pytest.mark.integration; CI selects this marker.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.domain import GalleryEmbedding, Identity
from app.storage.postgres.gallery_repo import PostgresGalleryRepository

pytestmark = pytest.mark.integration

_EMB = [0.01 * (i % 11) for i in range(768)]


@pytest.mark.asyncio
async def test_seeded_entry_without_tracklet_id_round_trips(db_pool: Any) -> None:
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(
        Identity(identity_id="grandma", display_name="grandma", enrolled_at=datetime.now(UTC))
    )

    # As _seed_multiview_gallery builds it: no origin_tracklet_id (defaults "").
    entry = GalleryEmbedding(
        gallery_entry_id="11111111-1111-1111-1111-111111111111",
        identity_id="grandma",
        embedding=tuple(_EMB),
        seen_at=datetime.now(UTC),
        quality=0.8,
        face_confirmed=True,
        camera_id="cam01",
        orientation=0,
        state="operator_verified",
    )
    assert entry.origin_tracklet_id == ""  # the condition that broke the write

    await repo.upsert_gallery_entry(entry)

    stored = await repo.list_gallery_entries(identity_id="grandma", active_only=False)
    assert len(stored) == 1
    assert stored[0].identity_id == "grandma"
    assert stored[0].orientation == 0
    assert stored[0].origin_tracklet_id == ""  # NULL in DB maps back to ""
