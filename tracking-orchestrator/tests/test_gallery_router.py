"""Tests for the gallery enrollment router.

Exercises the three behavioural contracts:
1. Happy path: creates named ``GalleryEmbedding`` rows and upserts the
   ``Identity`` record; the orphaned empty rows are untouched.
2. Missing-tracklet path: 404 when no embeddings exist for the tracklet.
3. Idempotent identity: enrolling the same identity twice does not reset
   ``enrolled_at`` on the original record.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain import GalleryEmbedding
from app.routers.gallery import router as gallery_router
from app.routers.gallery import set_context
from app.storage.base import InMemoryGalleryRepository


def _seed_embeddings(repo: InMemoryGalleryRepository, tracklet_id: str, count: int = 2) -> None:
    """Pre-populate the in-memory gallery with empty-identity entries for a tracklet."""

    async def _write() -> None:
        for i in range(count):
            entry = GalleryEmbedding(
                gallery_entry_id=f"entry-{i}",
                identity_id="",
                embedding=[0.1 * (i + 1)] * 128,
                seen_at=datetime.now(UTC),
                quality=0.9 - 0.1 * i,
                origin_tracklet_id=tracklet_id,
                camera_id="kitchen-1",
            )
            await repo.upsert_gallery_entry(entry)

    asyncio.run(_write())


@pytest.fixture
def client():
    repo = InMemoryGalleryRepository()
    _seed_embeddings(repo, tracklet_id="t-1")
    set_context(gallery_repo=repo)
    app = FastAPI()
    app.include_router(gallery_router)
    return TestClient(app), repo


class TestEnrollHappyPath:
    def test_returns_200_with_count(self, client):
        tc, _ = client
        resp = tc.post(
            "/internal/gallery/enroll",
            json={"identity_id": "grandma", "tracklet_id": "t-1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["identity_id"] == "grandma"
        assert body["enrolled_count"] == 2
        assert body["enrolled_at"]

    def test_named_entries_written(self, client):
        tc, repo = client
        tc.post(
            "/internal/gallery/enroll",
            json={"identity_id": "grandma", "tracklet_id": "t-1"},
        )

        async def _list():
            return await repo.list_gallery_entries(identity_id="grandma")

        named = asyncio.run(_list())
        assert len(named) == 2
        for e in named:
            assert e.identity_id == "grandma"
            assert e.origin_tracklet_id == "t-1"

    def test_identity_record_created(self, client):
        tc, repo = client
        tc.post(
            "/internal/gallery/enroll",
            json={
                "identity_id": "grandma",
                "tracklet_id": "t-1",
                "display_name": "Grandma",
            },
        )

        async def _get():
            return await repo.get_identity("grandma")

        identity = asyncio.run(_get())
        assert identity is not None
        assert identity.display_name == "Grandma"
        assert identity.is_active

    def test_empty_identity_rows_are_preserved(self, client):
        """Original anonymous entries must not be deleted — they are invisible to search_similar."""
        tc, repo = client
        tc.post(
            "/internal/gallery/enroll",
            json={"identity_id": "grandma", "tracklet_id": "t-1"},
        )

        async def _all():
            # list_gallery_entries only returns rows joined with identities;
            # read the raw store directly.
            return list(repo._entries.values())

        all_rows = asyncio.run(_all())
        empty = [r for r in all_rows if r.identity_id == ""]
        assert len(empty) == 2, "original anonymous entries must survive"


class TestEnrollMissingTracklet:
    def test_404_when_tracklet_has_no_embeddings(self, client):
        tc, _ = client
        resp = tc.post(
            "/internal/gallery/enroll",
            json={"identity_id": "grandma", "tracklet_id": "nonexistent-tracklet"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "gallery.no_embeddings_for_tracklet"


class TestEnrollIdempotentIdentity:
    def test_enrolled_at_preserved_on_re_enroll(self, client):
        """Re-enrolling must not overwrite enrolled_at with a newer timestamp."""
        tc, repo = client

        # First enrollment.
        tc.post(
            "/internal/gallery/enroll",
            json={"identity_id": "grandma", "tracklet_id": "t-1"},
        )

        async def _get():
            return await repo.get_identity("grandma")

        first_identity = asyncio.run(_get())
        assert first_identity is not None
        first_enrolled_at = first_identity.enrolled_at

        # Seed more entries for a second tracklet.
        _seed_embeddings(repo, tracklet_id="t-2", count=1)

        # Second enrollment.
        tc.post(
            "/internal/gallery/enroll",
            json={"identity_id": "grandma", "tracklet_id": "t-2"},
        )
        second_identity = asyncio.run(_get())
        assert second_identity is not None
        assert second_identity.enrolled_at == first_enrolled_at, (
            "enrolled_at must not be reset on subsequent enrollments"
        )
