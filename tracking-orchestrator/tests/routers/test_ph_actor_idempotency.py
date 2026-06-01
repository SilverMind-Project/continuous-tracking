"""Actor extraction and idempotency tests for PH API."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.domain import PersonHypothesis
from app.main import create_app
from app.routers.ph import set_ph_repository, set_revision_publisher
from app.storage.base import InMemoryPHRepository


@pytest.fixture
def repo() -> InMemoryPHRepository:
    return InMemoryPHRepository()


@pytest.fixture
def client(repo: InMemoryPHRepository) -> TestClient:
    app = create_app()
    set_ph_repository(repo)
    set_revision_publisher(None)
    return TestClient(app)


def _make_ph(ph_id: str, cameras: frozenset[str] | None = None) -> PersonHypothesis:
    now = datetime.now(UTC)
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.1, 0.0),
        state_cov=(0.1,) * 16,
        born_at=now,
        last_seen_at=now,
        last_seen_camera="cam-1",
        observation_count=5,
        current_identity_id=None,
        gallery_mean=None,
        height_estimate_m=None,
        active_cameras=cameras or frozenset(["cam-1"]),
    )


@pytest.mark.asyncio
async def test_actor_header_is_persisted(repo: InMemoryPHRepository):
    """Actor from X-Actor-Subject header is stored in the revision."""
    ph = _make_ph("ph-1")
    await repo.save(ph)

    revision = await repo.correct_identity(
        ph_id="ph-1",
        new_identity_id="alice",
        reason="test",
        actor="dr_smith",
    )
    assert revision.actor == "dr_smith"


@pytest.mark.asyncio
async def test_repeated_idempotency_key_returns_same_revision(repo: InMemoryPHRepository):
    """Same idempotency key returns the identical revision object."""
    ph = _make_ph("ph-1")
    await repo.save(ph)

    rev1 = await repo.correct_identity(
        ph_id="ph-1",
        new_identity_id="alice",
        reason="test",
        actor="dr_smith",
        idempotency_key="idem-abc-123",
    )
    assert rev1 is not None

    # Second call with same key returns the same revision.
    rev2 = await repo.correct_identity(
        ph_id="ph-1",
        new_identity_id="bob",  # different identity, but idempotent
        reason="test",
        actor="dr_smith",
        idempotency_key="idem-abc-123",
    )
    assert rev2 is rev1  # same object
    assert rev2.new_identity_id == "alice"  # original result, not "bob"


@pytest.mark.asyncio
async def test_idempotency_key_stored_for_merge(repo: InMemoryPHRepository):
    """Idempotency works for merge operations too."""
    await repo.save(_make_ph("ph-1", cameras=frozenset(["cam-1"])))
    await repo.save(_make_ph("ph-2", cameras=frozenset(["cam-2"])))

    rev1 = await repo.merge(
        source_ph_id="ph-1",
        target_ph_id="ph-2",
        actor="admin",
        reason="duplicate",
        idempotency_key="merge-key-1",
    )
    rev2 = await repo.merge(
        source_ph_id="ph-1",
        target_ph_id="ph-2",
        actor="admin",
        reason="duplicate",
        idempotency_key="merge-key-1",
    )
    assert rev2 is rev1
