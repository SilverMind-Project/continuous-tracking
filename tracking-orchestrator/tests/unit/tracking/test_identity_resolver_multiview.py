"""Orientation-aware resolver gallery query tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.domain import (
    GalleryEmbedding,
    Identity,
    OrientationBin,
    ViewPrototype,
)
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig


def _normalize(vals: list[float]) -> list[float]:
    arr = np.asarray(vals, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr.tolist()


# Distinct embeddings for different views.
_FRONT_EMB = _normalize([1.0] + [0.0] * 767)
_BACK_EMB = _normalize([0.0] * 384 + [1.0] + [0.0] * 383)

_ALICE_ID = "alice"
_BOB_ID = "bob"


class _FakePH:
    """Minimal IdentityResolvableEntity for testing multiview resolver."""

    def __init__(
        self,
        entity_id: str,
        obs_ids: list[str],
        camera_ids: list[str],
        current_identity_id: str | None = None,
        view_prototypes: tuple[ViewPrototype, ...] = (),
    ) -> None:
        self.entity_id = entity_id
        self._obs_ids = obs_ids
        self._camera_ids = camera_ids
        self.current_identity_id = current_identity_id
        self.current_identity_committed_at = None
        self.last_seen_at = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        self.started_at = self.last_seen_at
        self._view_prototypes = view_prototypes

    @property
    def observation_ids(self) -> list[str]:
        return self._obs_ids

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def view_prototypes(self) -> tuple[ViewPrototype, ...]:
        return self._view_prototypes


@pytest.fixture
async def gallery_with_alice_back() -> InMemoryGalleryRepository:
    """Gallery where alice has back-facing entries and bob has front-facing entries."""
    repo = InMemoryGalleryRepository()
    await repo.upsert_identity(
        Identity(
            identity_id=_ALICE_ID,
            display_name="Alice",
            enrolled_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    await repo.upsert_identity(
        Identity(
            identity_id=_BOB_ID,
            display_name="Bob",
            enrolled_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    # Add alice back entries.
    for i in range(5):
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"alice-back-{i}",
                identity_id=_ALICE_ID,
                embedding=_BACK_EMB,
                seen_at=datetime(2026, 6, 1, tzinfo=UTC),
                quality=0.8,
                face_confirmed=True,
                orientation=OrientationBin.BACK,
            )
        )
    # Add bob front entries.
    for i in range(5):
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"bob-front-{i}",
                identity_id=_BOB_ID,
                embedding=_FRONT_EMB,
                seen_at=datetime(2026, 6, 1, tzinfo=UTC),
                quality=0.8,
                face_confirmed=True,
                orientation=OrientationBin.FRONT,
            )
        )
    return repo


@pytest.mark.asyncio
async def test_multiview_gallery_max_over_views(
    gallery_with_alice_back: InMemoryGalleryRepository,
) -> None:
    """A PH whose back prototype matches alice's back gallery entries
    resolves to alice via max-over-views."""
    config = ResolverConfig(
        commit_prob=0.65,
        enable_multiview_gallery=True,
    )
    resolver = IdentityResolver(
        gallery_repo=gallery_with_alice_back,
        config=config,
    )

    back_proto = ViewPrototype(
        orientation=OrientationBin.BACK,
        embedding=tuple(_BACK_EMB),
        count=5,
    )
    ph = _FakePH(
        entity_id="ph-1",
        obs_ids=["obs-1"],
        camera_ids=["cam-1"],
        view_prototypes=(back_proto,),
    )

    outcome = await resolver.resolve(
        hypotheses=[ph],
        new_face_anchors=[],
        captured_at=datetime(2026, 6, 1, 12, 0, 1, tzinfo=UTC),
    )

    assert len(outcome.decisions) == 1
    decision = outcome.decisions[0]
    top_id, top_prob = decision.posterior.top_identity()
    # The back prototype should match alice's back entries.
    assert top_id == _ALICE_ID
    assert top_prob > 0.5


@pytest.mark.asyncio
async def test_multiview_off_uses_single_query(
    gallery_with_alice_back: InMemoryGalleryRepository,
) -> None:
    """With multiview off, the resolver uses the mean-of-embeddings single query."""
    config = ResolverConfig(
        commit_prob=0.65,
        enable_multiview_gallery=False,
    )
    resolver = IdentityResolver(
        gallery_repo=gallery_with_alice_back,
        config=config,
    )

    ph = _FakePH(
        entity_id="ph-1",
        obs_ids=["obs-1"],
        camera_ids=["cam-1"],
        view_prototypes=(),
    )

    outcome = await resolver.resolve(
        hypotheses=[ph],
        new_face_anchors=[],
        captured_at=datetime(2026, 6, 1, 12, 0, 1, tzinfo=UTC),
    )

    assert len(outcome.decisions) == 1
    decision = outcome.decisions[0]
    assert decision.posterior is not None


@pytest.mark.asyncio
async def test_multiview_shadow_does_not_crash(
    gallery_with_alice_back: InMemoryGalleryRepository,
) -> None:
    """When multiview is OFF but the entity has prototypes, the shadow
    comparison runs without crashing."""
    config = ResolverConfig(
        commit_prob=0.65,
        enable_multiview_gallery=False,
    )
    resolver = IdentityResolver(
        gallery_repo=gallery_with_alice_back,
        config=config,
    )

    back_proto = ViewPrototype(
        orientation=OrientationBin.BACK,
        embedding=tuple(_BACK_EMB),
        count=5,
    )
    ph = _FakePH(
        entity_id="ph-1",
        obs_ids=["alice-back-0"],
        camera_ids=["cam-1"],
        view_prototypes=(back_proto,),
    )

    outcome = await resolver.resolve(
        hypotheses=[ph],
        new_face_anchors=[],
        captured_at=datetime(2026, 6, 1, 12, 0, 1, tzinfo=UTC),
    )

    # The shadow comparison should have run without crashing.
    assert len(outcome.decisions) == 1
