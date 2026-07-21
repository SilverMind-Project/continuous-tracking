"""M03: hard vote-age cutoff wiring across every resolver gallery query path.

Identity-continuity M03 (finding V3): ``GalleryRepository.search_similar``'s
``max_age_seconds`` filter existed as unused plumbing -- no resolver call site
passed it. ``IdentityResolver._gallery_search_kwargs()`` is the single helper
every ``search_similar`` call site must route through (task 2 of the
milestone) so a future fourth call site cannot forget the cutoff or the
``VOTING_STATES`` filter. These tests prove:

* a structural guard that every current call site uses the helper;
* every live vote query (single-query fallback, multiview, and the
  cross-camera-assist diagnostic query) carries the configured cutoff;
* ``gallery_vote_max_age_s=None`` disables the cutoff everywhere;
* the multiview shadow comparison receives the identical cutoff as the live
  query it is compared against (apples-to-apples).
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from app.domain import GalleryEmbedding, Identity, OrientationBin, ViewPrototype
from app.storage.base import VOTING_STATES
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

# Real wall-clock time, not a fixed past date: InMemoryGalleryRepository.search_similar's
# max_age_seconds cutoff filters against datetime.now(UTC) internally, not against the
# resolve() `captured_at` parameter. A hardcoded past _NOW eventually ages the seeded
# gallery entry out of the 12h (43200s) cutoff purely from real time passing, unrelated
# to any resolver behavior under test (found and fixed identity-continuity M09, 2026-07-21).
_NOW = datetime.now(UTC)
_ALICE_ID = "alice"


class _FakePH:
    """Minimal IdentityResolvableEntity, mirroring the multiview test fixture."""

    def __init__(
        self,
        entity_id: str,
        obs_ids: list[str],
        view_prototypes: tuple[ViewPrototype, ...] = (),
    ) -> None:
        self.entity_id = entity_id
        self._obs_ids = obs_ids
        self.camera_ids = ["cam-1"]
        self.current_identity_id = None
        self.current_identity_committed_at = None
        self.last_independent_identity_evidence_at = None
        self.last_seen_at = _NOW
        self.started_at = _NOW
        self._view_prototypes = view_prototypes

    @property
    def observation_ids(self) -> list[str]:
        return self._obs_ids

    @property
    def view_prototypes(self) -> tuple[ViewPrototype, ...]:
        return self._view_prototypes


class _RecordingGalleryRepo(InMemoryGalleryRepository):
    """Records the kwargs every search_similar call receives."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
        states: frozenset[str] | None = VOTING_STATES,
    ) -> list[tuple[GalleryEmbedding, float]]:
        self.calls.append({"max_age_seconds": max_age_seconds, "states": states})
        return await super().search_similar(
            embedding,
            limit=limit,
            camera_id=camera_id,
            max_age_seconds=max_age_seconds,
            states=states,
        )


async def _seed_alice(repo: InMemoryGalleryRepository) -> None:
    """Seed a verified alice entry whose origin_tracklet_id matches "obs-1".

    The single-query fallback path first calls
    ``list_gallery_entries_for_tracklets`` (task 4's uncut corpus-building
    read) to build its query embedding from the PH's own recent entries;
    without a matching ``origin_tracklet_id`` that call returns empty and
    ``_from_gallery`` short-circuits before ever reaching ``search_similar``.
    """
    await repo.upsert_identity(
        Identity(identity_id=_ALICE_ID, display_name="Alice", enrolled_at=_NOW)
    )
    await repo.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="g-alice",
            identity_id=_ALICE_ID,
            embedding=[1.0] + [0.0] * 7,
            seen_at=_NOW,
            origin_tracklet_id="obs-1",
            state="operator_verified",
        )
    )


def test_all_call_sites_use_the_shared_helper() -> None:
    """Structural guard (mirrors test_gallery_scoring_unification.py's pattern):
    a future call site that recomputes kwargs inline instead of delegating to
    ``_gallery_search_kwargs()`` is a defect, per the milestone's task 2.
    """
    for method in (
        IdentityResolver._from_gallery,
        IdentityResolver._from_gallery_multiview,
        IdentityResolver._has_cross_camera_reid_assist,
    ):
        source = inspect.getsource(method)
        assert "self._gallery_search_kwargs()" in source, (
            f"{method.__name__} must route through _gallery_search_kwargs()"
        )


@pytest.mark.asyncio
async def test_all_vote_sites_pass_cutoff() -> None:
    repo = _RecordingGalleryRepo()
    await _seed_alice(repo)
    config = ResolverConfig(gallery_vote_max_age_s=43200, enable_multiview_gallery=True)
    resolver = IdentityResolver(gallery_repo=repo, config=config)

    # Single-query fallback path (no view prototypes).
    ph_fallback = _FakePH(entity_id="ph-fallback", obs_ids=["obs-1"])
    await resolver._from_gallery(ph_fallback)

    # Multiview path.
    proto = ViewPrototype(orientation=OrientationBin.FRONT, embedding=(1.0,) + (0.0,) * 7, count=5)
    ph_multiview = _FakePH(entity_id="ph-multiview", obs_ids=["obs-2"], view_prototypes=(proto,))
    await resolver._from_gallery(ph_multiview)

    # Cross-camera-assist diagnostic query.
    await resolver._has_cross_camera_reid_assist(ph_fallback, _ALICE_ID)

    assert len(repo.calls) == 3
    for call in repo.calls:
        assert call["max_age_seconds"] == 43200
        assert call["states"] == VOTING_STATES


@pytest.mark.asyncio
async def test_none_disables_cutoff() -> None:
    repo = _RecordingGalleryRepo()
    await _seed_alice(repo)
    config = ResolverConfig(gallery_vote_max_age_s=None)
    resolver = IdentityResolver(gallery_repo=repo, config=config)

    ph = _FakePH(entity_id="ph-1", obs_ids=["obs-1"])
    await resolver._from_gallery(ph)

    assert len(repo.calls) == 1
    assert repo.calls[0]["max_age_seconds"] is None
    assert repo.calls[0]["states"] == VOTING_STATES


@pytest.mark.asyncio
async def test_shadow_receives_same_cutoff() -> None:
    """The multiview-shadow comparison (``_shadow_gallery_compare`` ->
    ``_from_gallery(enable_multiview=True)``) re-runs the query with a
    different path than the live evaluation, but must carry the identical
    cutoff so the comparison stays apples-to-apples (milestone resolver call
    site note on ``:1418``/the shadow machinery).
    """
    repo = _RecordingGalleryRepo()
    await _seed_alice(repo)
    proto = ViewPrototype(orientation=OrientationBin.FRONT, embedding=(1.0,) + (0.0,) * 7, count=5)
    config = ResolverConfig(
        gallery_vote_max_age_s=43200,
        enable_multiview_gallery=False,  # live eval uses the single-query path
        multiview_shadow_sample_rate=1.0,  # always sample the multiview shadow
    )
    resolver = IdentityResolver(gallery_repo=repo, config=config)
    ph = _FakePH(entity_id="ph-1", obs_ids=["obs-1"], view_prototypes=(proto,))

    await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=_NOW)

    # Three search_similar calls fire in this cycle: the live single-query
    # fallback (line 449), the shadow multiview comparison (line 502, since
    # multiview_shadow_sample_rate=1.0 forces sampling), and the
    # cross-camera-assist diagnostic query (line 644, since this fresh
    # PH-1/alice commit has revises_previous=True). Confirmed by stack-trace
    # instrumentation, not guessed. The exact count is secondary -- every
    # query this resolve() cycle issues must carry the identical cutoff.
    assert len(repo.calls) == 3
    for call in repo.calls:
        assert call["max_age_seconds"] == 43200
        assert call["states"] == VOTING_STATES
