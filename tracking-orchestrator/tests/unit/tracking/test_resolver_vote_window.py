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
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import GalleryEmbedding, Identity, OrientationBin, ViewPrototype
from app.storage.base import VOTING_STATES
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking import identity_resolver as identity_resolver_mod
from app.tracking.identity.channels import VoteWindow
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

# Fixed capture-time anchor: vote window and recency decay use captured_at, never wall clock.
_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
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
        seen_after: datetime | None = None,
        seen_before: datetime | None = None,
        states: frozenset[str] | None = VOTING_STATES,
        exclude_seed_sources: frozenset[str] | None = None,
        active_only: bool = True,
    ) -> list[tuple[GalleryEmbedding, float]]:
        self.calls.append(
            {
                "seen_after": seen_after,
                "seen_before": seen_before,
                "states": states,
                "active_only": active_only,
            }
        )
        return await super().search_similar(
            embedding,
            limit=limit,
            camera_id=camera_id,
            seen_after=seen_after,
            seen_before=seen_before,
            states=states,
            exclude_seed_sources=exclude_seed_sources,
            active_only=active_only,
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
    """Structural guard: every gallery vote query routes through _gallery_search_kwargs,
    and identity_resolver never reads wall-clock datetime.now().
    """
    for method in (
        IdentityResolver._from_gallery,
        IdentityResolver._from_gallery_multiview,
        IdentityResolver._has_cross_camera_reid_assist,
    ):
        source = inspect.getsource(method)
        assert "self._gallery_search_kwargs(" in source, (
            f"{method.__name__} must route through _gallery_search_kwargs()"
        )
        assert "max_age_seconds=" not in source, (
            f"{method.__name__} must not build the vote-age cutoff inline"
        )
    module_source = inspect.getsource(identity_resolver_mod)
    assert "datetime.now(" not in module_source, (
        "identity_resolver must evaluate recency and vote windows on capture time, never wall clock"
    )


@pytest.mark.asyncio
async def test_all_vote_sites_pass_cutoff() -> None:
    repo = _RecordingGalleryRepo()
    await _seed_alice(repo)
    config = ResolverConfig(gallery_vote_max_age_s=43200, enable_multiview_gallery=True)
    resolver = IdentityResolver(gallery_repo=repo, config=config)
    window = VoteWindow(as_of=_NOW, max_age_s=43200.0)

    # Single-query fallback path (no view prototypes).
    ph_fallback = _FakePH(entity_id="ph-fallback", obs_ids=["obs-1"])
    await resolver._from_gallery(ph_fallback, window=window)

    # Multiview path.
    proto = ViewPrototype(orientation=OrientationBin.FRONT, embedding=(1.0,) + (0.0,) * 7, count=5)
    ph_multiview = _FakePH(entity_id="ph-multiview", obs_ids=["obs-2"], view_prototypes=(proto,))
    await resolver._from_gallery(ph_multiview, window=window)

    # Cross-camera-assist diagnostic query.
    await resolver._has_cross_camera_reid_assist(ph_fallback, _ALICE_ID, window=window)

    assert len(repo.calls) == 3
    for call in repo.calls:
        assert call["seen_after"] == _NOW - timedelta(seconds=43200)
        assert call["seen_before"] == _NOW
        assert call["states"] == VOTING_STATES


@pytest.mark.asyncio
async def test_none_disables_cutoff() -> None:
    repo = _RecordingGalleryRepo()
    await _seed_alice(repo)
    config = ResolverConfig(gallery_vote_max_age_s=None)
    resolver = IdentityResolver(gallery_repo=repo, config=config)
    window = VoteWindow(as_of=_NOW, max_age_s=None)

    ph = _FakePH(entity_id="ph-1", obs_ids=["obs-1"])
    await resolver._from_gallery(ph, window=window)

    assert len(repo.calls) == 1
    assert repo.calls[0]["seen_after"] is None
    assert repo.calls[0]["seen_before"] == _NOW
    assert repo.calls[0]["states"] == VOTING_STATES


@pytest.mark.asyncio
async def test_shadow_receives_same_cutoff() -> None:
    """The multiview-shadow comparison re-runs the query with a different path
    than the live evaluation, but must carry the identical capture-time window.
    """
    repo = _RecordingGalleryRepo()
    await _seed_alice(repo)
    proto = ViewPrototype(orientation=OrientationBin.FRONT, embedding=(1.0,) + (0.0,) * 7, count=5)
    config = ResolverConfig(
        gallery_vote_max_age_s=43200,
        enable_multiview_gallery=False,
        multiview_shadow_sample_rate=1.0,
    )
    resolver = IdentityResolver(gallery_repo=repo, config=config)
    ph = _FakePH(entity_id="ph-1", obs_ids=["obs-1"], view_prototypes=(proto,))

    await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=_NOW)

    assert len(repo.calls) == 3
    for call in repo.calls:
        assert call["seen_after"] == _NOW - timedelta(seconds=43200)
        assert call["seen_before"] == _NOW
        assert call["states"] == VOTING_STATES


@pytest.mark.asyncio
async def test_historical_rows_vote_at_capture_time_and_not_after_window() -> None:
    """Rows stamped in 2026-06 vote at a 2026-06 capture time, expire 13h later,
    and future rows stamped after captured_at never leak into an earlier replay frame."""
    repo = InMemoryGalleryRepository()
    await _seed_alice(repo)
    await repo.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="g-alice-future",
            identity_id=_ALICE_ID,
            embedding=[0.0, 1.0] + [0.0] * 6,
            seen_at=_NOW + timedelta(minutes=5),
            origin_tracklet_id="obs-future",
            state="operator_verified",
        )
    )
    proto_now = ViewPrototype(
        orientation=OrientationBin.FRONT, embedding=(1.0,) + (0.0,) * 7, count=5
    )
    proto_future = ViewPrototype(
        orientation=OrientationBin.FRONT, embedding=(0.0, 1.0) + (0.0,) * 6, count=5
    )
    config = ResolverConfig(gallery_vote_max_age_s=43200, enable_multiview_gallery=True)
    resolver = IdentityResolver(gallery_repo=repo, config=config)

    # At _NOW + 1h, g-alice votes (within 12h window).
    ph_in = _FakePH(entity_id="ph-in", obs_ids=["obs-x"], view_prototypes=(proto_now,))
    in_outcome = await resolver.resolve(
        hypotheses=[ph_in], new_face_anchors=[], captured_at=_NOW + timedelta(hours=1)
    )
    top_id, top_prob = in_outcome.decisions[0].posterior.top_identity()
    assert top_id == _ALICE_ID and top_prob > 0.5

    # At _NOW + 13h, g-alice is past the 12h window and cannot vote.
    ph_late = _FakePH(entity_id="ph-late", obs_ids=["obs-y"], view_prototypes=(proto_now,))
    late_outcome = await resolver.resolve(
        hypotheses=[ph_late], new_face_anchors=[], captured_at=_NOW + timedelta(hours=13)
    )
    assert late_outcome.decisions[0].identity_id is None

    # At _NOW, g-alice-future (stamped at _NOW + 5 min) must NOT vote.
    ph_future = _FakePH(entity_id="ph-future", obs_ids=["obs-z"], view_prototypes=(proto_future,))
    future_outcome = await resolver.resolve(
        hypotheses=[ph_future], new_face_anchors=[], captured_at=_NOW
    )
    assert future_outcome.decisions[0].identity_id is None
