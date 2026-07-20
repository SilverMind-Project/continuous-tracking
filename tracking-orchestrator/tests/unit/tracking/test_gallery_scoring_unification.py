"""Structural and behavioral guards proving one scorer serves every gallery path.

Identity-continuity M01 (findings V1/V2): before this milestone,
``_from_gallery_multiview`` scored hits inline and skipped the verified
trust multiplier, recency decay, and vote caps that ``_score_gallery_hits``
applied on the single-query fallback path. These tests pin that both paths
now route through the single shared ``gallery_scoring.score_hits`` and
produce identical per-hit scores for identical raw input, so a future
regression that reintroduces inline scoring on either path fails loudly.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from app.domain import GalleryEmbedding, OrientationBin, ViewPrototype
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking import identity_resolver as identity_resolver_module
from app.tracking.identity.gallery_scoring import ScoredHit
from app.tracking.identity.gallery_scoring import score_hits as _real_score_hits
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

_NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)


class _FakePH:
    """Minimal IdentityResolvableEntity, mirroring test_identity_resolver_multiview.py."""

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


def test_score_gallery_hits_calls_shared_scorer() -> None:
    source = inspect.getsource(IdentityResolver._score_gallery_hits)
    assert "score_hits(" in source, (
        "_score_gallery_hits must call the shared gallery_scoring.score_hits; "
        "inline scoring in the resolver is a defect (identity-continuity M01)"
    )


def test_from_gallery_multiview_calls_shared_scorer() -> None:
    source = inspect.getsource(IdentityResolver._from_gallery_multiview)
    assert "score_hits(" in source, (
        "_from_gallery_multiview must call the shared gallery_scoring.score_hits; "
        "inline scoring in the resolver is a defect (identity-continuity M01)"
    )


@pytest.mark.asyncio
async def test_multiview_and_fallback_paths_score_identical_hits_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feed the same raw (entry, similarity) hits into both paths; per-hit
    scoring (logit, trust, recency, weighted_logit, boosted) must match.
    """
    hits: list[tuple[GalleryEmbedding, float]] = [
        (
            GalleryEmbedding(
                gallery_entry_id="fresh-verified",
                identity_id="alice",
                embedding=[1.0] * 8,
                seen_at=_NOW,
                state="operator_verified",
                source_episode_id="ep1",
                camera_id="cam-1",
                orientation=OrientationBin.FRONT,
            ),
            0.60,
        ),
        (
            GalleryEmbedding(
                gallery_entry_id="stale-pending",
                identity_id="bob",
                embedding=[0.5] * 8,
                seen_at=_NOW.replace(day=1),  # ~19 days old
                state="pending_review",
                source_episode_id="ep2",
                camera_id="cam-1",
                orientation=OrientationBin.BACK,
            ),
            0.55,
        ),
    ]

    repo = InMemoryGalleryRepository()

    async def _fake_search_similar(
        *,
        embedding: list[float],
        limit: int = 20,
        max_age_seconds: int | None = None,
        states: frozenset[str] | None = None,
    ) -> list[tuple[GalleryEmbedding, float]]:
        return hits

    async def _fake_list_for_tracklets(
        *, tracklet_ids: set[str], limit: int = 20
    ) -> list[GalleryEmbedding]:
        return [hits[0][0]]

    monkeypatch.setattr(repo, "search_similar", _fake_search_similar)
    monkeypatch.setattr(repo, "list_gallery_entries_for_tracklets", _fake_list_for_tracklets)

    resolver = IdentityResolver(gallery_repo=repo, config=ResolverConfig())

    captured: list[list[ScoredHit]] = []

    def _spy(*args: object, **kwargs: object) -> list[ScoredHit]:
        result = _real_score_hits(*args, **kwargs)  # type: ignore[arg-type]
        captured.append(result)
        return result

    monkeypatch.setattr(identity_resolver_module, "score_hits", _spy)

    ph_fallback = _FakePH(entity_id="ph-fallback", obs_ids=["obs-1"])
    await resolver._from_gallery(ph_fallback, enable_multiview=False)

    back_proto = ViewPrototype(
        orientation=OrientationBin.BACK,
        embedding=tuple([0.5] * 8),
        count=5,
    )
    ph_multiview = _FakePH(
        entity_id="ph-multiview", obs_ids=["obs-2"], view_prototypes=(back_proto,)
    )
    await resolver._from_gallery(ph_multiview, enable_multiview=True)

    assert len(captured) == 2
    fallback_scored, multiview_scored = captured

    by_id_fallback = {hit.entry.gallery_entry_id: hit for hit in fallback_scored}
    by_id_multiview = {hit.entry.gallery_entry_id: hit for hit in multiview_scored}
    assert by_id_fallback.keys() == by_id_multiview.keys()

    # Approx (not ==) on the floats: each path computes `now` independently
    # via its own `datetime.now(UTC)` call, so recency_factor can differ by a
    # sub-microsecond amount between the two invocations. That noise must
    # not mask a real divergence in the scoring logic itself.
    for entry_id, fb_hit in by_id_fallback.items():
        mv_hit = by_id_multiview[entry_id]
        assert fb_hit.boosted == mv_hit.boosted
        assert fb_hit.logit == pytest.approx(mv_hit.logit)
        assert fb_hit.trust_multiplier == pytest.approx(mv_hit.trust_multiplier)
        assert fb_hit.recency_factor == pytest.approx(mv_hit.recency_factor)
        assert fb_hit.weighted_logit == pytest.approx(mv_hit.weighted_logit)
