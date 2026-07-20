"""Orientation-aware resolver gallery query tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.domain import (
    GalleryEmbedding,
    Identity,
    OrientationBin,
    ViewPrototype,
)
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking import identity_resolver as identity_resolver_module
from app.tracking.identity.gallery_scoring import ScoredHit
from app.tracking.identity.gallery_scoring import score_hits as _real_score_hits
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
        self.last_independent_identity_evidence_at = None
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
    """Gallery where alice has back-facing entries and bob has front-facing entries.

    ``seen_at`` must be relative to real wall-clock time (``datetime.now(UTC)``),
    not a hardcoded calendar date: M01 makes the multiview path apply the same
    recency decay as the fallback path, so a fixed past date silently drifts
    stale as the real clock advances and eventually crushes alice's vote
    weight toward zero (found while implementing M01; the pre-M01 multiview
    path ignored ``seen_at`` entirely, so this never surfaced before).
    """
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
    # Add alice back entries. state="operator_verified" so search_similar's
    # verified-only default (M03) can see them.
    for i in range(5):
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"alice-back-{i}",
                identity_id=_ALICE_ID,
                embedding=_BACK_EMB,
                seen_at=datetime.now(UTC),
                quality=0.8,
                face_confirmed=True,
                orientation=OrientationBin.BACK,
                state="operator_verified",
            )
        )
    # Add bob front entries.
    for i in range(5):
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"bob-front-{i}",
                identity_id=_BOB_ID,
                embedding=_FRONT_EMB,
                seen_at=datetime.now(UTC),
                quality=0.8,
                face_confirmed=True,
                orientation=OrientationBin.FRONT,
                state="operator_verified",
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


@pytest.mark.asyncio
async def test_multiview_applies_trust_and_recency() -> None:
    """V1 regression: recency decay must reduce an old verified hit's vote
    weight on the multiview path.

    Fails on pre-M01 code, which scored the multiview path inline with no
    trust or recency at all: fresh_prob would equal old_prob exactly there,
    since entry age never entered the computation.
    """

    async def _top_prob_for_age(age_days: int) -> float:
        now_ref = datetime.now(UTC)
        repo = InMemoryGalleryRepository()
        await repo.upsert_identity(
            Identity(
                identity_id=_ALICE_ID,
                display_name="Alice",
                enrolled_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="alice-back-0",
                identity_id=_ALICE_ID,
                embedding=_BACK_EMB,
                seen_at=now_ref - timedelta(days=age_days),
                quality=0.8,
                face_confirmed=True,
                orientation=OrientationBin.BACK,
                state="operator_verified",
            )
        )
        config = ResolverConfig(commit_prob=0.65, enable_multiview_gallery=True)
        resolver = IdentityResolver(gallery_repo=repo, config=config)
        back_proto = ViewPrototype(
            orientation=OrientationBin.BACK, embedding=tuple(_BACK_EMB), count=5
        )
        ph = _FakePH(
            entity_id="ph-1",
            obs_ids=["obs-1"],
            camera_ids=["cam-1"],
            view_prototypes=(back_proto,),
        )
        outcome = await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=now_ref)
        _, top_prob = outcome.decisions[0].posterior.top_identity()
        return top_prob

    fresh_prob = await _top_prob_for_age(0)
    old_prob = await _top_prob_for_age(14)  # two half-lives at the default 7-day half-life

    assert fresh_prob > old_prob


@pytest.mark.asyncio
async def test_auto_verified_votes_at_1_5_no_backstop() -> None:
    """Identity-continuity M02: an auto_verified gallery row votes at the
    1.5x trust multiplier (below operator_verified's 2.0x, above a
    non-voting row's 1.0x) and never trips the non-voting-state backstop
    counter. M01 built the scorer to expect this state; this is the first
    test where an auto_verified row actually exists end to end through a
    real resolver call, not just a pure gallery_scoring.py unit test.

    Uses a partial-similarity (~0.65) gallery embedding rather than an exact
    match: an exact match drives the logistic curve so close to 1.0 that
    both trust multipliers saturate the posterior at the same normalized
    1.0, masking the very difference this test exists to prove.
    """
    now_ref = datetime.now(UTC)
    # cos-similarity to _BACK_EMB is ~0.65: below the logistic midpoint
    # (reid_decision_sim=0.70) so neither trust multiplier saturates the
    # weighted logit past 1.0, leaving the UNKNOWN complement (and therefore
    # the normalized top probability) sensitive to the trust difference.
    partial_back = np.zeros(768, dtype=np.float32)
    partial_back[384] = 0.65
    partial_back[0] = (1.0 - 0.65**2) ** 0.5
    partial_back_emb = _normalize(partial_back.tolist())
    metrics_singleton = identity_resolver_module.metrics.metrics

    async def _top_prob_for_state(state: str) -> float:
        repo = InMemoryGalleryRepository()
        await repo.upsert_identity(
            Identity(
                identity_id=_ALICE_ID,
                display_name="Alice",
                enrolled_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"alice-back-{state}",
                identity_id=_ALICE_ID,
                embedding=partial_back_emb,
                seen_at=now_ref,
                quality=0.8,
                face_confirmed=True,
                orientation=OrientationBin.BACK,
                state=state,
            )
        )
        config = ResolverConfig(commit_prob=0.65, enable_multiview_gallery=True)
        resolver = IdentityResolver(gallery_repo=repo, config=config)
        before = metrics_singleton.reid_rejected_vector_vote_attempts_total._value.get()
        back_proto = ViewPrototype(
            orientation=OrientationBin.BACK, embedding=tuple(_BACK_EMB), count=5
        )
        ph = _FakePH(
            entity_id="ph-1",
            obs_ids=["obs-1"],
            camera_ids=["cam-1"],
            view_prototypes=(back_proto,),
        )
        outcome = await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=now_ref)
        after = metrics_singleton.reid_rejected_vector_vote_attempts_total._value.get()
        assert after == before  # backstop never fires for a voting state
        _, top_prob = outcome.decisions[0].posterior.top_identity()
        return top_prob

    auto_verified_prob = await _top_prob_for_state("auto_verified")
    operator_verified_prob = await _top_prob_for_state("operator_verified")

    assert 0.0 < auto_verified_prob < operator_verified_prob


@pytest.mark.asyncio
async def test_multiview_vote_caps_same_episode() -> None:
    """Ten near-duplicate crops from one episode must vote once, not ten times."""
    now = datetime.now(UTC)

    def _make_entry(entry_id: str) -> GalleryEmbedding:
        return GalleryEmbedding(
            gallery_entry_id=entry_id,
            identity_id=_ALICE_ID,
            embedding=_BACK_EMB,
            seen_at=now,
            quality=0.8,
            face_confirmed=True,
            orientation=OrientationBin.BACK,
            state="operator_verified",
            source_episode_id="ep-shared",
            camera_id="cam-1",
        )

    config = ResolverConfig(commit_prob=0.65, enable_multiview_gallery=True)
    back_proto = ViewPrototype(orientation=OrientationBin.BACK, embedding=tuple(_BACK_EMB), count=5)
    ph = _FakePH(
        entity_id="ph-1", obs_ids=["obs-1"], camera_ids=["cam-1"], view_prototypes=(back_proto,)
    )

    ten_dupes_repo = InMemoryGalleryRepository()
    await ten_dupes_repo.upsert_identity(
        Identity(
            identity_id=_ALICE_ID,
            display_name="Alice",
            enrolled_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    for i in range(10):
        await ten_dupes_repo.upsert_gallery_entry(_make_entry(f"alice-back-dup-{i}"))
    ten_dupes_resolver = IdentityResolver(gallery_repo=ten_dupes_repo, config=config)
    ten_dupes_posterior = await ten_dupes_resolver._from_gallery_multiview(ph, (back_proto,))

    single_repo = InMemoryGalleryRepository()
    await single_repo.upsert_identity(
        Identity(
            identity_id=_ALICE_ID,
            display_name="Alice",
            enrolled_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    await single_repo.upsert_gallery_entry(_make_entry("alice-back-single"))
    single_resolver = IdentityResolver(gallery_repo=single_repo, config=config)
    single_posterior = await single_resolver._from_gallery_multiview(ph, (back_proto,))

    # All ten duplicates land in the same (identity, episode, camera,
    # orientation) group, so the capped result must equal a single vote --
    # not be amplified by count.
    assert ten_dupes_posterior.distribution[_ALICE_ID] == pytest.approx(
        single_posterior.distribution[_ALICE_ID]
    )


@pytest.mark.asyncio
async def test_backstop_counter_increments_on_multiview_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-voting-state row that reaches the multiview scorer (e.g. via a
    future query-filter regression) must increment the observability
    backstop counter, exactly like it already does on the fallback path.
    """
    from app.observability.metrics import metrics as global_metrics

    repo = InMemoryGalleryRepository()
    pending_entry = GalleryEmbedding(
        gallery_entry_id="alice-pending",
        identity_id=_ALICE_ID,
        embedding=_BACK_EMB,
        seen_at=datetime.now(UTC),
        state="pending_review",
        orientation=OrientationBin.BACK,
    )

    async def _fake_search_similar(
        *,
        embedding: list[float],
        limit: int = 20,
        max_age_seconds: int | None = None,
        states: frozenset[str] | None = None,
    ) -> list[tuple[GalleryEmbedding, float]]:
        return [(pending_entry, 0.9)]

    monkeypatch.setattr(repo, "search_similar", _fake_search_similar)

    config = ResolverConfig(commit_prob=0.65, enable_multiview_gallery=True)
    resolver = IdentityResolver(gallery_repo=repo, config=config)
    back_proto = ViewPrototype(orientation=OrientationBin.BACK, embedding=tuple(_BACK_EMB), count=5)
    ph = _FakePH(
        entity_id="ph-1", obs_ids=["obs-1"], camera_ids=["cam-1"], view_prototypes=(back_proto,)
    )

    counter = global_metrics.reid_rejected_vector_vote_attempts_total
    before = counter._value.get()

    await resolver._from_gallery_multiview(ph, (back_proto,))

    assert counter._value.get() == before + 1


@pytest.mark.asyncio
async def test_shadow_path_uses_unified_scorer(
    monkeypatch: pytest.MonkeyPatch,
    gallery_with_alice_back: InMemoryGalleryRepository,
) -> None:
    """The multiview shadow comparison (live config off, sampled at 1.0)
    must route through the same shared scorer as the live paths, not a
    third inline implementation.
    """
    config = ResolverConfig(
        commit_prob=0.65,
        enable_multiview_gallery=False,
        multiview_shadow_sample_rate=1.0,
    )
    resolver = IdentityResolver(gallery_repo=gallery_with_alice_back, config=config)

    back_proto = ViewPrototype(orientation=OrientationBin.BACK, embedding=tuple(_BACK_EMB), count=5)
    ph = _FakePH(
        entity_id="ph-1", obs_ids=["obs-1"], camera_ids=["cam-1"], view_prototypes=(back_proto,)
    )

    calls: list[list[ScoredHit]] = []

    def _spy(*args: object, **kwargs: object) -> list[ScoredHit]:
        result = _real_score_hits(*args, **kwargs)  # type: ignore[arg-type]
        calls.append(result)
        return result

    monkeypatch.setattr(identity_resolver_module, "score_hits", _spy)

    outcome = await resolver.resolve(
        hypotheses=[ph],
        new_face_anchors=[],
        captured_at=datetime(2026, 6, 1, 12, 0, 1, tzinfo=UTC),
    )

    assert len(outcome.decisions) == 1
    # The shadow comparison (multiview, forced on by sample_rate=1.0) must
    # have exercised the shared scorer at least once.
    assert len(calls) >= 1
