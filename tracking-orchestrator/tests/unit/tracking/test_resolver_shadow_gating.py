"""Resolver shadow-query sample-rate gating (M05 / finding F6).

Covers the headline regression: at default config, ``resolve()`` must issue
zero shadow gallery queries. Shadow comparisons only fire when explicitly
sampled in via ``coherence_shadow_sample_rate`` / ``multiview_shadow_sample_rate``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from prometheus_client import CollectorRegistry, Counter

from app.domain import (
    GalleryEmbedding,
    Identity,
    IdentityDecision,
    OrientationBin,
    PosteriorDist,
    ViewPrototype,
)
from app.observability import metrics as metrics_pkg
from app.observability.metrics import Metrics, build_metrics
from app.storage.gallery import VERIFIED_ONLY, InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig


@pytest.fixture
def fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> Metrics:
    fresh = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh)
    return fresh


def _labeled_counter_value(counter: Counter, label_name: str, label_value: str) -> float:
    return sum(
        sample.value
        for metric in counter.collect()
        for sample in metric.samples
        if sample.name.endswith("_total") and sample.labels.get(label_name) == label_value
    )


class _CountingGalleryRepo(InMemoryGalleryRepository):
    """Counts gallery I/O calls so tests can assert on shadow-query frequency."""

    def __init__(self) -> None:
        super().__init__()
        self.search_similar_calls = 0
        self.list_for_tracklets_calls = 0

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
        states: frozenset[str] | None = VERIFIED_ONLY,
    ) -> list[tuple[GalleryEmbedding, float]]:
        self.search_similar_calls += 1
        return await super().search_similar(
            embedding=embedding,
            limit=limit,
            camera_id=camera_id,
            max_age_seconds=max_age_seconds,
            states=states,
        )

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
        allowed_states: frozenset[str] | None = VERIFIED_ONLY,
        model_versions: set[str] | None = None,
    ) -> list[GalleryEmbedding]:
        self.list_for_tracklets_calls += 1
        return await super().list_gallery_entries_for_tracklets(
            tracklet_ids=tracklet_ids,
            limit=limit,
            allowed_states=allowed_states,
            model_versions=model_versions,
        )

    @property
    def gallery_call_count(self) -> int:
        return self.search_similar_calls + self.list_for_tracklets_calls


def _normalize(vals: list[float]) -> list[float]:
    arr = np.asarray(vals, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr.tolist()


_FRONT_EMB = _normalize([1.0] + [0.0] * 767)
_BACK_EMB = _normalize([0.0] * 384 + [1.0] + [0.0] * 383)
_BOB_BACK_EMB = _normalize([0.0] * 767 + [1.0])
_ALICE_ID = "alice"
_BOB_ID = "bob"


class _FakePH:
    """Minimal IdentityResolvableEntity for resolver shadow-gating tests."""

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
async def gallery() -> _CountingGalleryRepo:
    """Gallery with alice front entries (single-query path) and back entries
    (multiview path), so both shadow comparisons have real work to do."""
    repo = _CountingGalleryRepo()
    await repo.upsert_identity(
        Identity(
            identity_id=_ALICE_ID,
            display_name="Alice",
            enrolled_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    await repo.upsert_identity(
        Identity(
            identity_id=_BOB_ID, display_name="Bob", enrolled_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
    )
    for i in range(5):
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"alice-front-{i}",
                identity_id=_ALICE_ID,
                embedding=_FRONT_EMB,
                seen_at=datetime(2026, 6, 1, tzinfo=UTC),
                quality=0.8,
                face_confirmed=True,
                orientation=OrientationBin.FRONT,
                state="operator_verified",
                origin_tracklet_id="obs-1",
            )
        )
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
                state="operator_verified",
            )
        )
    # Bob's back entries live at a third, mutually orthogonal direction so a
    # view prototype pointed at them out-votes alice in the multiview path
    # while the single-query (live) path, keyed to alice's front entries via
    # origin_tracklet_id, is unaffected — this is what makes the multiview
    # shadow's decision genuinely flip in the mismatch test below.
    for i in range(5):
        await repo.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"bob-back-{i}",
                identity_id=_BOB_ID,
                embedding=_BOB_BACK_EMB,
                seen_at=datetime(2026, 6, 1, tzinfo=UTC),
                quality=0.8,
                face_confirmed=True,
                orientation=OrientationBin.BACK,
                state="operator_verified",
            )
        )
    return repo


def _ph_no_prototypes() -> _FakePH:
    # Already committed to alice: the resolved identity will match, so
    # revises_previous is False and _has_cross_camera_reid_assist (a third,
    # unrelated gallery call gated on revision) never fires. Isolates the
    # gallery-call count to exactly what the shadow gate under test controls.
    return _FakePH(
        entity_id="ph-1",
        obs_ids=["obs-1"],
        camera_ids=["cam-1"],
        current_identity_id=_ALICE_ID,
        view_prototypes=(),
    )


def _ph_with_back_prototype() -> _FakePH:
    back_proto = ViewPrototype(orientation=OrientationBin.BACK, embedding=tuple(_BACK_EMB), count=5)
    return _FakePH(
        entity_id="ph-1",
        obs_ids=["obs-1"],
        camera_ids=["cam-1"],
        current_identity_id=_ALICE_ID,
        view_prototypes=(back_proto,),
    )


def _ph_with_mismatched_prototype() -> _FakePH:
    """Committed to alice, but the view prototype matches bob's back entries —
    the live (single-query) path still resolves alice; the multiview shadow
    resolves bob, so the two decisions genuinely disagree."""
    bob_proto = ViewPrototype(
        orientation=OrientationBin.BACK, embedding=tuple(_BOB_BACK_EMB), count=5
    )
    return _FakePH(
        entity_id="ph-1",
        obs_ids=["obs-1"],
        camera_ids=["cam-1"],
        current_identity_id=_ALICE_ID,
        view_prototypes=(bob_proto,),
    )


@pytest.mark.asyncio
async def test_default_config_issues_zero_shadow_gallery_queries(
    gallery: _CountingGalleryRepo,
    fresh_metrics: Metrics,
) -> None:
    """Headline regression (F6): default config runs the live gallery query
    exactly once per resolve() call, with no shadow query on top."""
    resolver = IdentityResolver(gallery_repo=gallery, config=ResolverConfig())
    ph = _ph_with_back_prototype()  # has view_prototypes -> multiview shadow eligible

    await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=ph.last_seen_at)

    # Single-query path: one list_gallery_entries_for_tracklets + one search_similar.
    assert gallery.list_for_tracklets_calls == 1
    assert gallery.search_similar_calls == 1


@pytest.mark.asyncio
async def test_coherence_shadow_runs_at_rate_one_and_counts_mismatch(
    gallery: _CountingGalleryRepo,
    fresh_metrics: Metrics,
) -> None:
    """At rate=1.0 the shadow query fires. Here the live and boosted decisions
    agree (only alice has matching entries), so the mismatch counter must stay
    at zero — proving the gate does not spuriously count agreement as a
    mismatch. The flip case (shadow disagrees with live) is covered by
    ``test_coherence_boost_shadow_counts_when_decision_would_change`` in
    ``test_identity_resolver_robustness.py``, which exercises the identical
    ``_shadow_gallery_compare`` -> ``_shadow_mismatch`` path this test uses."""
    config = ResolverConfig(
        enable_embedding_coherence_boost=False,
        coherence_shadow_sample_rate=1.0,
    )
    resolver = IdentityResolver(gallery_repo=gallery, config=config)
    ph = _ph_no_prototypes()

    await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=ph.last_seen_at)

    # Live query (1) + coherence shadow query (1).
    assert gallery.search_similar_calls == 2
    assert (
        _labeled_counter_value(
            fresh_metrics.identity_shadow_mismatch_total, "feature", "coherence_boost"
        )
        == 0.0
    )


@pytest.mark.asyncio
async def test_coherence_shadow_disabled_when_boost_already_enabled(
    gallery: _CountingGalleryRepo,
    fresh_metrics: Metrics,
) -> None:
    """No shadow regardless of rate when the boost is already live."""
    config = ResolverConfig(
        enable_embedding_coherence_boost=True,
        coherence_shadow_sample_rate=1.0,
    )
    resolver = IdentityResolver(gallery_repo=gallery, config=config)
    ph = _ph_no_prototypes()

    await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=ph.last_seen_at)

    assert gallery.search_similar_calls == 1


@pytest.mark.asyncio
async def test_multiview_shadow_runs_at_rate_one(
    gallery: _CountingGalleryRepo,
    fresh_metrics: Metrics,
) -> None:
    config = ResolverConfig(
        enable_multiview_gallery=False,
        multiview_shadow_sample_rate=1.0,
    )
    resolver = IdentityResolver(gallery_repo=gallery, config=config)
    ph = _ph_with_back_prototype()  # prototype matches alice's own back entries -> no flip

    await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=ph.last_seen_at)

    # Live single-query path (list_for_tracklets + search_similar) plus the
    # multiview shadow's one search_similar (single qualifying prototype).
    assert gallery.list_for_tracklets_calls == 1
    assert gallery.search_similar_calls == 2
    assert (
        _labeled_counter_value(
            fresh_metrics.identity_shadow_mismatch_total, "feature", "multiview_gallery"
        )
        == 0.0
    )


@pytest.mark.asyncio
async def test_multiview_shadow_counts_mismatch_when_decision_would_change(
    gallery: _CountingGalleryRepo,
    fresh_metrics: Metrics,
) -> None:
    """The shadow's decision can genuinely disagree with live: here the
    prototype matches bob's entries while the live single-query path (keyed to
    the PH's own alice-front observations) still resolves alice."""
    config = ResolverConfig(
        enable_multiview_gallery=False,
        multiview_shadow_sample_rate=1.0,
    )
    resolver = IdentityResolver(gallery_repo=gallery, config=config)
    ph = _ph_with_mismatched_prototype()

    outcome = await resolver.resolve(
        hypotheses=[ph], new_face_anchors=[], captured_at=ph.last_seen_at
    )

    assert outcome.decisions[0].identity_id == _ALICE_ID  # live path unaffected
    assert (
        _labeled_counter_value(
            fresh_metrics.identity_shadow_mismatch_total, "feature", "multiview_gallery"
        )
        == 1.0
    )


@pytest.mark.asyncio
async def test_multiview_shadow_disabled_when_flag_already_enabled(
    gallery: _CountingGalleryRepo,
    fresh_metrics: Metrics,
) -> None:
    """No multiview shadow regardless of rate when multiview is already live."""
    config = ResolverConfig(
        enable_multiview_gallery=True,
        multiview_shadow_sample_rate=1.0,
    )
    resolver = IdentityResolver(gallery_repo=gallery, config=config)
    ph = _ph_with_back_prototype()

    await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=ph.last_seen_at)

    # Multiview live path issues one search_similar per qualifying prototype;
    # no list_for_tracklets call (multiview path skips the single-query fetch)
    # and no extra shadow query.
    assert gallery.list_for_tracklets_calls == 0
    assert gallery.search_similar_calls == 1


@pytest.mark.asyncio
async def test_shadow_gate_determinism(
    gallery: _CountingGalleryRepo,
    fresh_metrics: Metrics,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """random.random() >= rate must never sample in; < rate must always sample in."""
    config = ResolverConfig(
        enable_embedding_coherence_boost=False,
        coherence_shadow_sample_rate=0.5,
    )

    monkeypatch.setattr("app.tracking.identity_resolver.random.random", lambda: 0.99)
    resolver = IdentityResolver(gallery_repo=gallery, config=config)
    ph = _ph_no_prototypes()
    await resolver.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=ph.last_seen_at)
    assert gallery.search_similar_calls == 1  # live query only, no shadow

    gallery.search_similar_calls = 0
    monkeypatch.setattr("app.tracking.identity_resolver.random.random", lambda: 0.01)
    resolver2 = IdentityResolver(gallery_repo=gallery, config=config)
    await resolver2.resolve(hypotheses=[ph], new_face_anchors=[], captured_at=ph.last_seen_at)
    assert gallery.search_similar_calls == 2  # live + shadow


def test_commit_source_matches_metric_label_for_face_and_reid_and_prior() -> None:
    """decision_source and the identity_commits_total label share one
    computation (dedup of the former commit_src/commit_source split)."""
    resolver = IdentityResolver(gallery_repo=InMemoryGalleryRepository(), config=ResolverConfig())
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    ph = _FakePH(entity_id="ph-1", obs_ids=["obs-1"], camera_ids=["cam-1"])

    face_decision = resolver._commit(
        ph,
        PosteriorDist({"alice": 0.90, "UNKNOWN": 0.10}),
        PosteriorDist({"alice": 0.90}),
        PosteriorDist({}),
        now,
        evidence_identity_ids=frozenset({"alice"}),
    )
    assert isinstance(face_decision, IdentityDecision)
    assert face_decision.decision_source == "face"

    reid_decision = resolver._commit(
        ph,
        PosteriorDist({"alice": 0.90, "UNKNOWN": 0.10}),
        PosteriorDist({}),
        PosteriorDist({"alice": 0.90}),
        now,
        evidence_identity_ids=frozenset({"alice"}),
    )
    assert reid_decision.decision_source == "reid"
