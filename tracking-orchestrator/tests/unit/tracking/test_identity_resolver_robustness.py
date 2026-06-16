"""M3 identity resolver robustness guards."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry, Counter

from app.domain import (
    FaceAnchor,
    GalleryEmbedding,
    GlobalTrack,
    Identity,
    IdentityDecision,
    PosteriorDist,
)
from app.observability import metrics as metrics_pkg
from app.observability.metrics import Metrics, build_metrics
from app.storage.base import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig


@pytest.fixture
def fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> Metrics:
    fresh = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh)
    return fresh


def _counter_value(samples: Iterator[object]) -> float:
    return sum(
        sample.value
        for metric in samples
        for sample in metric.samples
        if sample.name.endswith("_total")
    )


def _metric_total(counter: Counter) -> float:
    return _counter_value(iter(counter.collect()))


def _make_gt(
    *,
    ph_id: str = "ph-1",
    current_identity_id: str | None = None,
    committed_at: datetime | None = None,
    tracklet_ids: list[str] | None = None,
    camera_ids: list[str] | None = None,
) -> GlobalTrack:
    now = datetime.now(UTC)
    return GlobalTrack(
        global_track_id=ph_id,
        camera_ids=camera_ids or ["cam-a"],
        tracklet_ids=tracklet_ids or ["t1"],
        started_at=now,
        last_seen_at=now,
        current_identity_id=current_identity_id,
        current_identity_committed_at=committed_at,
        state="active",
    )


def _anchor(person_id: str, *, confidence: float = 0.95, quality: float = 0.9) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        confidence=confidence,
        quality=quality,
        tracklet_id="t1",
    )


async def _duplicate_identity_resolver(
    *,
    enable_guard: bool,
) -> IdentityResolver:
    gallery = InMemoryGalleryRepository()
    now = datetime.now(UTC)
    for identity_id in ("amma", "grandma"):
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=now)
        )
    for tracklet_id in ("t-held", "t-new"):
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"{tracklet_id}-amma",
                identity_id="amma",
                embedding=[1.0, 0.0],
                seen_at=now,
                origin_tracklet_id=tracklet_id,
                face_confirmed=True,
            )
        )
    return IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(
            commit_prob=0.50,
            commit_prob_dense=0.50,
            commit_margin_dense=0.20,
            prior_weight=0.30,
            enable_duplicate_active_identity_guard=enable_guard,
        ),
    )


async def _resolver(config: ResolverConfig) -> IdentityResolver:
    gallery = InMemoryGalleryRepository()
    identity = Identity(
        identity_id="alice",
        display_name="Alice",
        enrolled_at=datetime.now(UTC),
    )
    await gallery.upsert_identity(identity)
    return IdentityResolver(gallery_repo=gallery, identities=[identity], config=config)


class _ControlledGallery(InMemoryGalleryRepository):
    def __init__(self, matches: list[tuple[str, float] | tuple[str, float, str]]) -> None:
        super().__init__()
        self._matches = matches

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
    ) -> list[tuple[GalleryEmbedding, float]]:
        del embedding, limit, camera_id, max_age_seconds
        now = datetime.now(UTC)
        return [
            (
                GalleryEmbedding(
                    gallery_entry_id=f"{identity_id}-match",
                    identity_id=identity_id,
                    embedding=[1.0, 0.0],
                    seen_at=now,
                    quality=0.9,
                    origin_tracklet_id=f"{identity_id}-old",
                    face_confirmed=True,
                    camera_id=camera_id,
                ),
                score,
            )
            for identity_id, score, camera_id in (
                match if len(match) == 3 else (match[0], match[1], "") for match in self._matches
            )
        ]


async def _coherence_resolver(
    config: ResolverConfig,
    matches: list[tuple[str, float]],
    *,
    coherent: bool = True,
) -> IdentityResolver:
    gallery = _ControlledGallery(matches)
    now = datetime.now(UTC)
    for identity_id in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=now)
        )
    second_embedding = [1.0, 0.0] if coherent else [0.0, 1.0]
    for idx, embedding in enumerate(([1.0, 0.0], second_embedding)):
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id=f"query-{idx}",
                identity_id="",
                embedding=embedding,
                seen_at=now + timedelta(milliseconds=idx),
                quality=0.9,
                origin_tracklet_id="t1",
                face_confirmed=False,
            )
        )
    return IdentityResolver(gallery_repo=gallery, config=config)


def _labeled_counter_value(counter: Counter, label_name: str, label_value: str) -> float:
    return sum(
        sample.value
        for metric in counter.collect()
        for sample in metric.samples
        if sample.name.endswith("_total") and sample.labels.get(label_name) == label_value
    )


@pytest.mark.asyncio
async def test_quality_gate_blocks_low_quality_commit(fresh_metrics: Metrics) -> None:
    config = ResolverConfig(enable_quality_gate=True)
    resolver = await _resolver(config)

    low_quality = await resolver.resolve(
        hypotheses=[_make_gt()],
        new_face_anchors=[_anchor("alice")],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.20},
    )

    assert low_quality.decisions[0].identity_id is None
    assert _metric_total(fresh_metrics.identity_quality_gate_blocks_total) == 1.0

    high_quality = await resolver.resolve(
        hypotheses=[_make_gt()],
        new_face_anchors=[_anchor("alice")],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.50},
    )

    assert high_quality.decisions[0].identity_id == "alice"
    assert _metric_total(fresh_metrics.identity_quality_gate_blocks_total) == 1.0


@pytest.mark.asyncio
async def test_quality_gate_does_not_block_maintenance(fresh_metrics: Metrics) -> None:
    resolver = await _resolver(ResolverConfig(enable_quality_gate=True))
    committed_at = datetime.now(UTC) - timedelta(seconds=5)

    outcome = await resolver.resolve(
        hypotheses=[
            _make_gt(current_identity_id="alice", committed_at=committed_at),
        ],
        new_face_anchors=[],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.10},
    )

    assert outcome.decisions[0].identity_id == "alice"
    assert outcome.decisions[0].revises_previous is False
    assert _metric_total(fresh_metrics.identity_quality_gate_blocks_total) == 0.0


@pytest.mark.asyncio
async def test_quality_gate_face_lock_threshold(fresh_metrics: Metrics) -> None:
    config = ResolverConfig(
        enable_quality_gate=True,
        min_quality_to_commit=0.35,
        min_quality_to_face_lock=0.45,
        face_commit_min_confidence=0.70,
    )
    resolver = await _resolver(config)

    outcome = await resolver.resolve(
        hypotheses=[_make_gt()],
        new_face_anchors=[_anchor("alice")],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.40},
    )

    assert outcome.decisions[0].identity_id == "alice"
    assert resolver.get_face_locked_identity("ph-1") is None
    assert _metric_total(fresh_metrics.identity_quality_gate_blocks_total) == 1.0


@pytest.mark.asyncio
async def test_quality_gate_shadow_counts_without_blocking(fresh_metrics: Metrics) -> None:
    resolver = await _resolver(ResolverConfig(enable_quality_gate=False))

    outcome = await resolver.resolve(
        hypotheses=[_make_gt()],
        new_face_anchors=[_anchor("alice")],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.20},
    )

    assert outcome.decisions[0].identity_id == "alice"
    assert _metric_total(fresh_metrics.identity_quality_gate_blocks_total) == 1.0


def test_flip_debounce_blocks_quick_reversal(fresh_metrics: Metrics) -> None:
    now = datetime.now(UTC)
    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        config=ResolverConfig(enable_flip_debounce=True),
    )
    entity = _make_gt(current_identity_id="alice", committed_at=now - timedelta(seconds=1))
    posterior = PosteriorDist({"bob": 0.70, "alice": 0.20, "UNKNOWN": 0.10})
    face = PosteriorDist({"bob": 1.0})

    decision = resolver._commit(
        entity,
        posterior,
        face,
        PosteriorDist({}),
        now,
    )

    assert decision.identity_id == "alice"
    assert decision.revises_previous is False
    assert _metric_total(fresh_metrics.identity_flips_total) == 0.0

    after_window = _make_gt(
        current_identity_id="alice",
        committed_at=now - timedelta(seconds=11),
    )
    allowed = resolver._commit(
        after_window,
        posterior,
        face,
        PosteriorDist({}),
        now,
    )

    assert allowed.identity_id == "bob"
    assert allowed.revises_previous is True
    assert _metric_total(fresh_metrics.identity_flips_total) == 1.0


def test_flip_debounce_allows_first_commit(fresh_metrics: Metrics) -> None:
    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        config=ResolverConfig(enable_flip_debounce=True),
    )

    decision = resolver._commit(
        _make_gt(),
        PosteriorDist({"alice": 0.70, "UNKNOWN": 0.30}),
        PosteriorDist({"alice": 1.0}),
        PosteriorDist({}),
        datetime.now(UTC),
    )

    assert isinstance(decision, IdentityDecision)
    assert decision.identity_id == "alice"
    assert _metric_total(fresh_metrics.identity_flips_total) == 0.0


@pytest.mark.asyncio
async def test_coherence_boost_shadow_counts_when_decision_would_change(
    fresh_metrics: Metrics,
) -> None:
    resolver = await _coherence_resolver(
        ResolverConfig(
            enable_embedding_coherence_boost=False,
            identified_entry_boost_min_sim=0.99,
            commit_prob=0.64,
            commit_prob_dense=0.64,
        ),
        matches=[("alice", 0.70), ("bob", 0.69)],
    )

    outcome = await resolver.resolve(
        hypotheses=[_make_gt()],
        new_face_anchors=[],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.9},
    )

    assert outcome.decisions[0].identity_id is None
    assert (
        _labeled_counter_value(
            fresh_metrics.identity_shadow_mismatch_total,
            "feature",
            "coherence_boost",
        )
        == 1.0
    )


@pytest.mark.asyncio
async def test_coherence_boost_holds_identity(fresh_metrics: Metrics) -> None:
    resolver = await _coherence_resolver(
        ResolverConfig(
            enable_embedding_coherence_boost=True,
            identified_entry_boost_min_sim=0.99,
            commit_prob=0.64,
            commit_prob_dense=0.64,
        ),
        matches=[("alice", 0.70), ("bob", 0.69)],
    )

    outcome = await resolver.resolve(
        hypotheses=[_make_gt()],
        new_face_anchors=[],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.9},
    )

    assert outcome.decisions[0].identity_id == "alice"
    assert _metric_total(fresh_metrics.identity_flips_total) == 0.0


@pytest.mark.asyncio
async def test_coherence_boost_no_miscommit_without_stable_appearance() -> None:
    resolver = await _coherence_resolver(
        ResolverConfig(
            enable_embedding_coherence_boost=True,
            identified_entry_boost_min_sim=0.99,
            commit_prob=0.64,
            commit_prob_dense=0.64,
        ),
        matches=[("alice", 0.70), ("bob", 0.69)],
        coherent=False,
    )

    outcome = await resolver.resolve(
        hypotheses=[_make_gt()],
        new_face_anchors=[],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.9},
    )

    assert outcome.decisions[0].identity_id is None


async def _propagation_resolver(
    similarity: float, *, enable_guard: bool = False
) -> IdentityResolver:
    gallery = InMemoryGalleryRepository()
    now = datetime.now(UTC)
    for identity_id in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=now)
        )
    await gallery.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="src",
            identity_id="alice",
            embedding=[1.0, 0.0],
            seen_at=now,
            origin_tracklet_id="t-src",
            camera_id="cam-a",
        )
    )
    await gallery.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id="dst",
            identity_id="",
            embedding=[similarity, (1.0 - similarity**2) ** 0.5],
            seen_at=now,
            origin_tracklet_id="t-dst",
            camera_id="cam-b",
        )
    )
    return IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(
            cross_gt_face_propagation_threshold=0.72,
            face_commit_min_confidence=0.70,
            enable_quality_gate=True,
            enable_duplicate_active_identity_guard=enable_guard,
        ),
    )


@pytest.mark.asyncio
async def test_propagation_threshold_respected(fresh_metrics: Metrics) -> None:
    src = _make_gt(ph_id="src", tracklet_ids=["t-src"], camera_ids=["cam-a"])
    dst = _make_gt(ph_id="dst", tracklet_ids=["t-dst"], camera_ids=["cam-b"])
    anchor = FaceAnchor(person_id="alice", confidence=1.0, quality=1.0, tracklet_id="t-src")

    below = await _propagation_resolver(0.71)
    below_anchors = await below._propagate_face_anchors([src, dst], [anchor])
    assert len(below_anchors) == 1
    assert _metric_total(fresh_metrics.face_propagations_total) == 0.0

    above = await _propagation_resolver(0.73)
    above_anchors = await above._propagate_face_anchors([src, dst], [anchor])
    assert len(above_anchors) == 2
    assert _metric_total(fresh_metrics.face_propagations_total) == 1.0


@pytest.mark.asyncio
async def test_propagation_respects_quality_gate(fresh_metrics: Metrics) -> None:
    resolver = await _propagation_resolver(0.73)
    src = _make_gt(ph_id="src", tracklet_ids=["t-src"], camera_ids=["cam-a"])
    dst = _make_gt(ph_id="dst", tracklet_ids=["t-dst"], camera_ids=["cam-b"])

    outcome = await resolver.resolve(
        hypotheses=[src, dst],
        new_face_anchors=[
            FaceAnchor(person_id="alice", confidence=1.0, quality=1.0, tracklet_id="t-src")
        ],
        captured_at=datetime.now(UTC),
        ph_qualities={"src": 0.9, "dst": 0.1},
    )

    decisions = {decision.ph_id: decision for decision in outcome.decisions}
    assert decisions["src"].identity_id == "alice"
    assert decisions["dst"].identity_id is None
    assert _metric_total(fresh_metrics.identity_quality_gate_blocks_total) >= 1.0


@pytest.mark.asyncio
async def test_propagation_negative_relatives(fresh_metrics: Metrics) -> None:
    resolver = await _propagation_resolver(0.68)
    src = _make_gt(ph_id="src", tracklet_ids=["t-src"], camera_ids=["cam-a"])
    dst = _make_gt(ph_id="dst", tracklet_ids=["t-dst"], camera_ids=["cam-b"])

    anchors = await resolver._propagate_face_anchors(
        [src, dst],
        [FaceAnchor(person_id="alice", confidence=1.0, quality=1.0, tracklet_id="t-src")],
    )

    assert len(anchors) == 1
    assert _metric_total(fresh_metrics.face_propagations_total) == 0.0


@pytest.mark.asyncio
async def test_reid_cross_camera_assist_metric(fresh_metrics: Metrics) -> None:
    resolver = await _coherence_resolver(
        ResolverConfig(
            identified_entry_boost_min_sim=0.99,
            commit_prob=0.64,
            commit_prob_dense=0.64,
        ),
        matches=[("alice", 0.82, "cam-a")],
    )

    outcome = await resolver.resolve(
        hypotheses=[_make_gt(tracklet_ids=["t1"], camera_ids=["cam-b"])],
        new_face_anchors=[],
        captured_at=datetime.now(UTC),
        ph_qualities={"ph-1": 0.9},
    )

    assert outcome.decisions[0].identity_id == "alice"
    assert _metric_total(fresh_metrics.reid_cross_camera_assist_total) == 1.0


# ---------------------------------------------------------------------------
# Sticky maintenance tests
# ---------------------------------------------------------------------------


def test_sticky_maintenance_holds_identity_when_posterior_is_unknown(
    fresh_metrics: Metrics,
) -> None:
    """With sticky maintenance enabled, a PH with committed identity within
    the maintenance window keeps its identity even when the posterior argmax
    is UNKNOWN (no face, weak ReID)."""
    now = datetime.now(UTC)
    gt = _make_gt(
        current_identity_id="alice",
        committed_at=now - timedelta(seconds=5),
    )
    # Posterior says UNKNOWN dominates (no face evidence, weak ReID).
    posterior = PosteriorDist({"UNKNOWN": 0.90, "alice": 0.10})
    face = PosteriorDist({})  # no face evidence
    reid = PosteriorDist({})  # no ReID evidence

    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        config=ResolverConfig(
            enable_sticky_maintenance=True,
            prior_maintenance_max_age_s=120.0,
        ),
    )

    decision = resolver._commit(gt, posterior, face, reid, now)
    # Sticky maintenance should hold the identity (not demote to UNKNOWN).
    assert decision.identity_id == "alice"
    assert not decision.revises_previous  # identity didn't change


def test_sticky_maintenance_overturned_by_face_contradiction(
    fresh_metrics: Metrics,
) -> None:
    """A strong face anchor for a different identity contradicts and overturns
    sticky maintenance."""
    now = datetime.now(UTC)
    gt = _make_gt(
        current_identity_id="alice",
        committed_at=now - timedelta(seconds=5),
    )
    # Posterior favors bob, but sticky would normally hold alice without
    # the face contradiction.
    posterior = PosteriorDist({"bob": 0.70, "UNKNOWN": 0.20, "alice": 0.10})
    # Face evidence strongly supports bob (contradiction).
    face = PosteriorDist({"bob": 0.90})
    reid = PosteriorDist({})

    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        config=ResolverConfig(
            enable_sticky_maintenance=True,
            prior_maintenance_max_age_s=120.0,
            contradiction_face_confidence=0.70,
        ),
    )

    # Register alice and bob as known identities.
    resolver.register_identity(Identity(identity_id="alice", display_name="Alice", enrolled_at=now))
    resolver.register_identity(Identity(identity_id="bob", display_name="Bob", enrolled_at=now))

    decision = resolver._commit(
        gt,
        posterior,
        face,
        reid,
        now,
        best_face_confidence=0.85,  # above contradiction threshold
    )
    # The strong face contradiction should overturn the held identity.
    assert decision.identity_id == "bob"
    assert decision.revises_previous


def test_sticky_maintenance_decays_outside_window(
    fresh_metrics: Metrics,
) -> None:
    """Outside the maintenance window with no evidence, identity decays to
    UNKNOWN even with sticky maintenance enabled."""
    now = datetime.now(UTC)
    gt = _make_gt(
        current_identity_id="alice",
        committed_at=now - timedelta(seconds=200),  # outside 120 s window
    )
    posterior = PosteriorDist({"UNKNOWN": 0.90, "alice": 0.10})
    face = PosteriorDist({})
    reid = PosteriorDist({})

    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        config=ResolverConfig(
            enable_sticky_maintenance=True,
            prior_maintenance_max_age_s=120.0,
        ),
    )

    decision = resolver._commit(gt, posterior, face, reid, now)
    # Outside the window, identity should decay.
    assert decision.identity_id is None
    assert decision.revises_previous


def test_sticky_maintenance_shadow_counts_when_disabled(
    fresh_metrics: Metrics,
) -> None:
    """When sticky maintenance is disabled, the shadow metric increments when
    the sticky rule would have made a different decision."""
    now = datetime.now(UTC)
    gt = _make_gt(
        current_identity_id="alice",
        committed_at=now - timedelta(seconds=5),
    )
    posterior = PosteriorDist({"UNKNOWN": 0.90, "alice": 0.10})
    face = PosteriorDist({})
    reid = PosteriorDist({})

    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        config=ResolverConfig(
            enable_sticky_maintenance=False,  # shadow mode
            prior_maintenance_max_age_s=120.0,
        ),
    )

    # Run commit: live decision should be UNKNOWN (no evidence, no maintenance
    # window because identity_unchanged is False).
    decision = resolver._commit(gt, posterior, face, reid, now)
    # Live: identity drops to UNKNOWN.
    assert decision.identity_id is None

    # Shadow counter should have incremented because sticky maintenance would
    # have held alice.
    assert (
        _metric_total(
            fresh_metrics.identity_shadow_mismatch_total.labels(feature="sticky_maintenance")
        )
        >= 1.0
    )


def test_sticky_maintenance_no_contradiction_with_same_face(
    fresh_metrics: Metrics,
) -> None:
    """A face anchor for the same identity is not a contradiction."""
    now = datetime.now(UTC)
    gt = _make_gt(
        current_identity_id="alice",
        committed_at=now - timedelta(seconds=5),
    )
    posterior = PosteriorDist({"alice": 0.80, "UNKNOWN": 0.20})
    face = PosteriorDist({"alice": 0.95})  # supports alice, not contradiction
    reid = PosteriorDist({})

    resolver = IdentityResolver(
        gallery_repo=InMemoryGalleryRepository(),
        config=ResolverConfig(
            enable_sticky_maintenance=True,
            prior_maintenance_max_age_s=120.0,
        ),
    )

    decision = resolver._commit(gt, posterior, face, reid, now, best_face_confidence=0.90)
    # Identity should be held (no contradiction).
    assert decision.identity_id == "alice"
    assert not decision.revises_previous


@pytest.mark.asyncio
async def test_duplicate_active_identity_guard_blocks_second_reid_assignment(
    fresh_metrics: Metrics,
) -> None:
    resolver = await _duplicate_identity_resolver(enable_guard=True)
    now = datetime.now(UTC)

    outcome = await resolver.resolve(
        hypotheses=[
            _make_gt(
                ph_id="ph-held",
                current_identity_id="amma",
                committed_at=now - timedelta(seconds=5),
                tracklet_ids=["t-held"],
            ),
            _make_gt(ph_id="ph-new", tracklet_ids=["t-new"]),
        ],
        new_face_anchors=[],
        captured_at=now,
    )

    decisions = {decision.ph_id: decision for decision in outcome.decisions}
    assert decisions["ph-held"].identity_id == "amma"
    assert decisions["ph-new"].identity_id is None
    assert "duplicate_active_identity_blocked: amma" in decisions["ph-new"].reason
    assert (
        _labeled_counter_value(
            fresh_metrics.identity_shadow_mismatch_total,
            "feature",
            "duplicate_active_identity",
        )
        == 0.0
    )


@pytest.mark.asyncio
async def test_duplicate_active_identity_guard_shadows_when_disabled(
    fresh_metrics: Metrics,
) -> None:
    resolver = await _duplicate_identity_resolver(enable_guard=False)
    now = datetime.now(UTC)

    outcome = await resolver.resolve(
        hypotheses=[
            _make_gt(
                ph_id="ph-held",
                current_identity_id="amma",
                committed_at=now - timedelta(seconds=5),
                tracklet_ids=["t-held"],
            ),
            _make_gt(ph_id="ph-new", tracklet_ids=["t-new"]),
        ],
        new_face_anchors=[],
        captured_at=now,
    )

    decisions = {decision.ph_id: decision for decision in outcome.decisions}
    assert decisions["ph-held"].identity_id == "amma"
    assert decisions["ph-new"].identity_id == "amma"
    assert (
        _labeled_counter_value(
            fresh_metrics.identity_shadow_mismatch_total,
            "feature",
            "duplicate_active_identity",
        )
        == 1.0
    )


@pytest.mark.asyncio
async def test_duplicate_active_identity_guard_allows_strong_direct_faces() -> None:
    gallery = InMemoryGalleryRepository()
    now = datetime.now(UTC)
    await gallery.upsert_identity(
        Identity(identity_id="amma", display_name="amma", enrolled_at=now)
    )
    await gallery.upsert_identity(
        Identity(identity_id="grandma", display_name="grandma", enrolled_at=now)
    )
    resolver = IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(
            commit_prob=0.50,
            enable_duplicate_active_identity_guard=True,
            duplicate_identity_direct_face_min_confidence=0.90,
        ),
    )

    outcome = await resolver.resolve(
        hypotheses=[
            _make_gt(ph_id="ph-a", tracklet_ids=["t-a"]),
            _make_gt(ph_id="ph-b", tracklet_ids=["t-b"]),
        ],
        new_face_anchors=[
            FaceAnchor(person_id="amma", confidence=0.95, quality=0.95, tracklet_id="t-a"),
            FaceAnchor(person_id="amma", confidence=0.96, quality=0.95, tracklet_id="t-b"),
        ],
        captured_at=now,
    )

    assert {decision.ph_id: decision.identity_id for decision in outcome.decisions} == {
        "ph-a": "amma",
        "ph-b": "amma",
    }


@pytest.mark.asyncio
async def test_duplicate_active_identity_guard_blocks_unobserved_incumbent() -> None:
    """An open PH that holds amma but is not observed this frame still blocks a
    second PH from acquiring amma via ReID alone (the keyframe-mislabel case)."""
    resolver = await _duplicate_identity_resolver(enable_guard=True)
    now = datetime.now(UTC)

    # Only ph-new is observed this frame; ph-held is open but undetected, so it
    # is absent from ``hypotheses`` yet present in ``open_ph_identities``.
    outcome = await resolver.resolve(
        hypotheses=[_make_gt(ph_id="ph-new", tracklet_ids=["t-new"])],
        new_face_anchors=[],
        captured_at=now,
        open_ph_identities={"ph-held": "amma", "ph-new": ""},
    )

    decisions = {decision.ph_id: decision for decision in outcome.decisions}
    assert decisions["ph-new"].identity_id is None
    assert "duplicate_active_identity_blocked: amma" in decisions["ph-new"].reason


@pytest.mark.asyncio
async def test_duplicate_active_identity_guard_allows_unobserved_incumbent_with_face() -> None:
    """Strong direct face evidence bypasses the unobserved-incumbent block, so
    genuine cross-camera continuation/handoff still commits."""
    gallery = InMemoryGalleryRepository()
    now = datetime.now(UTC)
    await gallery.upsert_identity(
        Identity(identity_id="amma", display_name="amma", enrolled_at=now)
    )
    resolver = IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(
            commit_prob=0.50,
            enable_duplicate_active_identity_guard=True,
            duplicate_identity_direct_face_min_confidence=0.90,
        ),
    )

    outcome = await resolver.resolve(
        hypotheses=[_make_gt(ph_id="ph-new", tracklet_ids=["t-new"])],
        new_face_anchors=[
            FaceAnchor(person_id="amma", confidence=0.96, quality=0.95, tracklet_id="t-new"),
        ],
        captured_at=now,
        open_ph_identities={"ph-held": "amma"},
    )

    decisions = {decision.ph_id: decision for decision in outcome.decisions}
    assert decisions["ph-new"].identity_id == "amma"


@pytest.mark.asyncio
async def test_duplicate_active_identity_guard_does_not_trust_propagated_face() -> None:
    """A *propagated* face anchor must not grant the strong-direct-face bypass.

    src has a genuine 'alice' face; dst has no own face but receives a high-
    confidence propagated 'alice' anchor that would otherwise commit alice.
    The bypass is computed from the original (pre-propagation) anchors, so dst
    is still blocked — this is the keyframe-mislabel contamination path.
    """
    resolver = await _propagation_resolver(0.95, enable_guard=True)
    src = _make_gt(ph_id="src", tracklet_ids=["t-src"], camera_ids=["cam-a"])
    dst = _make_gt(ph_id="dst", tracklet_ids=["t-dst"], camera_ids=["cam-b"])

    outcome = await resolver.resolve(
        hypotheses=[src, dst],
        new_face_anchors=[
            FaceAnchor(person_id="alice", confidence=1.0, quality=1.0, tracklet_id="t-src")
        ],
        captured_at=datetime.now(UTC),
        ph_qualities={"src": 0.9, "dst": 0.9},
    )

    decisions = {decision.ph_id: decision for decision in outcome.decisions}
    assert decisions["src"].identity_id == "alice"
    assert decisions["dst"].identity_id is None
    assert "duplicate_active_identity_blocked: alice" in decisions["dst"].reason
