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
) -> GlobalTrack:
    now = datetime.now(UTC)
    return GlobalTrack(
        global_track_id=ph_id,
        camera_ids=["cam-a"],
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
    def __init__(self, matches: list[tuple[str, float]]) -> None:
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
                ),
                score,
            )
            for identity_id, score in self._matches
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
