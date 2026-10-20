"""resolver.face_quality_mode: how service-measured face quality weighs recognized face evidence.

``FaceAnchor.quality`` is the evidence-path weight; ``FaceAnchor.face_quality`` is the person-id
measurement. "off" ignores the measurement, "enabled" multiplies it in, and "shadow" decides as
"off" while counting where "enabled" would have decided differently.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from app.domain import FaceAnchor, GlobalTrack, Identity, IdentityDecision
from app.observability import metrics as metrics_pkg
from app.observability.metrics import Metrics, build_metrics
from app.storage.base import InMemoryGalleryRepository
from app.tracking.identity_resolver import FaceQualityMode, IdentityResolver, ResolverConfig

_NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
# Raw similarity just above the commit gates; commit_prob 0.8 separates a well-measured face
# (commits either way) from a poor one (commits only when quality is ignored).
_ANCHOR = FaceAnchor(
    person_id="alice",
    match_score=0.72,
    tracklet_id="t1",
    face_quality=0.9,
    face_quality_version="heuristic-v1",
)


@pytest.fixture
def fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> Metrics:
    fresh = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh)
    return fresh


def _mismatches(metrics: Metrics) -> float:
    return sum(
        sample.value
        for family in metrics.identity_shadow_mismatch_total.collect()
        for sample in family.samples
        if sample.name.endswith("_total") and sample.labels.get("feature") == "face_quality"
    )


async def _decide(anchor: FaceAnchor, mode: FaceQualityMode, **config: float) -> IdentityDecision:
    gallery = InMemoryGalleryRepository()
    for identity_id in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=_NOW)
        )
    resolver = IdentityResolver(
        gallery_repo=gallery, config=ResolverConfig(face_quality_mode=mode, **config)
    )
    entity = GlobalTrack(
        global_track_id="ph-1",
        camera_ids=["cam-a"],
        tracklet_ids=["t1"],
        started_at=_NOW,
        last_seen_at=_NOW,
        state="active",
    )
    outcome = await resolver.resolve(
        hypotheses=[entity],
        new_face_anchors=[anchor],
        captured_at=_NOW,
        ph_qualities={"ph-1": 0.9},
    )
    [decision] = outcome.decisions
    return decision


def _alice(decision: IdentityDecision) -> float:
    return decision.posterior.distribution["alice"]


async def test_enabled_weighs_a_better_measured_face_strictly_higher() -> None:
    good = await _decide(replace(_ANCHOR, face_quality=0.9), "enabled")
    poor = await _decide(replace(_ANCHOR, face_quality=0.2), "enabled")
    assert _alice(good) > _alice(poor)


async def test_off_ignores_face_quality_entirely() -> None:
    good = await _decide(replace(_ANCHOR, face_quality=0.9), "off")
    poor = await _decide(replace(_ANCHOR, face_quality=0.2), "off")
    unmeasured = await _decide(replace(_ANCHOR, face_quality=None, face_quality_version=""), "off")
    assert _alice(good) == _alice(poor) == _alice(unmeasured)


async def test_enabled_uses_the_conservative_default_for_an_unmeasured_native_face() -> None:
    unmeasured = await _decide(
        replace(_ANCHOR, face_quality=None, face_quality_version=""),
        "enabled",
        face_quality_missing_default=0.5,
    )
    half = await _decide(replace(_ANCHOR, face_quality=0.5), "enabled")
    perfect = await _decide(replace(_ANCHOR, face_quality=1.0), "enabled")
    assert _alice(unmeasured) == _alice(half)
    assert _alice(unmeasured) < _alice(perfect)


async def test_enabled_does_not_discount_an_external_anchor_again() -> None:
    external = replace(_ANCHOR, origin="cc_assertion", face_quality=None, quality=0.5)
    assert _alice(await _decide(external, "enabled")) == _alice(await _decide(external, "off"))


async def test_enabled_multiplies_evidence_path_and_face_quality() -> None:
    # A propagated anchor (evidence-path quality attenuated to 0.8) of a 0.5-quality face weighs
    # the same as a direct anchor of a 0.4-quality face.
    propagated = await _decide(replace(_ANCHOR, quality=0.8, face_quality=0.5), "enabled")
    direct = await _decide(replace(_ANCHOR, quality=1.0, face_quality=0.4), "enabled")
    assert _alice(propagated) == pytest.approx(_alice(direct))


async def test_shadow_decides_as_off_and_counts_only_real_disagreements(
    fresh_metrics: Metrics,
) -> None:
    poor = replace(_ANCHOR, face_quality=0.1)
    assert (await _decide(poor, "off", commit_prob=0.8)).identity_id == "alice"
    assert (await _decide(poor, "enabled", commit_prob=0.8)).identity_id is None
    assert _mismatches(fresh_metrics) == 0

    shadow = await _decide(poor, "shadow", commit_prob=0.8)
    assert shadow.identity_id == "alice"
    assert _alice(shadow) == _alice(await _decide(poor, "off", commit_prob=0.8))
    assert _mismatches(fresh_metrics) == 1

    # A well-measured face commits under both evaluations: no disagreement is counted.
    agreeing = await _decide(replace(_ANCHOR, face_quality=0.9), "shadow", commit_prob=0.8)
    assert agreeing.identity_id == "alice"
    assert _mismatches(fresh_metrics) == 1


async def test_shadow_counts_a_change_of_best_anchor_and_its_contradiction(
    fresh_metrics: Metrics,
) -> None:
    # alice is held. A poorly measured face names bob slightly more strongly than a well measured
    # face names alice: weighing quality changes which anchor is best, and with it whether the
    # held identity counts as contradicted.
    gallery = InMemoryGalleryRepository()
    for identity_id in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=_NOW)
        )
    held = _NOW - timedelta(seconds=5)
    entity = GlobalTrack(
        global_track_id="ph-1",
        camera_ids=["cam-a"],
        tracklet_ids=["t1"],
        started_at=_NOW,
        last_seen_at=_NOW,
        state="active",
        current_identity_id="alice",
        current_identity_committed_at=held,
        last_independent_identity_evidence_at=held,
    )
    anchors = [
        replace(_ANCHOR, person_id="bob", match_score=0.75, face_quality=0.1),
        replace(_ANCHOR, person_id="alice", match_score=0.72, face_quality=0.9),
    ]

    async def decide(mode: FaceQualityMode) -> str | None:
        resolver = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(face_quality_mode=mode, enable_sticky_maintenance=True),
        )
        outcome = await resolver.resolve(
            hypotheses=[entity], new_face_anchors=anchors, captured_at=_NOW
        )
        return outcome.decisions[0].identity_id

    assert await decide("off") == "bob"
    assert await decide("enabled") == "alice"
    assert await decide("shadow") == "bob"
    assert _mismatches(fresh_metrics) == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"face_quality_mode": "on"}, "face_quality_mode"),
        ({"face_quality_missing_default": 1.5}, "face_quality_missing_default"),
    ],
)
def test_invalid_face_quality_config_is_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ResolverConfig(**overrides)  # type: ignore[arg-type]
