"""Unit tests for combine_channels() and ChannelLikelihood clock semantics.

Proves numerical equivalence (to 1e-12) between combine_channels() and the
legacy combine_posteriors() multiplicative rule across a seeded table of random
distributions, and verifies multi-channel evidence_at aggregation.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import IdentityDecision, PersonHypothesis, PosteriorDist
from app.tracking.identity.channels import ChannelLikelihood
from app.tracking.identity.posterior import combine_channels
from app.tracking.identity_resolver import IdentityResolver


def _legacy_combine_posteriors(
    prior: PosteriorDist,
    face: PosteriorDist,
    reid: PosteriorDist,
    height: PosteriorDist | None = None,
    *,
    face_weight_multiplier: float = 3.0,
    height_weight_multiplier: float = 1.5,
) -> PosteriorDist:
    """Exact pre-refactor reference formula pinned for equivalence verification."""
    all_ids: set[str] = set(prior.distribution.keys())
    all_ids.update(face.distribution.keys())
    all_ids.update(reid.distribution.keys())
    if height is not None and height.distribution:
        all_ids.update(height.distribution.keys())

    if not all_ids:
        return PosteriorDist({"UNKNOWN": 1.0})

    def _w(dist: PosteriorDist, ident: str) -> float:
        if not dist.distribution:
            return 1.0
        if ident in dist.distribution:
            return dist.distribution[ident]
        return 1.0 / (len(dist.distribution) + 1)

    combined: dict[str, float] = {}
    for ident in sorted(all_ids):
        fw = _w(face, ident)
        if ident in face.distribution:
            fw = fw * face_weight_multiplier

        hw = _w(height, ident) if height is not None else 1.0
        if height is not None and height.distribution and ident in height.distribution:
            hw = hw * height_weight_multiplier

        combined[ident] = _w(prior, ident) * fw * _w(reid, ident) * hw

    return PosteriorDist(combined)


def _rand_dist(rng: random.Random, universe: list[str]) -> PosteriorDist:
    if rng.random() < 0.25:
        return PosteriorDist({})
    k = rng.randint(1, len(universe))
    chosen = rng.sample(universe, k)
    raw = {ident: rng.uniform(0.01, 1.0) for ident in chosen}
    return PosteriorDist(raw)


def test_combine_channels_matches_legacy_reference_to_1e12() -> None:
    rng = random.Random(20261120)
    universe = ["alice", "bob", "carol", "dave", "UNKNOWN"]

    for _ in range(200):
        prior = _rand_dist(rng, universe)
        face = _rand_dist(rng, universe)
        reid = _rand_dist(rng, universe)
        height = None if rng.random() < 0.3 else _rand_dist(rng, universe)
        face_mult = rng.choice([1.0, 2.5, 3.0, 4.0])
        height_mult = rng.choice([1.0, 1.5, 2.0])

        expected = _legacy_combine_posteriors(
            prior,
            face,
            reid,
            height,
            face_weight_multiplier=face_mult,
            height_weight_multiplier=height_mult,
        )
        channels = [
            ChannelLikelihood("face", face, face_mult),
            ChannelLikelihood("gallery", reid, 1.0),
        ]
        if height is not None:
            channels.append(ChannelLikelihood("height", height, height_mult))
        actual = combine_channels(prior, channels)

        assert list(actual.distribution.keys()) == list(expected.distribution.keys())
        for ident, exp_prob in expected.distribution.items():
            assert actual.distribution[ident] == pytest.approx(exp_prob, abs=1e-12)


def test_full_support_channel_uses_pre_scaled_likelihood_with_unit_multiplier() -> None:
    """When a channel covers all identities + UNKNOWN, power-scaling before PosteriorDist
    changes the normalized posterior while support_multiplier alone cancels out."""
    prior = PosteriorDist({"alice": 0.5, "bob": 0.5, "UNKNOWN": 0.1})
    raw = {"alice": 0.7, "bob": 0.2, "UNKNOWN": 0.1}

    base = combine_channels(prior, (ChannelLikelihood("wearable", PosteriorDist(raw), 1.0),))
    scalar_only = combine_channels(prior, (ChannelLikelihood("wearable", PosteriorDist(raw), 5.0),))
    power_scaled = combine_channels(
        prior,
        (
            ChannelLikelihood(
                "wearable",
                PosteriorDist({k: v**2.0 for k, v in raw.items()}),
                1.0,
            ),
        ),
    )

    assert scalar_only.distribution["alice"] == pytest.approx(base.distribution["alice"], abs=1e-12)
    assert power_scaled.distribution["alice"] > base.distribution["alice"]


def test_resolve_clock_evidence_selects_newest_time_across_channels_for_decided_id() -> None:
    """Channels report their own evidence_at; resolver selects the newest for the decided id."""
    t0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    t_template = t0 - timedelta(seconds=10)
    t_wearable = t0 - timedelta(seconds=2)
    t_other = t0

    ph = PersonHypothesis(
        ph_id="ph-1",
        state_mean=(0.0, 0.0, 0.0, 0.0),
        state_cov=(0.1, 0.0, 0.0, 0.0) * 4,
        born_at=t0 - timedelta(seconds=30),
        last_seen_at=t0,
        last_seen_camera="cam-1",
        observation_count=5,
        current_identity_id=None,
    )
    decision = IdentityDecision(
        ph_id="ph-1",
        identity_id="alice",
        posterior=PosteriorDist({"alice": 0.9, "UNKNOWN": 0.1}),
        revises_previous=True,
        previous_identity_id=None,
        evidence_backed=True,
    )
    face_ch = ChannelLikelihood(
        channel="face",
        distribution=PosteriorDist({"alice": 0.8, "UNKNOWN": 0.2}),
        support_multiplier=3.0,
        evidence_at=t_other,
        clock_evidence_ids=frozenset({"alice", "bob"}),
        evidence_at_by_identity={"alice": t_template, "bob": t_other},
    )
    wearable_ch = ChannelLikelihood(
        channel="wearable",
        distribution=PosteriorDist({"alice": 0.9, "UNKNOWN": 0.1}),
        support_multiplier=1.0,
        evidence_at=t_wearable,
        clock_evidence_ids=frozenset({"alice"}),
    )

    # Without wearable, Alice gets her template's time (not Bob's newer t_other).
    has_ev_face, ts_face = IdentityResolver._resolve_clock_evidence(
        ph, decision, channels=(face_ch,)
    )
    assert has_ev_face is True
    assert ts_face == t_template

    # With wearable channel, newest across channels for Alice is t_wearable.
    has_ev_both, ts_both = IdentityResolver._resolve_clock_evidence(
        ph, decision, channels=(face_ch, wearable_ch)
    )
    assert has_ev_both is True
    assert ts_both == t_wearable
