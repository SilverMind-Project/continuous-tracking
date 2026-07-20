"""Pure-module tests for the unified gallery vote scorer (identity-continuity M01).

Exercises app/tracking/identity/gallery_scoring.py directly: no resolver
instance, no I/O, per the engineering-standards pure-function testing
strategy.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from app.domain import GalleryEmbedding, OrientationBin
from app.tracking.identity.gallery_scoring import (
    GalleryScoringConfig,
    aggregate_max_over_views,
    aggregate_mean,
    cap_votes,
    score_hits,
)

_NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)


def _entry(
    *,
    entry_id: str = "g-1",
    identity_id: str = "alice",
    state: str = "operator_verified",
    seen_at: datetime = _NOW,
    source_episode_id: str | None = "ep1",
    camera_id: str = "cam1",
    orientation: int = OrientationBin.FRONT,
) -> GalleryEmbedding:
    return GalleryEmbedding(
        gallery_entry_id=entry_id,
        identity_id=identity_id,
        embedding=[0.0] * 8,
        seen_at=seen_at,
        state=state,
        source_episode_id=source_episode_id,
        camera_id=camera_id,
        orientation=orientation,
    )


def _identity_logistic(x: float) -> float:
    """Passthrough so trust/recency/cap tests isolate their own arithmetic."""
    return x


def test_trust_multiplier_by_state() -> None:
    cfg = GalleryScoringConfig()
    calls: list[None] = []

    hits = [
        (_entry(entry_id="verified", state="operator_verified"), 0.5),
        (_entry(entry_id="auto", state="auto_verified"), 0.5),
        (_entry(entry_id="pending", state="pending_review"), 0.5),
        (_entry(entry_id="rejected", state="rejected"), 0.5),
    ]
    scored = score_hits(
        hits,
        now=_NOW,
        cfg=cfg,
        logistic=_identity_logistic,
        on_nonvoting_state=lambda: calls.append(None),
    )
    by_id = {hit.entry.gallery_entry_id: hit for hit in scored}

    assert by_id["verified"].trust_multiplier == 2.0
    assert by_id["auto"].trust_multiplier == 1.5
    assert by_id["pending"].trust_multiplier == 1.0
    assert by_id["rejected"].trust_multiplier == 1.0
    # Backstop fires once per non-voting-state hit (pending, rejected); never
    # for operator_verified or auto_verified.
    assert len(calls) == 2


def test_recency_half_life_config() -> None:
    cfg = GalleryScoringConfig(recency_half_life_days=7.0)
    one_half_life_ago = _NOW.replace(day=13)  # July 20 -> July 13 = 7 days
    one_hour_in_future = _NOW.replace(hour=13)
    naive_same_instant = datetime(2026, 7, 20, 12, 0, 0)  # no tzinfo

    scored = score_hits(
        [
            (_entry(entry_id="fresh", seen_at=_NOW), 0.5),
            (_entry(entry_id="one-half-life", seen_at=one_half_life_ago), 0.5),
            (_entry(entry_id="future", seen_at=one_hour_in_future), 0.5),
            (_entry(entry_id="naive", seen_at=naive_same_instant), 0.5),
        ],
        now=_NOW,
        cfg=cfg,
        logistic=_identity_logistic,
        on_nonvoting_state=lambda: None,
    )
    by_id = {hit.entry.gallery_entry_id: hit for hit in scored}

    assert by_id["fresh"].recency_factor == pytest.approx(1.0)
    assert by_id["one-half-life"].recency_factor == pytest.approx(0.5)
    # Negative age (clock-skewed future seen_at) clamps to 1.0.
    assert by_id["future"].recency_factor == pytest.approx(1.0)
    # A naive seen_at is treated as UTC, matching an aware timestamp at the
    # same wall-clock instant.
    assert by_id["naive"].recency_factor == pytest.approx(1.0)


def test_cap_votes_keeps_strongest_per_group() -> None:
    cfg = GalleryScoringConfig()
    hits = [
        (
            _entry(
                entry_id="weak",
                source_episode_id="ep1",
                camera_id="cam1",
                orientation=OrientationBin.FRONT,
            ),
            0.3,
        ),
        (
            _entry(
                entry_id="strong",
                source_episode_id="ep1",
                camera_id="cam1",
                orientation=OrientationBin.FRONT,
            ),
            0.9,
        ),
        # Distinct episode -> separate group; must not collapse into ep1.
        (
            _entry(
                entry_id="other-episode",
                source_episode_id="ep2",
                camera_id="cam1",
                orientation=OrientationBin.FRONT,
            ),
            0.4,
        ),
    ]
    scored = score_hits(
        hits, now=_NOW, cfg=cfg, logistic=_identity_logistic, on_nonvoting_state=lambda: None
    )
    capped = cap_votes(scored)
    ids = {hit.entry.gallery_entry_id for hit in capped}

    assert ids == {"strong", "other-episode"}


def test_identified_entry_boost_applied_before_weighting() -> None:
    """The floor raise lands on the logit, so trust/recency still scale it.

    A raw logit of 0.70 is below identified_entry_min_likelihood (0.80) but
    the entry qualifies for the boost (identity-labeled entry, sim >= 0.65).
    Boosting the logit before weighting means the floored value (0.80) is
    what gets multiplied by trust and recency: 0.80 * 2.0 * 1.0 = 1.6.
    Boosting after weighting (the pre-M01 placement on the single-query
    fallback path) would never fire here at all: the unfloored weighted
    value 0.70 * 2.0 * 1.0 = 1.4 already exceeds 0.80, and the old
    post-weighting condition only raises a value that is still below the
    floor.
    """
    cfg = GalleryScoringConfig(
        identified_entry_boost_min_sim=0.65, identified_entry_min_likelihood=0.80
    )
    scored = score_hits(
        [(_entry(identity_id="alice", state="operator_verified"), 0.70)],
        now=_NOW,
        cfg=cfg,
        logistic=_identity_logistic,
        on_nonvoting_state=lambda: None,
    )

    assert scored[0].boosted is True
    assert scored[0].logit == pytest.approx(0.80)
    assert scored[0].weighted_logit == pytest.approx(1.6)


def test_aggregate_max_over_views_unknown_complement() -> None:
    """A weak best match must not normalize to the only enrolled identity."""
    cfg = GalleryScoringConfig()
    scored = score_hits(
        [(_entry(identity_id="alice", state="operator_verified"), 0.3)],
        now=_NOW,
        cfg=cfg,
        logistic=_identity_logistic,  # weighted_logit = 0.3 * 2.0 * 1.0 = 0.6
        on_nonvoting_state=lambda: None,
    )
    capped = cap_votes(scored)
    avg = aggregate_max_over_views(capped, identities={"alice"}, cfg=cfg)

    assert avg["alice"] == pytest.approx(0.6)
    assert avg["UNKNOWN"] >= 1.0 - avg["alice"]


def test_aggregate_mean_matches_legacy_fixture() -> None:
    """Golden fixture pinned against the pre-M01 `_score_gallery_hits` arithmetic.

    All similarities are below identified_entry_boost_min_sim (0.65), so the
    identified-entry boost never engages here; this isolates the logistic
    curve, trust multiplier, recency decay, vote capping, and mean
    aggregation, none of which the M01 refactor is meant to change on the
    fallback path. Hand-derived arithmetic (independently computed, not
    copied from a test run):

      logistic(x) = 1 / (1 + exp(-10 * (x - 0.70)))
      logistic(0.60) = 0.26894142136999516
      logistic(0.62) = 0.31002551887238766
      logistic(0.55) = 0.18242552380635646
      logistic(0.50) = 0.11920292202211755

      Entry A (alice, sim .60, operator_verified, age 0d):
        weighted = 0.26894142136999516 * 2.0 * 1.0 = 0.5378828427399903
      Entry D (alice, sim .62, operator_verified, age 0d; SAME group as A --
        identity/episode/camera/orientation all match -- cap_votes keeps
        the stronger of the two):
        weighted = 0.31002551887238766 * 2.0 * 1.0 = 0.6200510377447753
      Entry B (alice, sim .55, operator_verified, age 7d -- one half-life,
        recency = 2**(-7/7) = 0.5; distinct episode/camera/orientation):
        weighted = 0.18242552380635646 * 2.0 * 0.5 = 0.18242552380635646
      Entry C (bob, sim .50, pending_review -- trust 1.0, age 0d):
        weighted = 0.11920292202211755 * 1.0 * 1.0 = 0.11920292202211755

      alice mean over its capped group {D, B}:
        (0.6200510377447753 + 0.18242552380635646) / 2 = 0.4012382807755659
      bob mean over its capped group {C}: 0.11920292202211755
    """
    cfg = GalleryScoringConfig()
    seven_days_ago = _NOW.replace(day=13)
    hits = [
        (
            _entry(
                entry_id="A",
                identity_id="alice",
                state="operator_verified",
                seen_at=_NOW,
                source_episode_id="ep1",
                camera_id="cam1",
                orientation=OrientationBin.FRONT,
            ),
            0.60,
        ),
        (
            _entry(
                entry_id="D",
                identity_id="alice",
                state="operator_verified",
                seen_at=_NOW,
                source_episode_id="ep1",
                camera_id="cam1",
                orientation=OrientationBin.FRONT,
            ),
            0.62,
        ),
        (
            _entry(
                entry_id="B",
                identity_id="alice",
                state="operator_verified",
                seen_at=seven_days_ago,
                source_episode_id="ep2",
                camera_id="cam2",
                orientation=OrientationBin.BACK,
            ),
            0.55,
        ),
        (
            _entry(
                entry_id="C",
                identity_id="bob",
                state="pending_review",
                seen_at=_NOW,
                source_episode_id="ep1",
                camera_id="cam1",
                orientation=OrientationBin.FRONT,
            ),
            0.50,
        ),
    ]

    scored = score_hits(
        hits,
        now=_NOW,
        cfg=cfg,
        logistic=lambda x: 1.0 / (1.0 + math.exp(-10.0 * (x - 0.70))),
        on_nonvoting_state=lambda: None,
    )
    capped = cap_votes(scored)
    avg = aggregate_mean(capped, identities={"alice", "bob"}, cfg=cfg)

    assert avg["alice"] == pytest.approx(0.4012382807755659, abs=1e-9)
    assert avg["bob"] == pytest.approx(0.11920292202211755, abs=1e-9)
