"""Tests for the prototype partition/rebuild service (M03 tasks 10-11)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np

from app.domain import OrientationBin
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.prototype_partition import (
    AcceptedObservation,
    partition_and_rebuild,
    rebuild_appearance,
)

CFG = WorldTrackerConfig()
BASE = datetime(2026, 6, 20, 9, 0, 0, tzinfo=UTC)


def _unit(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(8).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _obs(i: int, orientation: OrientationBin, embedding: list[float]) -> AcceptedObservation:
    return AcceptedObservation(
        embedding=embedding,
        orientation=orientation,
        orientation_confidence=0.9,
        quality=0.8,
        captured_at=BASE + timedelta(seconds=i),
    )


class TestRebuildDeterminism:
    def test_rebuild_is_order_independent(self) -> None:
        front = _unit(1)
        back = _unit(2)
        observations = [
            _obs(0, OrientationBin.FRONT, front),
            _obs(1, OrientationBin.BACK, back),
            _obs(2, OrientationBin.FRONT, front),
            _obs(3, OrientationBin.BACK, back),
        ]
        a = rebuild_appearance(observations, CFG)
        shuffled = observations[:]
        random.Random(7).shuffle(shuffled)
        b = rebuild_appearance(shuffled, CFG)

        assert a.accepted_count == b.accepted_count == 4
        assert a.gallery_mean is not None and b.gallery_mean is not None
        np.testing.assert_allclose(a.gallery_mean, b.gallery_mean, atol=1e-6)
        orients_a = {p.orientation for p in a.view_prototypes}
        orients_b = {p.orientation for p in b.view_prototypes}
        assert orients_a == orients_b
        by_orient_a = {p.orientation: p for p in a.view_prototypes}
        by_orient_b = {p.orientation: p for p in b.view_prototypes}
        for orient, pa in by_orient_a.items():
            np.testing.assert_allclose(pa.embedding, by_orient_b[orient].embedding, atol=1e-6)
            assert pa.count == by_orient_b[orient].count

    def test_rejected_samples_excluded_from_rebuild(self) -> None:
        good = _unit(1)
        observations = [
            _obs(0, OrientationBin.FRONT, good),
            # UNKNOWN orientation → rejected by the policy, not counted.
            _obs(1, OrientationBin.UNKNOWN, _unit(9)),
            _obs(2, OrientationBin.FRONT, good),
        ]
        result = rebuild_appearance(observations, CFG)
        assert result.accepted_count == 2
        assert {p.orientation for p in result.view_prototypes} == {OrientationBin.FRONT}


class TestPartition:
    def test_split_sides_rebuilt_independently(self) -> None:
        side_a = [_obs(0, OrientationBin.FRONT, _unit(1))]
        side_b = [_obs(1, OrientationBin.BACK, _unit(2))]
        rebuilt = partition_and_rebuild([side_a, side_b], CFG)
        assert len(rebuilt) == 2
        assert {p.orientation for p in rebuilt[0].view_prototypes} == {OrientationBin.FRONT}
        assert {p.orientation for p in rebuilt[1].view_prototypes} == {OrientationBin.BACK}
        # No appearance crossed the boundary.
        assert rebuilt[0].view_prototypes[0].embedding != rebuilt[1].view_prototypes[0].embedding

    def test_merge_concatenation_rebuilds_once(self) -> None:
        source = [_obs(0, OrientationBin.FRONT, _unit(1))]
        target = [_obs(1, OrientationBin.FRONT, _unit(1))]
        merged = partition_and_rebuild([source + target], CFG)
        assert len(merged) == 1
        assert merged[0].accepted_count == 2
        front = [p for p in merged[0].view_prototypes if p.orientation == OrientationBin.FRONT]
        assert front and front[0].count == 2

    def test_empty_partition_yields_empty_appearance(self) -> None:
        rebuilt = partition_and_rebuild([[]], CFG)
        assert rebuilt[0].gallery_mean is None
        assert rebuilt[0].view_prototypes == ()
        assert rebuilt[0].accepted_count == 0
