"""Tests for the PH-local appearance-update policy (M03 tasks 7-9)."""

from __future__ import annotations

import math

import numpy as np

from app.domain import OrientationBin, ViewPrototype
from app.tracking.world.appearance_policy import (
    AppearanceRejectReason,
    evaluate_appearance_update,
)
from app.tracking.world.config import WorldTrackerConfig


def _unit(seed: float) -> list[float]:
    """A deterministic 8-dim unit embedding."""
    rng = np.random.default_rng(int(seed * 1000))
    v = rng.standard_normal(8).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _proto(orientation: OrientationBin, embedding: list[float], count: int = 3) -> ViewPrototype:
    v = np.asarray(embedding, dtype=np.float32)
    unit = (v / np.linalg.norm(v)).tolist()
    return ViewPrototype(orientation=orientation, embedding=tuple(unit), count=count)


CFG = WorldTrackerConfig()


class TestAcceptance:
    def test_clean_consistent_sample_accepted(self) -> None:
        emb = _unit(1.0)
        existing = (_proto(OrientationBin.FRONT, emb),)
        decision = evaluate_appearance_update(
            embedding=emb,
            orientation=OrientationBin.FRONT,
            orientation_confidence=0.9,
            quality=0.8,
            existing_prototypes=existing,
            cfg=CFG,
        )
        assert decision.accept
        assert decision.reason is None

    def test_new_qualified_orientation_accepted(self) -> None:
        decision = evaluate_appearance_update(
            embedding=_unit(2.0),
            orientation=OrientationBin.BACK,
            orientation_confidence=0.9,
            quality=0.8,
            existing_prototypes=(_proto(OrientationBin.FRONT, _unit(1.0)),),
            cfg=CFG,
        )
        assert decision.accept


class TestRejection:
    def test_cross_person_outlier_rejected(self) -> None:
        front = _unit(1.0)
        outlier = (-np.asarray(front, dtype=np.float32)).tolist()  # cosine = -1
        decision = evaluate_appearance_update(
            embedding=outlier,
            orientation=OrientationBin.FRONT,
            orientation_confidence=0.9,
            quality=0.9,
            existing_prototypes=(_proto(OrientationBin.FRONT, front),),
            cfg=CFG,
        )
        assert not decision.accept
        assert decision.reason is AppearanceRejectReason.CROSS_PERSON_OUTLIER

    def test_low_quality_rejected(self) -> None:
        decision = evaluate_appearance_update(
            embedding=_unit(1.0),
            orientation=OrientationBin.FRONT,
            orientation_confidence=0.9,
            quality=0.05,
            existing_prototypes=(),
            cfg=CFG,
        )
        assert not decision.accept
        assert decision.reason is AppearanceRejectReason.LOW_QUALITY

    def test_unknown_orientation_rejected(self) -> None:
        decision = evaluate_appearance_update(
            embedding=_unit(1.0),
            orientation=OrientationBin.UNKNOWN,
            orientation_confidence=0.9,
            quality=0.9,
            existing_prototypes=(),
            cfg=CFG,
        )
        assert not decision.accept
        assert decision.reason is AppearanceRejectReason.UNKNOWN_ORIENTATION

    def test_low_orientation_confidence_blocks_new_prototype(self) -> None:
        decision = evaluate_appearance_update(
            embedding=_unit(1.0),
            orientation=OrientationBin.LEFT,
            orientation_confidence=0.1,
            quality=0.9,
            existing_prototypes=(),
            cfg=CFG,
        )
        assert not decision.accept
        assert decision.reason is AppearanceRejectReason.LOW_ORIENTATION_CONFIDENCE

    def test_non_finite_rejected(self) -> None:
        decision = evaluate_appearance_update(
            embedding=[math.nan] * 8,
            orientation=OrientationBin.FRONT,
            orientation_confidence=0.9,
            quality=0.9,
            existing_prototypes=(),
            cfg=CFG,
        )
        assert not decision.accept
        assert decision.reason is AppearanceRejectReason.NOT_FINITE

    def test_degenerate_norm_rejected(self) -> None:
        decision = evaluate_appearance_update(
            embedding=[0.0] * 8,
            orientation=OrientationBin.FRONT,
            orientation_confidence=0.9,
            quality=0.9,
            existing_prototypes=(),
            cfg=CFG,
        )
        assert not decision.accept
        assert decision.reason is AppearanceRejectReason.DEGENERATE_NORM

    def test_empty_embedding_rejected(self) -> None:
        decision = evaluate_appearance_update(
            embedding=None,
            orientation=OrientationBin.FRONT,
            orientation_confidence=0.9,
            quality=0.9,
            existing_prototypes=(),
            cfg=CFG,
        )
        assert not decision.accept
        assert decision.reason is AppearanceRejectReason.NO_EMBEDDING


class TestKillSwitch:
    def test_disabled_accepts_any_non_empty_embedding(self) -> None:
        cfg = WorldTrackerConfig(enable_appearance_outlier_rejection=False)
        front = _unit(1.0)
        outlier = (-np.asarray(front, dtype=np.float32)).tolist()
        decision = evaluate_appearance_update(
            embedding=outlier,
            orientation=OrientationBin.UNKNOWN,
            orientation_confidence=0.0,
            quality=0.0,
            existing_prototypes=(_proto(OrientationBin.FRONT, front),),
            cfg=cfg,
        )
        assert decision.accept
