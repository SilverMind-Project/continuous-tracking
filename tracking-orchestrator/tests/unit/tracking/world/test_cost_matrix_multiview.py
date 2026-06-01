"""Multiview association cost matrix tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.domain import OrientationBin, ViewPrototype
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.cost_matrix import pair_cost
from app.tracking.world.kalman import KalmanState, initialize


def _ph_state(x: float = 5.0, y: float = 3.0) -> KalmanState:
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    return initialize(x, y, 1.0, 2.0, now)


def _default_config(*, multiview: bool = False) -> WorldTrackerConfig:
    return WorldTrackerConfig(
        gate_chi2=9.21,
        alpha_geo=0.5,
        alpha_app=0.4,
        alpha_height=0.1,
        enable_multiview_association=multiview,
    )


def _normalize(vals: list[float]) -> list[float]:
    arr = np.asarray(vals, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr.tolist()


# ---------------------------------------------------------------------------
# Front embedding vs back embedding: distinct vectors
# ---------------------------------------------------------------------------

_FRONT_EMB = _normalize([1.0] + [0.0] * 767)
_BACK_EMB = _normalize([0.0] * 384 + [1.0] + [0.0] * 383)


def _front_proto(count: int = 3) -> ViewPrototype:
    return ViewPrototype(
        orientation=OrientationBin.FRONT,
        embedding=tuple(_FRONT_EMB),
        count=count,
    )


def _back_proto(count: int = 3) -> ViewPrototype:
    return ViewPrototype(
        orientation=OrientationBin.BACK,
        embedding=tuple(_BACK_EMB),
        count=count,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multiview_off_uses_gallery_mean() -> None:
    """When enable_multiview_association=False, gallery_mean is used
    even when prototypes exist."""
    cfg = _default_config(multiview=False)
    ks = _ph_state()

    # gallery_mean = FRONT embedding, observation = BACK embedding
    # With multiview off, the cost should be based on the (poor) match
    # between gallery_mean (front) and obs (back).
    cost = pair_cost(
        ph_state=ks,
        ph_gallery_mean=_FRONT_EMB,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=5.0,
        obs_floor_y_m=3.0,
        obs_embedding=_BACK_EMB,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=True,
        ph_view_prototypes=(_back_proto(),),  # has back proto but flag is off
    )
    # With front gallery_mean vs back obs, similarity should be low → cost high.
    # gallery_mean is front, obs is back — near-orthogonal → cost ≈ 1 - 0 + geometric
    assert cost > 0.3  # appearance cost should be significant


def test_multiview_on_matches_back_prototype() -> None:
    """With multiview on, a back observation matches the back prototype
    at low cost even when gallery_mean is front."""
    cfg = _default_config(multiview=True)
    ks = _ph_state()

    cost = pair_cost(
        ph_state=ks,
        ph_gallery_mean=_FRONT_EMB,  # front mean (poor for back obs)
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=5.0,
        obs_floor_y_m=3.0,
        obs_embedding=_BACK_EMB,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=True,
        ph_view_prototypes=(_front_proto(), _back_proto()),
    )
    # The back prototype should match the back observation well.
    # With max-over-prototypes, sim ≈ 1.0 → app_cost ≈ 0.0.
    assert cost < 0.3


def test_multiview_fallback_when_no_qualified_prototypes() -> None:
    """When prototypes exist but all have count < 2, fall back to gallery_mean."""
    cfg = _default_config(multiview=True)
    ks = _ph_state()

    # Back prototype with count=1 (below _PROTOTYPE_MIN_COUNT=2).
    low_count_proto = _back_proto(count=1)

    cost = pair_cost(
        ph_state=ks,
        ph_gallery_mean=_FRONT_EMB,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=5.0,
        obs_floor_y_m=3.0,
        obs_embedding=_BACK_EMB,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=True,
        ph_view_prototypes=(low_count_proto,),
    )
    # Falls back to gallery_mean (front) vs back obs → high cost.
    assert cost > 0.3


def test_multiview_empty_prototypes_fallback() -> None:
    """Empty prototypes → gallery_mean path, same as today."""
    cfg = _default_config(multiview=True)
    ks = _ph_state()

    # gallery_mean = BACK embedding, obs = BACK embedding → good match.
    cost = pair_cost(
        ph_state=ks,
        ph_gallery_mean=_BACK_EMB,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=5.0,
        obs_floor_y_m=3.0,
        obs_embedding=_BACK_EMB,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=True,
        ph_view_prototypes=(),
    )
    assert cost < 0.3


def test_single_mean_path_unchanged_with_multiview_off() -> None:
    """Byte-for-byte: when multiview is off, the result is identical
    to not passing prototypes at all."""
    cfg = _default_config(multiview=False)
    ks = _ph_state()

    cost1 = pair_cost(
        ph_state=ks,
        ph_gallery_mean=_BACK_EMB,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=5.0,
        obs_floor_y_m=3.0,
        obs_embedding=_BACK_EMB,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=True,
    )
    cost2 = pair_cost(
        ph_state=ks,
        ph_gallery_mean=_BACK_EMB,
        ph_current_identity_id=None,
        ph_height_m=None,
        obs_floor_x_m=5.0,
        obs_floor_y_m=3.0,
        obs_embedding=_BACK_EMB,
        obs_face_anchor_person_id=None,
        obs_face_anchor_confidence=0.0,
        obs_height_estimate_m=None,
        cfg=cfg,
        calibrated=True,
        ph_view_prototypes=(_back_proto(),),
    )
    assert cost1 == cost2  # prototypes ignored when flag is off
