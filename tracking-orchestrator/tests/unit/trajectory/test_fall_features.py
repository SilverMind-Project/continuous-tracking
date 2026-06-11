"""Tests for FallFeatureExtractor and its per-frame primitives."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import BoundingBox
from app.inference.schemas import NUM_KEYPOINTS, Keypoint
from app.trajectory.fall_features import (
    FallFeatureExtractor,
    FallFrameInput,
    _body_vertical_y,
    _mean_image_y,
)
from app.trajectory.posture import PostureScores

_T0 = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)


def _kps(y: float, score: float = 0.9) -> tuple[Keypoint, ...]:
    """All 17 COCO keypoints at crop-y ``y`` (so every body group sits at y)."""
    return tuple(Keypoint(x=0.5, y=y, score=score) for _ in range(NUM_KEYPOINTS))


def _bbox(y_min: float, height: float, width: float = 120.0) -> BoundingBox:
    return BoundingBox(
        x_min=100,
        y_min=round(y_min),
        x_max=round(100 + width),
        y_max=round(y_min + height),
    )


def _frame(
    t_s: float,
    *,
    hip_y_abs: float,
    height: float,
    scale: float = 1.0,
    keypoints: bool = True,
    lying: float = 0.0,
    motion: float | None = None,
    floor_speed: float | None = None,
) -> FallFrameInput:
    """One frame with the hip placed at ``hip_y_abs`` (post-scale) absolute pixels.

    The hip keypoint sits at crop-y 0.5, so ``y_min = hip_y_abs - 0.5 * height``.
    """
    height_s = height * scale
    hip_abs_s = hip_y_abs * scale
    y_min = hip_abs_s - 0.5 * height_s
    return FallFrameInput(
        captured_at=_T0 + timedelta(seconds=t_s),
        bbox=_bbox(y_min, height_s),
        keypoints=_kps(0.5) if keypoints else None,
        posture_scores=PostureScores(lying=lying, sitting=0.0, standing_walking=0.0),
        floor_speed_m_s=floor_speed,
        motion_energy_nu_s=motion,
    )


# Continuous timeline: standing, then a linear descent ramp, then a fallen hold.
_H_STAND = 300.0
_H_LYING = 120.0
_HIP_STAND = 250.0  # absolute hip y while upright
_T_STAND = 1.5
_DROP_HEIGHTS = 0.6
_T_DROP = 0.6
_T_HOLD = 1.0


def _state_at(t_s: float) -> tuple[float, float]:
    """Return (hip_y_abs, bbox_height) at time ``t_s`` for the canonical fall."""
    drop_px = _DROP_HEIGHTS * _H_STAND
    if t_s < _T_STAND:
        return _HIP_STAND, _H_STAND
    if t_s < _T_STAND + _T_DROP:
        frac = (t_s - _T_STAND) / _T_DROP
        return _HIP_STAND + frac * drop_px, _H_STAND + frac * (_H_LYING - _H_STAND)
    return _HIP_STAND + drop_px, _H_LYING


def _fall_frames(
    fps: float,
    *,
    scale: float = 1.0,
    drop_keypoints_at: set[int] | None = None,
    motion: float | None = None,
) -> list[FallFrameInput]:
    """Sample the canonical fall timeline at ``fps``."""
    drop_kp = drop_keypoints_at or set()
    total = _T_STAND + _T_DROP + _T_HOLD
    dt = 1.0 / fps
    frames: list[FallFrameInput] = []
    i = 0
    t = 0.0
    while t <= total + 1e-9:
        hip_y_abs, height = _state_at(t)
        frames.append(
            _frame(
                t,
                hip_y_abs=hip_y_abs,
                height=height,
                scale=scale,
                keypoints=i not in drop_kp,
                motion=motion,
            )
        )
        i += 1
        t = i * dt
    return frames


def _run(frames: list[FallFrameInput], ph_id: str = "ph-1") -> object:
    extractor = FallFeatureExtractor()
    features = None
    for frame in frames:
        features = extractor.update(ph_id, frame)
    assert features is not None
    return features


def test_synthetic_fall_spikes_descent_and_drops_height_ratio() -> None:
    features = _run(_fall_frames(fps=10.0))
    assert features.max_descent_rate_hps > 0.8
    assert features.height_ratio_now < 0.5


def test_controlled_sit_stays_below_descent_threshold() -> None:
    # Drop 0.35 heights over 1.6 s (gentle, controlled).
    hip0, height0 = 250.0, 300.0
    drop_px = 0.35 * height0
    fps, t_stand, t_drop = 10.0, 1.0, 1.6
    total = t_stand + t_drop
    dt = 1.0 / fps
    frames: list[FallFrameInput] = []
    i = 0
    t = 0.0
    while t <= total + 1e-9:
        if t < t_stand:
            hip_y, height = hip0, height0
        else:
            frac = min((t - t_stand) / t_drop, 1.0)
            hip_y, height = hip0 + frac * drop_px, height0 - frac * 80.0
        frames.append(_frame(t, hip_y_abs=hip_y, height=height))
        i += 1
        t = i * dt
    features = _run(frames)
    assert features.max_descent_rate_hps < 0.4


def test_walk_away_keeps_height_ratio_and_no_descent_spike() -> None:
    # Bbox shrinks uniformly about its centre; hip (at centre) holds absolute-y.
    fps, total = 10.0, 2.0
    dt = 1.0 / fps
    center = 250.0
    frames: list[FallFrameInput] = []
    i = 0
    t = 0.0
    while t <= total + 1e-9:
        frac = t / total
        height = 300.0 - frac * 30.0  # 10 % shrink over the window
        hip_y = center  # uniform shrink leaves the centre keypoint put
        frames.append(_frame(t, hip_y_abs=hip_y, height=height))
        i += 1
        t = i * dt
    features = _run(frames)
    assert features.height_ratio_now > 0.8
    assert features.max_descent_rate_hps < 0.2


def test_camera_distance_invariance() -> None:
    near = _run(_fall_frames(fps=10.0, scale=1.0), ph_id="near")
    far = _run(_fall_frames(fps=10.0, scale=3.0), ph_id="far")
    assert near.max_descent_rate_hps == pytest.approx(far.max_descent_rate_hps, abs=1e-6)
    assert near.height_ratio_now == pytest.approx(far.height_ratio_now, abs=1e-6)
    assert near.samples == far.samples


def test_sparse_frames_still_exceed_descent_threshold() -> None:
    features = _run(_fall_frames(fps=2.0))
    assert features.max_descent_rate_hps > 0.8


def test_missing_keypoints_midfall_tolerated() -> None:
    # Blank the keypoints on two ramp frames (15..16 at 10 fps fall onset).
    features = _run(_fall_frames(fps=10.0, drop_keypoints_at={15, 16}))
    assert features.max_descent_rate_hps > 0.8
    # Frames with keypoints=None are retained: same buffer count as the full run.
    full = _run(_fall_frames(fps=10.0))
    assert features.samples == full.samples


def test_post_event_motion_none_until_window_complete() -> None:
    # The canonical timeline ends ~1 s after impact; the 2 s post-window is open.
    features = _run(_fall_frames(fps=10.0, motion=0.04))
    assert features.post_event_motion_nu_s is None


def test_post_event_motion_mean_once_window_closes() -> None:
    extractor = FallFeatureExtractor(buffer_s=6.0)
    features = None
    for frame in _fall_frames(fps=10.0, motion=0.04):
        features = extractor.update("ph-1", frame)
    # Hold still for a further 2.5 s so the post-event window fully closes.
    last_t = _T_STAND + _T_DROP + _T_HOLD
    hip_y, height = _state_at(last_t)
    for k in range(1, 26):
        features = extractor.update(
            "ph-1",
            _frame(last_t + k * 0.1, hip_y_abs=hip_y, height=height, motion=0.04),
        )
    assert features is not None
    assert features.post_event_motion_nu_s == pytest.approx(0.04, abs=1e-6)


def test_floor_speed_sampled_at_descent_event() -> None:
    frames = [
        _frame(0.0, hip_y_abs=250.0, height=300.0, floor_speed=0.3),
        _frame(0.1, hip_y_abs=250.0, height=300.0, floor_speed=0.3),
        # Sharp descent on the next frame; its floor speed must be the event speed.
        _frame(0.2, hip_y_abs=420.0, height=120.0, floor_speed=0.7),
    ]
    features = _run(frames)
    assert features.max_descent_rate_hps > 0.8
    assert features.floor_speed_at_event_m_s == 0.7


def test_explicit_evict_resets_anchor() -> None:
    extractor = FallFeatureExtractor()
    for frame in _fall_frames(fps=10.0):
        extractor.update("ph-1", frame)
    extractor.evict("ph-1")
    fresh = extractor.update("ph-1", _frame(100.0, hip_y_abs=250.0, height=300.0))
    assert fresh.samples == 1
    assert fresh.height_ratio_now == 1.0  # no anchor yet -> 1.0 by definition


def test_idle_eviction_frees_other_ph_state() -> None:
    extractor = FallFeatureExtractor()
    extractor.update("ph-a", _frame(0.0, hip_y_abs=250.0, height=300.0))
    extractor.update("ph-a", _frame(0.1, hip_y_abs=250.0, height=300.0))
    # A different PH updates 400 s later -> ph-a is idle-evicted (> 300 s).
    extractor.update("ph-b", _frame(400.0, hip_y_abs=250.0, height=300.0))
    revived = extractor.update("ph-a", _frame(400.1, hip_y_abs=250.0, height=300.0))
    assert revived.samples == 1  # state was freed, fresh buffer


def test_enter_grace_height_ratio_is_one() -> None:
    extractor = FallFeatureExtractor()
    f1 = extractor.update("ph-1", _frame(0.0, hip_y_abs=250.0, height=300.0))
    f2 = extractor.update("ph-1", _frame(0.3, hip_y_abs=400.0, height=120.0))
    assert f1.height_ratio_now == 1.0  # < 1 s of data
    assert f2.height_ratio_now == 1.0  # still < 1 s of data


# ── Primitive helpers ─────────────────────────────────────────────────────────


def test_mean_image_y_skips_low_score_keypoints() -> None:
    bbox = _bbox(100.0, 300.0)
    kps = list(_kps(0.5))
    kps[11] = Keypoint(x=0.5, y=0.5, score=0.1)  # left hip below floor
    kps[12] = Keypoint(x=0.5, y=0.7, score=0.9)  # right hip visible
    y = _mean_image_y(tuple(kps), (11, 12), bbox, 0.3)
    assert y == pytest.approx(100.0 + 0.7 * 300.0)


def test_body_vertical_y_falls_back_head_then_shoulder() -> None:
    bbox = _bbox(100.0, 300.0)
    # Hips visible -> hip midpoint.
    assert _body_vertical_y(_kps(0.5), bbox, 0.3) == pytest.approx(250.0)

    # Hips occluded -> head proxy (set head to a distinct crop-y).
    kps = list(_kps(0.6))
    for i in (11, 12):
        kps[i] = Keypoint(x=0.5, y=0.6, score=0.1)
    for i in (0, 1, 2, 3, 4):
        kps[i] = Keypoint(x=0.5, y=0.2, score=0.9)
    assert _body_vertical_y(tuple(kps), bbox, 0.3) == pytest.approx(100.0 + 0.2 * 300.0)

    # Hips and head occluded -> shoulder midpoint.
    for i in (0, 1, 2, 3, 4):
        kps[i] = Keypoint(x=0.5, y=0.2, score=0.1)
    for i in (5, 6):
        kps[i] = Keypoint(x=0.5, y=0.4, score=0.9)
    assert _body_vertical_y(tuple(kps), bbox, 0.3) == pytest.approx(100.0 + 0.4 * 300.0)


def test_body_vertical_y_none_when_all_occluded() -> None:
    bbox = _bbox(100.0, 300.0)
    assert _body_vertical_y(_kps(0.5, score=0.05), bbox, 0.3) is None
