"""Unit tests for single-view standing height metrology, uncertainty, and upright gating."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cts_contracts.optics import focal_length_px

from app.domain import BoundingBox
from app.inference.pose import Keypoint, PoseResult
from app.tracking.camera_pose import (
    is_upright_standing,
    pose_from_homography,
    standing_height_m,
    standing_height_with_sigma,
)
from tests.unit.tracking.test_camera_pose import _synthetic_camera


def _project_point(
    pt_world: tuple[float, float, float],
    rot: np.ndarray,
    centre: np.ndarray,
    f_px: float,
    principal: tuple[float, float],
) -> tuple[float, float]:
    p_cam = rot @ (np.array(pt_world, dtype=np.float64) - centre)
    assert p_cam[2] > 0.1, f"Point behind camera: {p_cam}"
    u = f_px * (p_cam[0] / p_cam[2]) + principal[0]
    v = f_px * (p_cam[1] / p_cam[2]) + principal[1]
    return (float(u), float(v))


def _make_pose_result(
    *,
    shoulder_y: float = 200.0,
    hip_y: float = 340.0,
    ankle_y: float = 520.0,
    shoulder_x: float = 500.0,
    hip_x: float = 500.0,
    ankle_score: float = 0.9,
    shoulder_score: float = 0.9,
    hip_score: float = 0.9,
) -> PoseResult:
    kps = [Keypoint(x=shoulder_x, y=shoulder_y, score=0.9) for _ in range(17)]
    # COCO-17: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip,
    # 15=left_ankle, 16=right_ankle
    kps[5] = Keypoint(x=shoulder_x - 20.0, y=shoulder_y, score=shoulder_score)
    kps[6] = Keypoint(x=shoulder_x + 20.0, y=shoulder_y, score=shoulder_score)
    kps[11] = Keypoint(x=hip_x - 15.0, y=hip_y, score=hip_score)
    kps[12] = Keypoint(x=hip_x + 15.0, y=hip_y, score=hip_score)
    kps[15] = Keypoint(x=hip_x - 10.0, y=ankle_y, score=ankle_score)
    kps[16] = Keypoint(x=hip_x + 10.0, y=ankle_y, score=ankle_score)
    return PoseResult(keypoints=kps)


@pytest.mark.parametrize("left_handed_floor", [False, True])
@pytest.mark.parametrize(
    "foot_xy",
    [
        (2.0, 0.0),
        (3.0, 0.6),
        (5.0, -1.2),
        (4.0, 1.5),
    ],
)
def test_standing_height_m_recovers_true_height_within_2cm(
    left_handed_floor: bool, foot_xy: tuple[float, float]
) -> None:
    """Noise-free synthetic 1.60 m person recovered within < 2 cm at 2 m, 3 m, 5 m, and edges."""
    h_i2f, rot, centre, f_px, principal = _synthetic_camera(
        centre_m=(0.0, 0.0, 2.5),
        tilt_deg=-30.0,
        yaw_deg=0.0,
        fov_deg=80.0,
        left_handed_floor=left_handed_floor,
    )
    pose = pose_from_homography(h_i2f, f_px, principal)

    true_height = 1.60
    head_px = _project_point((foot_xy[0], foot_xy[1], true_height), rot, centre, f_px, principal)

    est = standing_height_m(pose, foot_xy, head_px)
    assert est is not None
    assert abs(est - true_height) < 0.02


def test_standing_height_with_sigma_matches_monte_carlo() -> None:
    """Propagated uncertainty matches empirical Monte Carlo std within 20%."""
    h_i2f, rot, centre, f_px, principal = _synthetic_camera(
        centre_m=(0.0, 0.0, 2.5),
        tilt_deg=-30.0,
        yaw_deg=0.0,
        fov_deg=75.0,
    )
    pose = pose_from_homography(h_i2f, f_px, principal)

    foot_xy = (3.5, 0.3)
    true_height = 1.65
    head_px = _project_point((foot_xy[0], foot_xy[1], true_height), rot, centre, f_px, principal)
    sigma_head_px = 4.0
    foot_cov = np.array([[0.01, 0.002], [0.002, 0.008]], dtype=np.float64)

    res = standing_height_with_sigma(
        pose,
        foot_xy,
        foot_cov,
        head_px,
        sigma_head_px=sigma_head_px,
        sigma_model_m=0.0,
    )
    assert res is not None
    h_est, sigma_est = res
    assert h_est == pytest.approx(true_height, abs=0.005)

    rng = np.random.default_rng(42)
    draws: list[float] = []
    for _ in range(1000):
        df = rng.multivariate_normal([0.0, 0.0], foot_cov)
        du, dv = rng.normal(0.0, sigma_head_px, size=2)
        h = standing_height_m(
            pose,
            (foot_xy[0] + float(df[0]), foot_xy[1] + float(df[1])),
            (head_px[0] + float(du), head_px[1] + float(dv)),
        )
        assert h is not None
        draws.append(h)

    empirical_sigma = float(np.std(draws, ddof=1))
    rel_err = abs(sigma_est - empirical_sigma) / empirical_sigma
    assert rel_err < 0.20, f"sigma_est={sigma_est:.4f} vs empirical={empirical_sigma:.4f}"


def test_standing_height_rejects_overhead_ray_and_implausible_range() -> None:
    h_i2f, rot, centre, f_px, principal = _synthetic_camera(
        centre_m=(0.0, 0.0, 2.5),
        tilt_deg=-85.0,  # almost straight down
        yaw_deg=0.0,
        fov_deg=75.0,
    )
    pose = pose_from_homography(h_i2f, f_px, principal)
    # Person standing directly beneath camera -> ray almost vertical -> rejected.
    head_px = _project_point((0.02, 0.0, 1.65), rot, centre, f_px, principal)
    assert standing_height_m(pose, (0.02, 0.0), head_px) is None

    # Normal camera, out-of-range height (0.5 m -> sitting/child below floor gate).
    h2, r2, c2, f2, p2 = _synthetic_camera(centre_m=(0.0, 0.0, 2.5), tilt_deg=-30.0)
    pose2 = pose_from_homography(h2, f2, p2)
    low_head = _project_point((3.0, 0.0, 0.50), r2, c2, f2, p2)
    assert standing_height_m(pose2, (3.0, 0.0), low_head) is None
    tall_head = _project_point((3.0, 0.0, 2.40), r2, c2, f2, p2)
    assert standing_height_m(pose2, (3.0, 0.0), tall_head) is None


def test_is_upright_standing_gate() -> None:
    width_px, height_px = 1280, 720
    good_box = BoundingBox(x_min=540, y_min=150, x_max=700, y_max=560)
    good_pose = _make_pose_result()

    assert (
        is_upright_standing(good_pose, good_box, frame_width_px=width_px, frame_height_px=height_px)
        is True
    )
    assert (
        is_upright_standing(None, good_box, frame_width_px=width_px, frame_height_px=height_px)
        is False
    )
    top_trunc = BoundingBox(x_min=540, y_min=2, x_max=700, y_max=560)
    assert (
        is_upright_standing(
            good_pose, top_trunc, frame_width_px=width_px, frame_height_px=height_px
        )
        is False
    )
    bot_trunc = BoundingBox(x_min=540, y_min=150, x_max=700, y_max=718)
    assert (
        is_upright_standing(
            good_pose, bot_trunc, frame_width_px=width_px, frame_height_px=height_px
        )
        is False
    )
    edge_box = BoundingBox(x_min=20, y_min=150, x_max=160, y_max=560)
    assert (
        is_upright_standing(good_pose, edge_box, frame_width_px=width_px, frame_height_px=height_px)
        is False
    )
    wide_box = BoundingBox(x_min=440, y_min=250, x_max=760, y_max=520)
    assert (
        is_upright_standing(good_pose, wide_box, frame_width_px=width_px, frame_height_px=height_px)
        is False
    )
    bent_pose = _make_pose_result(shoulder_x=500.0, hip_x=600.0, shoulder_y=200.0, hip_y=300.0)
    assert (
        is_upright_standing(bent_pose, good_box, frame_width_px=width_px, frame_height_px=height_px)
        is False
    )
    no_ankles = _make_pose_result(ankle_score=0.1)
    assert (
        is_upright_standing(no_ankles, good_box, frame_width_px=width_px, frame_height_px=height_px)
        is False
    )
    inverted = _make_pose_result(shoulder_y=360.0, hip_y=220.0)
    assert (
        is_upright_standing(inverted, good_box, frame_width_px=width_px, frame_height_px=height_px)
        is False
    )
    assert math.isfinite(focal_length_px(width_px, 75.0))
