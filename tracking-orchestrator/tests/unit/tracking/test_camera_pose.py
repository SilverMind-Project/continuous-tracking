"""Unit tests for single-view camera pose recovery and validation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cts_contracts.optics import focal_length_px

from app.tracking.camera_pose import (
    CameraPoseUnreliableReason,
    pose_from_homography,
    validate_camera_pose,
)


def _synthetic_camera(
    *,
    centre_m: tuple[float, float, float] = (1.5, 2.0, 2.4),
    tilt_deg: float = -35.0,
    yaw_deg: float = 20.0,
    fov_deg: float = 75.0,
    width_px: int = 1920,
    height_px: int = 1080,
    left_handed_floor: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float]]:
    """Build a synthetic pinhole camera looking down at z=0 and its image->floor homography.

    Returns ``(H_image_to_floor, R_world_to_cam, C_world, f_px, (cx, cy))``.
    """
    f_px = focal_length_px(width_px, fov_deg)
    cx, cy = width_px / 2.0, height_px / 2.0
    k_mat = np.array([[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)  # negative = looking down

    # Optical axis z_c in world coords: horizontal azimuth `yaw`, elevation `tilt`.
    z_c = np.array(
        [
            math.cos(tilt) * math.cos(yaw),
            math.cos(tilt) * math.sin(yaw),
            math.sin(tilt),
        ],
        dtype=np.float64,
    )
    z_c /= np.linalg.norm(z_c)

    # World up is +Z. Camera image +y points down, so camera +x is horizontal right.
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_c = np.cross(z_c, world_up)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)  # points downward in world (negative Z component)
    y_c /= np.linalg.norm(y_c)

    if left_handed_floor:
        # Mirror floor Y axis (X right, Y down, Z up -- floor-plan editor convention).
        # Camera basis rows express camera axes in the left-handed world frame:
        # flip Y component of every world vector and keep x_c/y_c orthogonal.
        flip_y = np.diag([1.0, -1.0, 1.0])
        rot = np.vstack([x_c, y_c, z_c]) @ flip_y
    else:
        rot = np.vstack([x_c, y_c, z_c])

    centre = np.array(centre_m, dtype=np.float64)
    t_vec = -rot @ centre

    # Floor-to-image homography: H_f2i = K @ [r1, r2, t]
    h_f2i = k_mat @ np.column_stack([rot[:, 0], rot[:, 1], t_vec])
    h_i2f = np.linalg.inv(h_f2i)
    # Normalize arbitrarily to confirm scale invariance.
    h_i2f = h_i2f / h_i2f[2, 2] * -3.7
    return h_i2f, rot, centre, f_px, (cx, cy)


@pytest.mark.parametrize("left_handed_floor", [False, True])
@pytest.mark.parametrize("yaw_deg", [0.0, 35.0, -110.0, 175.0])
def test_pose_from_homography_recovers_centre_and_rotation(
    left_handed_floor: bool, yaw_deg: float
) -> None:
    """Recovered camera centre < 1 cm and rotation < 0.1 deg for right- and left-handed floors."""
    h_i2f, r_true, c_true, f_px, principal = _synthetic_camera(
        centre_m=(1.5, 2.0, 2.4),
        tilt_deg=-35.0,
        yaw_deg=yaw_deg,
        fov_deg=75.0,
        left_handed_floor=left_handed_floor,
    )

    pose = pose_from_homography(h_i2f, f_px, principal)

    centre_err_m = float(np.linalg.norm(pose.centre_m - c_true))
    assert centre_err_m < 0.01, f"centre error {centre_err_m:.4f} m >= 0.01 m"

    # Relative rotation matrix r_err = r_rec @ r_true^T should be close to I_3.
    r_err = pose.rotation @ r_true.T
    cos_angle = max(-1.0, min(1.0, (float(np.trace(r_err)) - 1.0) / 2.0))
    rot_err_deg = math.degrees(math.acos(cos_angle))
    assert rot_err_deg < 0.1, f"rotation error {rot_err_deg:.4f} deg >= 0.1 deg"

    assert pose.tilt_deg == pytest.approx(-35.0, abs=0.1)


def test_validate_camera_pose_accepts_valid_camera() -> None:
    h_i2f, _, _, _, _ = _synthetic_camera(
        centre_m=(2.0, 1.0, 2.4),
        tilt_deg=-35.0,
        fov_deg=75.0,
    )
    res = validate_camera_pose(
        "cam-1",
        h_i2f,
        image_width_px=1920,
        image_height_px=1080,
        horizontal_fov_deg=75.0,
        mounting_height_m=2.4,
        tilt_deg=-35.0,
    )
    assert res.reliable is True
    assert res.reason is None
    assert res.pose is not None
    assert res.recovered_height_m == pytest.approx(2.4, abs=0.01)
    assert res.recovered_tilt_deg == pytest.approx(-35.0, abs=0.1)


def test_validate_camera_pose_rejects_15pct_fov_error() -> None:
    """A 15% FOV error on a ceiling mount shifts recovered height beyond the 0.25 m tolerance."""
    h_i2f, _, _, _, _ = _synthetic_camera(
        centre_m=(2.0, 1.0, 2.4),
        tilt_deg=-65.0,
        fov_deg=75.0,
    )
    res = validate_camera_pose(
        "cam-1",
        h_i2f,
        image_width_px=1920,
        image_height_px=1080,
        horizontal_fov_deg=75.0 * 1.15,
        mounting_height_m=2.4,
        tilt_deg=-65.0,
    )
    assert res.reliable is False
    assert res.pose is None
    assert res.reason == CameraPoseUnreliableReason.HEIGHT_OUT_OF_TOLERANCE
    assert res.recovered_height_m is not None
    assert abs(res.recovered_height_m - 2.4) > 0.25


def test_validate_camera_pose_rejects_bad_tilt() -> None:
    h_i2f, _, _, _, _ = _synthetic_camera(
        centre_m=(2.0, 1.0, 2.4),
        tilt_deg=-35.0,
        fov_deg=75.0,
    )
    res = validate_camera_pose(
        "cam-1",
        h_i2f,
        image_width_px=1920,
        image_height_px=1080,
        horizontal_fov_deg=75.0,
        mounting_height_m=2.4,
        tilt_deg=-20.0,  # 15 deg off (> 10 deg tolerance)
    )
    assert res.reliable is False
    assert res.pose is None
    assert res.reason == CameraPoseUnreliableReason.TILT_OUT_OF_TOLERANCE


def test_validate_camera_pose_handles_missing_inputs_and_singular_matrix() -> None:
    missing_h = validate_camera_pose(
        "cam-1",
        None,
        image_width_px=1920,
        image_height_px=1080,
        horizontal_fov_deg=75.0,
        mounting_height_m=2.4,
        tilt_deg=-35.0,
    )
    assert missing_h.reliable is False
    assert missing_h.reason == CameraPoseUnreliableReason.MISSING_HOMOGRAPHY

    h_i2f, _, _, _, _ = _synthetic_camera()
    missing_profile = validate_camera_pose(
        "cam-1",
        h_i2f,
        image_width_px=1920,
        image_height_px=1080,
        horizontal_fov_deg=None,
        mounting_height_m=2.4,
        tilt_deg=-35.0,
    )
    assert missing_profile.reliable is False
    assert missing_profile.reason == CameraPoseUnreliableReason.MISSING_CAMERA_PROFILE

    singular = validate_camera_pose(
        "cam-1",
        np.zeros((3, 3), dtype=np.float64),
        image_width_px=1920,
        image_height_px=1080,
        horizontal_fov_deg=75.0,
        mounting_height_m=2.4,
        tilt_deg=-35.0,
    )
    assert singular.reliable is False
    assert singular.reason == CameraPoseUnreliableReason.SINGULAR_HOMOGRAPHY
