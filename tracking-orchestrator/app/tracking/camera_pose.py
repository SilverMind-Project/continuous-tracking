"""Single-view camera pose recovery, standing height metrology, and PH height fusion.

Pure geometry and statistics (numpy + ``cts_contracts.optics.focal_length_px`` only; no I/O or DB).

Coordinate frames and units:
- Floor frame: household floor plan in metres (``x_m``, ``y_m``, ``z_m`` with ``z_m`` pointing up
  from the floor plane ``z = 0``). Supports both right-handed floor frames and the top-left-origin
  floor-plan frame (X right, Y down on plan, Z up, left-handed) by fixing homography scale from
  positive camera depth at the image centre and picking the vertical basis direction so the camera
  centre lies above the floor (``centre_m[2] > 0``).
- Camera frame: right-handed OpenCV pinhole camera (``x`` right, ``y`` down, ``z`` forward).
- ``tilt_deg``: signed downward pitch of the optical axis in ``[-90, 0]`` degrees
  (``0`` = horizontal, ``-90`` = pointing straight down), matching the camera profile schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from cts_contracts.optics import focal_length_px

from app.domain import BoundingBox
from app.inference.schemas import PoseResult


class CameraPoseUnreliableReason(StrEnum):
    """Closed reason set explaining why a camera cannot produce height measurements."""

    MISSING_HOMOGRAPHY = "missing_homography"
    MISSING_CAMERA_PROFILE = "missing_camera_profile"
    SINGULAR_HOMOGRAPHY = "singular_homography"
    INVALID_POSE = "invalid_pose"
    HEIGHT_OUT_OF_TOLERANCE = "height_out_of_tolerance"
    TILT_OUT_OF_TOLERANCE = "tilt_out_of_tolerance"


@dataclass(frozen=True)
class CameraPose:
    """Recovered pinhole extrinsics and intrinsics for one calibrated camera.

    - ``rotation``: 3x3 orthogonal matrix mapping floor coordinates to camera coordinates.
    - ``centre_m``: 3-vector camera optical centre in floor frame, metres, ``z`` pointing up.
    - ``focal_px``: pinhole focal length in pixels (square pixels assumed).
    - ``principal_point_px``: ``(cx_px, cy_px)`` optical centre in image pixels.
    - ``tilt_deg``: optical-axis pitch in ``[-90, 0]`` degrees (``0`` horizontal, ``-90`` down).
    """

    rotation: npt.NDArray[np.float64]
    centre_m: npt.NDArray[np.float64]
    focal_px: float
    principal_point_px: tuple[float, float]
    tilt_deg: float


@dataclass(frozen=True)
class CameraPoseValidation:
    """Result of recovering and validating one camera's pose against its configured profile."""

    camera_id: str
    reliable: bool
    pose: CameraPose | None
    reason: CameraPoseUnreliableReason | None = None
    recovered_height_m: float | None = None
    expected_height_m: float | None = None
    recovered_tilt_deg: float | None = None
    expected_tilt_deg: float | None = None


def _intrinsics_matrix(
    focal_px: float, principal_point_px: tuple[float, float]
) -> npt.NDArray[np.float64]:
    cx, cy = principal_point_px
    return np.array(
        [
            [focal_px, 0.0, cx],
            [0.0, focal_px, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pose_from_homography(
    h_image_to_floor: npt.NDArray[np.float64] | list[list[float]],
    focal_px: float,
    principal_point_px: tuple[float, float],
) -> CameraPose:
    """Recover camera pose in the floor frame from a pixel-to-floor-metres homography.

    Args:
        h_image_to_floor: 3x3 homography mapping image pixels ``(u, v, 1)`` to floor metres
            ``(X_m, Y_m, 1)`` on the plane ``z = 0``.
        focal_px: Pin-hole focal length in pixels (positive, finite).
        principal_point_px: ``(cx_px, cy_px)`` image principal point in pixels.

    Raises:
        ValueError: If *focal_px* is invalid, the homography is singular/non-finite, or the
            recovered camera centre does not lie above the floor plane.
    """
    if not math.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError(f"focal_px must be finite and > 0, got {focal_px!r}")
    cx, cy = principal_point_px
    if not (math.isfinite(cx) and math.isfinite(cy)):
        raise ValueError(f"principal_point_px must be finite, got {principal_point_px!r}")

    h_i2f = np.asarray(h_image_to_floor, dtype=np.float64)
    if h_i2f.shape != (3, 3) or not np.all(np.isfinite(h_i2f)):
        raise ValueError("h_image_to_floor must be a finite 3x3 matrix")

    det_h = float(np.linalg.det(h_i2f))
    if not math.isfinite(det_h) or abs(det_h) < 1e-12:
        raise ValueError("h_image_to_floor is singular")

    h_f2i = np.linalg.inv(h_i2f)
    k = _intrinsics_matrix(focal_px, principal_point_px)
    k_inv = np.linalg.inv(k)

    # M = K^-1 H_floor_to_image ~ [r1, r2, t] up to a non-zero scalar lambda.
    m = k_inv @ h_f2i
    n1 = float(np.linalg.norm(m[:, 0]))
    n2 = float(np.linalg.norm(m[:, 1]))
    if not (math.isfinite(n1) and math.isfinite(n2)) or n1 < 1e-12 or n2 < 1e-12:
        raise ValueError("degenerate homography columns")

    lam = 2.0 / (n1 + n2)

    # Fix global scale sign so the floor point imaged at the principal point lies in front
    # of the camera (positive camera z). If the principal ray is parallel to the floor,
    # fall back to checking the bottom-centre of the image.
    probe_pixels = [(cx, cy), (cx, cy * 1.5)]
    sign = 1.0
    sign_found = False
    for pu, pv in probe_pixels:
        floor_h = h_i2f @ np.array([pu, pv, 1.0], dtype=np.float64)
        if abs(floor_h[2]) > 1e-9:
            floor_pt = np.array([floor_h[0] / floor_h[2], floor_h[1] / floor_h[2], 1.0])
            cam_z = float((m @ floor_pt)[2])
            if abs(cam_z) > 1e-9:
                sign = 1.0 if cam_z > 0.0 else -1.0
                sign_found = True
                break
    if not sign_found:
        sign = 1.0 if float(m[2, 2]) >= 0.0 else -1.0

    r1 = sign * lam * m[:, 0]
    r2 = sign * lam * m[:, 1]
    t = sign * lam * m[:, 2]

    # Pick the vertical floor-normal direction r3 = +/- (r1 x r2) so the camera optical centre
    # sits above the floor plane (C_z = -r3^T t > 0). Choosing the sign explicitly handles both
    # right-handed floor frames and top-left-origin (left-handed) floor-plan coordinates.
    r3_pos = np.cross(r1, r2)
    r3_norm = float(np.linalg.norm(r3_pos))
    if not math.isfinite(r3_norm) or r3_norm < 1e-9:
        raise ValueError("degenerate homography rotation columns")
    r3_pos = r3_pos / r3_norm
    r3 = r3_pos if float(-np.dot(r3_pos, t)) >= 0.0 else -r3_pos

    # Polar SVD yields the nearest orthogonal matrix R (preserving floor-frame handedness).
    raw_r = np.column_stack((r1, r2, r3))
    u_mat, _s_vals, vt_mat = np.linalg.svd(raw_r)
    rotation: npt.NDArray[np.float64] = u_mat @ vt_mat
    if not np.all(np.isfinite(rotation)):
        raise ValueError("non-finite rotation matrix")

    centre_m: npt.NDArray[np.float64] = -(rotation.T @ t)
    if not np.all(np.isfinite(centre_m)) or float(centre_m[2]) <= 0.0:
        raise ValueError(f"recovered camera centre is not above floor: {centre_m.tolist()}")

    # Camera optical axis in floor coordinates is R^T @ [0, 0, 1] = R[2, :].
    # Signed tilt in [-90, 0] when looking downward (optical_axis_z <= 0).
    optical_axis_z = max(-1.0, min(1.0, float(rotation[2, 2])))
    tilt_deg = math.degrees(math.asin(optical_axis_z))

    return CameraPose(
        rotation=rotation,
        centre_m=centre_m,
        focal_px=float(focal_px),
        principal_point_px=(float(cx), float(cy)),
        tilt_deg=tilt_deg,
    )


def validate_camera_pose(
    camera_id: str,
    h_image_to_floor: npt.NDArray[np.float64] | list[list[float]] | None,
    *,
    image_width_px: int,
    image_height_px: int,
    horizontal_fov_deg: float | None,
    mounting_height_m: float | None,
    tilt_deg: float | None,
    pose_height_tolerance_m: float = 0.25,
    pose_tilt_tolerance_deg: float = 10.0,
) -> CameraPoseValidation:
    """Recover a camera pose and validate it against configured mount height and tilt."""
    if h_image_to_floor is None:
        return CameraPoseValidation(
            camera_id=camera_id,
            reliable=False,
            pose=None,
            reason=CameraPoseUnreliableReason.MISSING_HOMOGRAPHY,
            expected_height_m=mounting_height_m,
            expected_tilt_deg=tilt_deg,
        )

    if (
        image_width_px <= 0
        or image_height_px <= 0
        or horizontal_fov_deg is None
        or mounting_height_m is None
        or tilt_deg is None
        or not (0.0 < horizontal_fov_deg < 180.0)
        or mounting_height_m <= 0.0
    ):
        return CameraPoseValidation(
            camera_id=camera_id,
            reliable=False,
            pose=None,
            reason=CameraPoseUnreliableReason.MISSING_CAMERA_PROFILE,
            expected_height_m=mounting_height_m,
            expected_tilt_deg=tilt_deg,
        )

    f_px = focal_length_px(image_width_px, horizontal_fov_deg)
    principal = (image_width_px / 2.0, image_height_px / 2.0)

    try:
        pose = pose_from_homography(h_image_to_floor, f_px, principal)
    except ValueError as exc:
        reason = (
            CameraPoseUnreliableReason.SINGULAR_HOMOGRAPHY
            if "singular" in str(exc).lower()
            else CameraPoseUnreliableReason.INVALID_POSE
        )
        return CameraPoseValidation(
            camera_id=camera_id,
            reliable=False,
            pose=None,
            reason=reason,
            expected_height_m=mounting_height_m,
            expected_tilt_deg=tilt_deg,
        )

    recovered_height_m = float(pose.centre_m[2])
    recovered_tilt_deg = float(pose.tilt_deg)

    if abs(recovered_height_m - mounting_height_m) > pose_height_tolerance_m:
        return CameraPoseValidation(
            camera_id=camera_id,
            reliable=False,
            pose=None,
            reason=CameraPoseUnreliableReason.HEIGHT_OUT_OF_TOLERANCE,
            recovered_height_m=recovered_height_m,
            expected_height_m=mounting_height_m,
            recovered_tilt_deg=recovered_tilt_deg,
            expected_tilt_deg=tilt_deg,
        )

    if abs(recovered_tilt_deg - tilt_deg) > pose_tilt_tolerance_deg:
        return CameraPoseValidation(
            camera_id=camera_id,
            reliable=False,
            pose=None,
            reason=CameraPoseUnreliableReason.TILT_OUT_OF_TOLERANCE,
            recovered_height_m=recovered_height_m,
            expected_height_m=mounting_height_m,
            recovered_tilt_deg=recovered_tilt_deg,
            expected_tilt_deg=tilt_deg,
        )

    return CameraPoseValidation(
        camera_id=camera_id,
        reliable=True,
        pose=pose,
        reason=None,
        recovered_height_m=recovered_height_m,
        expected_height_m=mounting_height_m,
        recovered_tilt_deg=recovered_tilt_deg,
        expected_tilt_deg=tilt_deg,
    )


def standing_height_m(
    pose: CameraPose,
    foot_floor_m: tuple[float, float],
    head_px: tuple[float, float],
    *,
    min_height_m: float = 0.9,
    max_height_m: float = 2.2,
    min_ray_sin: float = 0.15,
) -> float | None:
    """Estimate standing height in metres from a floor foot point and head image pixel.

    Back-projects *head_px* to a 3D ray from ``pose.centre_m`` and returns the ``z`` coordinate
    (metres above the floor) of the point on the vertical line through *foot_floor_m* closest to
    that ray. Returns ``None`` if the ray is nearly vertical (camera directly overhead), points
    away from the person, or produces a height outside ``[min_height_m, max_height_m]``.
    """
    fx, fy = foot_floor_m
    u, v = head_px
    if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(u) and math.isfinite(v)):
        return None

    cx, cy = pose.principal_point_px
    ray_cam = np.array(
        [(u - cx) / pose.focal_px, (v - cy) / pose.focal_px, 1.0],
        dtype=np.float64,
    )
    ray_floor = pose.rotation.T @ ray_cam
    ray_norm = float(np.linalg.norm(ray_floor))
    if not math.isfinite(ray_norm) or ray_norm < 1e-12:
        return None
    d = ray_floor / ray_norm

    # sin(angle between ray d and vertical unit vector u_hat = (0, 0, 1)) = hypot(d_x, d_y).
    denom = float(d[0] * d[0] + d[1] * d[1])
    if denom < min_ray_sin * min_ray_sin:
        return None

    # Line 1: P1(z) = (fx, fy, 0) + z * (0, 0, 1)
    # Line 2: P2(s) = C + s * d
    # Vector w0 from ray origin C to vertical line base P1(0).
    w0 = np.array([fx - pose.centre_m[0], fy - pose.centre_m[1], -pose.centre_m[2]])
    b = float(d[2])  # dot(u_hat, d)
    u_dot_w0 = float(w0[2])
    d_dot_w0 = float(np.dot(d, w0))

    s_param = (d_dot_w0 - b * u_dot_w0) / denom
    z_m = (b * d_dot_w0 - u_dot_w0) / denom

    if not (math.isfinite(s_param) and math.isfinite(z_m)):
        return None
    if s_param <= 0.0:
        return None
    if z_m < min_height_m or z_m > max_height_m:
        return None
    return float(z_m)


def standing_height_with_sigma(
    pose: CameraPose,
    foot_floor_m: tuple[float, float],
    foot_cov_m2: npt.NDArray[np.float64] | tuple[float, float, float, float] | None,
    head_px: tuple[float, float],
    *,
    sigma_head_px: float = 4.0,
    sigma_model_m: float = 0.04,
    min_height_m: float = 0.9,
    max_height_m: float = 2.2,
    min_ray_sin: float = 0.15,
) -> tuple[float, float] | None:
    """Estimate standing height and 1-sigma uncertainty in metres via central differences.

    Propagates 2x2 floor-footpoint covariance ``foot_cov_m2`` (m^2) and isotropic head-pixel
    noise ``sigma_head_px`` (px) through :func:`standing_height_m`, then adds ``sigma_model_m``
    (metres) in quadrature for hair, footwear, and unmodelled lens distortion.
    """
    h0 = standing_height_m(
        pose,
        foot_floor_m,
        head_px,
        min_height_m=min_height_m,
        max_height_m=max_height_m,
        min_ray_sin=min_ray_sin,
    )
    if h0 is None:
        return None

    # Allow finite-difference perturbations slightly outside [min_height_m, max_height_m]
    # so points right near the clamp boundary still yield a valid derivative.
    lo_eps = min_height_m - 0.5
    hi_eps = max_height_m + 0.5
    eps_m = 1e-3
    eps_px = 0.5

    def _eval(fx: float, fy: float, u: float, v: float) -> float | None:
        return standing_height_m(
            pose,
            (fx, fy),
            (u, v),
            min_height_m=lo_eps,
            max_height_m=hi_eps,
            min_ray_sin=min_ray_sin * 0.5,
        )

    fx, fy = foot_floor_m
    u, v = head_px

    h_fx_p = _eval(fx + eps_m, fy, u, v)
    h_fx_m = _eval(fx - eps_m, fy, u, v)
    h_fy_p = _eval(fx, fy + eps_m, u, v)
    h_fy_m = _eval(fx, fy - eps_m, u, v)
    h_u_p = _eval(fx, fy, u + eps_px, v)
    h_u_m = _eval(fx, fy, u - eps_px, v)
    h_v_p = _eval(fx, fy, u, v + eps_px)
    h_v_m = _eval(fx, fy, u, v - eps_px)

    if any(val is None for val in (h_fx_p, h_fx_m, h_fy_p, h_fy_m, h_u_p, h_u_m, h_v_p, h_v_m)):
        return None

    dh_dfx = (h_fx_p - h_fx_m) / (2.0 * eps_m)  # type: ignore[operator]
    dh_dfy = (h_fy_p - h_fy_m) / (2.0 * eps_m)  # type: ignore[operator]
    dh_du = (h_u_p - h_u_m) / (2.0 * eps_px)  # type: ignore[operator]
    dh_dv = (h_v_p - h_v_m) / (2.0 * eps_px)  # type: ignore[operator]

    var_m2 = sigma_model_m * sigma_model_m + (sigma_head_px * sigma_head_px) * (
        dh_du * dh_du + dh_dv * dh_dv
    )

    if foot_cov_m2 is not None:
        cov = np.asarray(foot_cov_m2, dtype=np.float64).reshape(2, 2)
        if np.all(np.isfinite(cov)):
            j_f = np.array([dh_dfx, dh_dfy], dtype=np.float64)
            foot_var = float(j_f @ cov @ j_f)
            if foot_var > 0.0 and math.isfinite(foot_var):
                var_m2 += foot_var

    if not math.isfinite(var_m2) or var_m2 <= 0.0:
        return None
    return (h0, math.sqrt(var_m2))


def is_upright_standing(
    pose: PoseResult | None,
    bbox: BoundingBox,
    *,
    frame_width_px: int,
    frame_height_px: int,
    top_margin_px: int = 4,
    edge_margin_frac: float = 0.1,
    max_torso_tilt_deg: float = 25.0,
    min_aspect_ratio: float = 1.3,
    min_keypoint_score: float = 0.3,
) -> bool:
    """Gate whether one detection represents a non-truncated upright standing person.

    Checks:
    1. Box top is not truncated (``y_min > top_margin_px``) and box bottom is inside the frame.
    2. Horizontal centre lies inside the central ``(1 - 2 * edge_margin_frac)`` band of the
       image to avoid wide-angle barrel distortion near image edges.
    3. Box height-to-width aspect ratio clears ``min_aspect_ratio``.
    4. RTMPose keypoints have visible shoulders, hips, and at least one ankle with
       ``shoulder_y < hip_y < ankle_y`` and torso angle from vertical ``<= max_torso_tilt_deg``.
    """
    if pose is None or frame_width_px <= 0 or frame_height_px <= 0:
        return False

    if bbox.y_min <= top_margin_px:
        return False
    if bbox.y_max >= frame_height_px - top_margin_px:
        return False

    if edge_margin_frac > 0.0:
        margin_x = edge_margin_frac * frame_width_px
        cx = (bbox.x_min + bbox.x_max) / 2.0
        if cx < margin_x or cx > (frame_width_px - margin_x):
            return False

    box_w = bbox.x_max - bbox.x_min
    box_h = bbox.y_max - bbox.y_min
    if box_w <= 0 or box_h <= 0:
        return False
    if (box_h / box_w) < min_aspect_ratio:
        return False

    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    lh = pose.get("left_hip")
    rh = pose.get("right_hip")
    la = pose.get("left_ankle")
    ra = pose.get("right_ankle")

    if min(ls.score, rs.score, lh.score, rh.score) < min_keypoint_score:
        return False

    visible_ankles = [kp for kp in (la, ra) if kp.score >= min_keypoint_score]
    if not visible_ankles:
        return False

    shoulder_x = (ls.x + rs.x) / 2.0
    shoulder_y = (ls.y + rs.y) / 2.0
    hip_x = (lh.x + rh.x) / 2.0
    hip_y = (lh.y + rh.y) / 2.0
    ankle_y = sum(kp.y for kp in visible_ankles) / len(visible_ankles)

    # Image y increases downward: shoulders above hips above ankles.
    if not (shoulder_y < hip_y < ankle_y):
        return False

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    torso_tilt_deg = math.degrees(math.atan2(abs(dx), dy))
    return torso_tilt_deg <= max_torso_tilt_deg


def update_ph_height_estimate(
    current_m: float | None,
    current_sigma_m: float | None,
    obs_m: float | None,
    obs_sigma_m: float | None,
    *,
    innovation_gate_sigma: float = 3.0,
    min_sigma_m: float = 0.02,
    default_sigma_m: float = 0.08,
) -> tuple[float | None, float | None, bool]:
    """Update a Person Hypothesis running height via inverse-variance fusion with a 3-sigma gate.

    Returns ``(new_height_m, new_sigma_m, accepted)``. When *obs_m* is ``None`` or rejected by
    the innovation gate, returns the existing estimate unchanged with ``accepted=False``.
    """
    if obs_m is None or not math.isfinite(obs_m):
        return (current_m, current_sigma_m, False)

    s_obs = (
        obs_sigma_m
        if (obs_sigma_m is not None and math.isfinite(obs_sigma_m) and obs_sigma_m > 0.0)
        else default_sigma_m
    )

    if current_m is None or not math.isfinite(current_m):
        return (float(obs_m), max(min_sigma_m, float(s_obs)), True)

    s_cur = (
        current_sigma_m
        if (
            current_sigma_m is not None and math.isfinite(current_sigma_m) and current_sigma_m > 0.0
        )
        else default_sigma_m
    )

    s_innov = math.hypot(s_cur, s_obs)
    if abs(obs_m - current_m) > innovation_gate_sigma * s_innov:
        return (current_m, s_cur, False)

    prec_cur = 1.0 / (s_cur * s_cur)
    prec_obs = 1.0 / (s_obs * s_obs)
    prec_new = prec_cur + prec_obs
    new_m = (prec_cur * current_m + prec_obs * obs_m) / prec_new
    new_sigma = max(min_sigma_m, math.sqrt(1.0 / prec_new))
    return (new_m, new_sigma, True)
