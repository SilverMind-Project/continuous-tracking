"""Posture classifier from RTMPose COCO-17 keypoints.

Pure function, no I/O.
"""

from __future__ import annotations

import math

from ..domain import BoundingBox, PostureType
from ..inference.schemas import Keypoint, PoseResult

_SCORE_FLOOR = 0.3
_TORSO_HORIZONTAL_MAX_DEG = 30.0
_SEATED_TORSO_ANGLE_MIN_DEG = 30.0
_LYING_AR_THRESHOLD = 1.2


def _midpoint(a: Keypoint, b: Keypoint) -> tuple[float, float]:
    return (a.x + b.x) / 2.0, (a.y + b.y) / 2.0


def _visible(*keypoints: Keypoint) -> bool:
    return all(k.score >= _SCORE_FLOOR for k in keypoints)


def _torso_angle_deg(pose: PoseResult) -> float | None:
    """Angle of the torso vector (shoulder-midpoint → hip-midpoint) from vertical.

    Returns None when required keypoints are below the confidence floor.
    """
    needed = [
        pose.get("left_shoulder"),
        pose.get("right_shoulder"),
        pose.get("left_hip"),
        pose.get("right_hip"),
    ]
    if not _visible(*needed):
        return None
    sx, sy = _midpoint(needed[0], needed[1])
    hx, hy = _midpoint(needed[2], needed[3])
    dx, dy = hx - sx, hy - sy
    if dx == 0 and dy == 0:
        return None
    # Angle from vertical: 0° = vertical, 90° = horizontal.
    # Use |dy| so the sign of dy (upward vs downward) does not affect the result.
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)


def classify_posture(pose: PoseResult, bbox: BoundingBox) -> PostureType:
    """Classify posture from COCO-17 keypoints and bounding box geometry.

    Rules (each uses only keypoints with score ≥ 0.3):
      - lying: torso within ~30° of horizontal, OR bbox aspect ratio > 1.2
        (person appears wide and short in the crop).
      - sitting: hip-to-knee vertical span small relative to shoulder-to-hip span;
        knees roughly level with hips.
      - standing: ankles below knees below hips, near-vertical torso.
      - unknown: insufficient visible keypoints.
    """
    torso_deg = _torso_angle_deg(pose)
    bbox_ar = bbox.width / max(bbox.height, 1)

    # Lying: horizontal torso or wide-short bbox
    if torso_deg is not None and torso_deg > (90.0 - _TORSO_HORIZONTAL_MAX_DEG):
        return "lying"
    if bbox_ar > _LYING_AR_THRESHOLD:
        return "lying"

    # Check keypoints needed for sitting vs standing
    left_knee = pose.get("left_knee")
    right_knee = pose.get("right_knee")
    left_ankle = pose.get("left_ankle")
    right_ankle = pose.get("right_ankle")
    left_hip = pose.get("left_hip")
    right_hip = pose.get("right_hip")

    knees_ok = _visible(left_knee, right_knee)
    ankles_ok = _visible(left_ankle, right_ankle)
    hips_ok = _visible(left_hip, right_hip)

    if hips_ok and knees_ok:
        # Hip-to-knee vertical span
        hip_y_mid = (left_hip.y + right_hip.y) / 2.0
        knee_y_mid = (left_knee.y + right_knee.y) / 2.0
        hip_knee_dy = abs(knee_y_mid - hip_y_mid)

        # Sitting: knees close to hips vertically → small hip-to-knee span
        if torso_deg is not None and torso_deg > _SEATED_TORSO_ANGLE_MIN_DEG:
            return "sitting"
        if hip_knee_dy < 0.12:
            return "sitting"

        # Standing: ankles below knees below hips, near-vertical torso
        if ankles_ok:
            ankle_y_mid = (left_ankle.y + right_ankle.y) / 2.0
            if ankle_y_mid > knee_y_mid and knee_y_mid > hip_y_mid:
                if torso_deg is not None and torso_deg < _SEATED_TORSO_ANGLE_MIN_DEG:
                    return "standing"
                if torso_deg is None:
                    return "standing"

    # Standing fallback: near-vertical torso + visible shoulders/hips
    if torso_deg is not None and torso_deg < _SEATED_TORSO_ANGLE_MIN_DEG:
        return "standing"

    return "unknown"
