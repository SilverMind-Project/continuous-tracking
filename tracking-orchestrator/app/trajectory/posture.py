"""Posture classifier from RTMPose COCO-17 keypoints.

Pure functions for per-frame classification, plus a stateful hysteresis
wrapper for temporal smoothing across consecutive frames.
"""

from __future__ import annotations

import math

from ..domain import BoundingBox, PostureType
from ..inference.schemas import Keypoint, PoseResult

_SCORE_FLOOR = 0.3
_TORSO_HORIZONTAL_MAX_DEG = 30.0  # torso within this of horizontal → lying
_SEATED_TORSO_ANGLE_MIN_DEG = 30.0  # torso beyond this from vertical → sitting
_KNEE_ANGLE_SITTING_MIN_DEG = 60.0  # knee bent beyond this → sitting
_KNEE_ANGLE_SITTING_MAX_DEG = 130.0  # beyond this is not a chair sitting posture
_HEAD_TORSO_DEVIATION_MAX = 0.5  # nose within this fraction of torso length → lying
_WALKING_VELOCITY_THRESHOLD = 0.008  # mean keypoint velocity (normalised px/frame)


def _midpoint(a: Keypoint, b: Keypoint) -> tuple[float, float]:
    return (a.x + b.x) / 2.0, (a.y + b.y) / 2.0


def _visible(*keypoints: Keypoint) -> bool:
    return all(k.score >= _SCORE_FLOOR for k in keypoints)


def _min_score(*keypoints: Keypoint) -> float:
    """Minimum confidence among keypoints, used to weight geometric features."""
    return min(k.score for k in keypoints) if keypoints else 0.0


def _torso_angle_deg(pose: PoseResult) -> float | None:
    """Angle of the torso vector (shoulder-midpoint → hip-midpoint) from vertical.

    0° = vertical, 90° = horizontal. Returns None when required keypoints
    are below the confidence floor.
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
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)


def _knee_angle_deg(pose: PoseResult, side: str) -> float | None:
    """Angle at the knee joint for one leg.

    Computes the angle between the thigh vector (hip→knee) and shin vector
    (knee→ankle). 0° = leg straight, ~90° = knee bent at right angle.

    Returns None when required keypoints are below the confidence floor.
    """
    hip = pose.get(f"{side}_hip")
    knee = pose.get(f"{side}_knee")
    ankle = pose.get(f"{side}_ankle")
    if not _visible(hip, knee, ankle):
        return None
    tx, ty = knee.x - hip.x, knee.y - hip.y
    sx, sy = ankle.x - knee.x, ankle.y - knee.y
    dot = tx * sx + ty * sy
    norm_t = math.sqrt(tx * tx + ty * ty)
    norm_s = math.sqrt(sx * sx + sy * sy)
    if norm_t == 0 or norm_s == 0:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (norm_t * norm_s)))
    return math.degrees(math.acos(cos_angle))


def _best_knee_angle_deg(pose: PoseResult) -> tuple[float, float] | None:
    """Return (min_angle, avg_score) for the most informative leg.

    The more bent knee (smaller angle) is the stronger sitting signal.
    """
    left = _knee_angle_deg(pose, "left")
    right = _knee_angle_deg(pose, "right")

    if left is not None and right is not None:
        left_score = _min_score(pose.get("left_hip"), pose.get("left_knee"), pose.get("left_ankle"))
        right_score = _min_score(
            pose.get("right_hip"), pose.get("right_knee"), pose.get("right_ankle")
        )
        # Use the more bent knee (smaller angle) as the signal.
        if left <= right:
            return (left, left_score)
        return (right, right_score)

    if left is not None:
        return (
            left,
            _min_score(pose.get("left_hip"), pose.get("left_knee"), pose.get("left_ankle")),
        )
    if right is not None:
        return (
            right,
            _min_score(pose.get("right_hip"), pose.get("right_knee"), pose.get("right_ankle")),
        )
    return None


def _head_torso_deviation(pose: PoseResult) -> float | None:
    """Vertical deviation of nose from the shoulder-hip midpoint, normalized by torso length.

    A lying person's head is roughly in line with the torso (small deviation).
    A standing/sitting person's head is well above the shoulders (large deviation).
    Returns None when required keypoints are below the confidence floor.
    """
    nose = pose.get("nose")
    shoulders = [pose.get("left_shoulder"), pose.get("right_shoulder")]
    hips = [pose.get("left_hip"), pose.get("right_hip")]
    if not _visible(nose, *shoulders, *hips):
        return None
    sx, sy = _midpoint(shoulders[0], shoulders[1])
    hx, hy = _midpoint(hips[0], hips[1])
    torso_len = math.sqrt((hx - sx) ** 2 + (hy - sy) ** 2)
    if torso_len == 0:
        return None
    # Distance from nose to the torso midline, normalized by torso length.
    # For a lying person (torso horizontal), nose is near the torso line.
    mid_y = (sy + hy) / 2.0
    return abs(nose.y - mid_y) / torso_len


def classify_posture(
    pose: PoseResult,
    bbox: BoundingBox,
    motion_energy: float | None = None,
) -> PostureType:
    """Classify posture from COCO-17 keypoints and optional motion energy.

    Rules (each uses only keypoints with score ≥ 0.3):

      - **lying**: torso near horizontal AND head roughly in line with torso.
      - **sitting**: torso tilted beyond threshold from vertical, OR knee angle
        in the bent range (60-130 degrees). Uses the more bent knee.
      - **walking**: near-vertical torso, ankle-below-knee-below-hip ordering,
        AND motion_energy above the walking threshold. Falls back to
        ``"standing"`` when motion_energy is unavailable.
      - **standing**: ankles below knees below hips, near-vertical torso.
        Fallback when only a vertical torso is visible.
      - **unknown**: insufficient visible keypoints.
    """
    torso_deg = _torso_angle_deg(pose)
    knee_info = _best_knee_angle_deg(pose)

    # -- lying ----------------------------------------------------------------
    if torso_deg is not None and torso_deg > (90.0 - _TORSO_HORIZONTAL_MAX_DEG):
        head_dev = _head_torso_deviation(pose)
        if head_dev is not None and head_dev < _HEAD_TORSO_DEVIATION_MAX:
            return "lying"

    # -- sitting --------------------------------------------------------------
    # Torso tilt beyond threshold.
    if torso_deg is not None and torso_deg > _SEATED_TORSO_ANGLE_MIN_DEG:
        # Knee angle must also be consistent with sitting (bent) — a tilted
        # torso with straight legs is more likely a person leaning, not sitting.
        if knee_info is not None:
            knee_angle, _ = knee_info
            if _KNEE_ANGLE_SITTING_MIN_DEG <= knee_angle <= _KNEE_ANGLE_SITTING_MAX_DEG:
                return "sitting"
        # No knee info but torso is tilted: weak sitting signal.
        left_knee = pose.get("left_knee")
        right_knee = pose.get("right_knee")
        left_hip = pose.get("left_hip")
        right_hip = pose.get("right_hip")
        if _visible(left_hip, right_hip, left_knee, right_knee):
            knee_y_mid = (left_knee.y + right_knee.y) / 2.0
            hip_y_mid = (left_hip.y + right_hip.y) / 2.0
            if abs(knee_y_mid - hip_y_mid) < 0.12:
                return "sitting"

    # Knee angle alone (independent of torso tilt).
    if knee_info is not None:
        knee_angle, knee_score = knee_info
        if _KNEE_ANGLE_SITTING_MIN_DEG <= knee_angle <= _KNEE_ANGLE_SITTING_MAX_DEG:
            # Require stronger evidence when torso isn't also tilted.
            if torso_deg is not None and torso_deg > _SEATED_TORSO_ANGLE_MIN_DEG:
                return "sitting"
            # Knee-only: require higher-confidence keypoints.
            if knee_score >= 0.5:
                return "sitting"

    # -- walking / standing ---------------------------------------------------
    # Guard: if the knees are visibly bent (in the sitting angle range), do
    # not classify as standing or walking regardless of torso angle.  This
    # prevents low-confidence bent-knee sitting poses from being misclassified
    # as standing when the torso happens to be vertical.
    knees_bent = (
        knee_info is not None
        and _KNEE_ANGLE_SITTING_MIN_DEG <= knee_info[0] <= _KNEE_ANGLE_SITTING_MAX_DEG
    )

    left_knee = pose.get("left_knee")
    right_knee = pose.get("right_knee")
    left_ankle = pose.get("left_ankle")
    right_ankle = pose.get("right_ankle")
    left_hip = pose.get("left_hip")
    right_hip = pose.get("right_hip")

    knees_ok = _visible(left_knee, right_knee)
    ankles_ok = _visible(left_ankle, right_ankle)
    hips_ok = _visible(left_hip, right_hip)

    if hips_ok and knees_ok and not knees_bent:
        # Check ankle-below-knee-below-hip ordering.
        hip_y_mid = (left_hip.y + right_hip.y) / 2.0
        knee_y_mid = (left_knee.y + right_knee.y) / 2.0
        has_ordering = knee_y_mid > hip_y_mid
        if ankles_ok:
            ankle_y_mid = (left_ankle.y + right_ankle.y) / 2.0
            has_ordering = ankle_y_mid > knee_y_mid and knee_y_mid > hip_y_mid

        if has_ordering:
            if torso_deg is not None and torso_deg < _SEATED_TORSO_ANGLE_MIN_DEG:
                if motion_energy is not None and motion_energy > _WALKING_VELOCITY_THRESHOLD:
                    return "walking"
                return "standing"
            if torso_deg is None:
                if motion_energy is not None and motion_energy > _WALKING_VELOCITY_THRESHOLD:
                    return "walking"
                return "standing"

    # Standing fallback: near-vertical torso with visible shoulders/hips,
    # and no evidence of bent knees.
    if torso_deg is not None and torso_deg < _SEATED_TORSO_ANGLE_MIN_DEG and not knees_bent:
        if motion_energy is not None and motion_energy > _WALKING_VELOCITY_THRESHOLD:
            return "walking"
        return "standing"

    return "unknown"


class PostureHysteresis:
    """Requires N consecutive frames of a new posture before committing the change.

    Uses a per-track state machine: each track has a *committed* posture and a
    *candidate* posture with a consecutive-frame counter.  A flip only occurs
    once the same candidate has been seen for ``required_consecutive`` frames.
    """

    def __init__(self, required_consecutive: int = 2) -> None:
        self._required = required_consecutive
        # track_id → (committed, candidate, consecutive_count)
        self._state: dict[str, tuple[PostureType, PostureType, int]] = {}

    def update(self, track_id: str, raw: PostureType) -> PostureType:
        """Return the hysteresis-smoothed posture for this track.

        On first observation the raw posture is committed immediately.
        Subsequent flips require ``required_consecutive`` consecutive frames.
        """
        entry = self._state.get(track_id)
        if entry is None:
            self._state[track_id] = (raw, raw, 1)
            return raw

        committed, candidate, count = entry
        if raw == candidate:
            count += 1
            if count >= self._required:
                # Commit the candidate.
                self._state[track_id] = (raw, raw, count)
                return raw
            self._state[track_id] = (committed, candidate, count)
            return committed
        else:
            # New candidate resets the counter but does not flip immediately.
            self._state[track_id] = (committed, raw, 1)
            return committed

    def evict(self, track_id: str) -> None:
        """Remove state for a closed track."""
        self._state.pop(track_id, None)
