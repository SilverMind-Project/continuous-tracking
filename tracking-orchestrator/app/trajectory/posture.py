"""Posture classifier from RTMPose COCO-17 keypoints.

Architecture
------------
Feature extraction is separated from classification so each stage is independently
testable and the classification rules remain legible.

  1. ``_extract_features`` — geometric features derived from the skeleton, normalised
     by torso length so thresholds are scale- and camera-distance-invariant.
  2. Three scorer functions — ``_score_lying``, ``_score_sitting``,
     ``_score_standing_or_walking`` — each return a float in [0, 1].  Soft scoring
     lets multiple signals accumulate evidence rather than firing on any single
     hard threshold, which reduces false positives on ambiguous poses (e.g. a
     person leaning forward is not immediately classified as sitting just because
     the torso is tilted).
  3. ``classify_posture`` picks the highest-scoring class; falls back to
     ``"unknown"`` when no class clears the evidence floor.
  4. ``PostureHysteresis`` — stateful temporal smoother (unchanged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from ..domain import BoundingBox, PostureType
from ..inference.schemas import Keypoint, PoseResult
from ..observability import metrics as _metrics

# ── Keypoint confidence floor ─────────────────────────────────────────────────
_SCORE_FLOOR = 0.3

# ── Geometric thresholds (in degrees) ─────────────────────────────────────────
_TORSO_HORIZONTAL_ONSET_DEG = 60.0  # torso above this angle from vertical → could be lying
_SEATED_TORSO_ANGLE_MIN_DEG = 30.0  # torso tilted beyond this → sitting signal
_KNEE_ANGLE_SITTING_MIN_DEG = 60.0  # knee bent beyond this → sitting signal
_KNEE_ANGLE_SITTING_MAX_DEG = 150.0  # wider than the naive 90° range: 2D projection
# from overhead/front cameras foreshortens the knee,
# making a true 90° bend appear as 130-150° in image space.
_KNEE_BENT_GUARD_MAX_DEG = 130.0  # conservative upper bound for the standing hard-veto;
# kept below _KNEE_ANGLE_SITTING_MAX_DEG so a walking
# person with a mildly bent stride knee (135°) is not
# misclassified as "unknown" by the guard.

# ── Normalised-skeleton thresholds ────────────────────────────────────────────
# Distances are expressed in torso-length units after normalising hip-midpoint
# to the origin and scaling by shoulder-to-hip distance.
_HEAD_TORSO_DEVIATION_MAX = 0.5  # nose within this fraction of torso length → lying
_KNEE_HIP_PROXIMITY_MAX = 0.5  # knee within this fraction of torso → near-hip (sitting)
_KNEE_HIP_STANDING_BLOCK = 0.4  # norm_knee_dy below this → block standing (knees near hips
# are incompatible with an upright standing posture)

# ── Composite evidence floor ──────────────────────────────────────────────────
_MIN_EVIDENCE = 0.5  # minimum scorer output to commit a posture class

# ── Motion threshold ──────────────────────────────────────────────────────────
_WALKING_VELOCITY_THRESHOLD = 0.008  # mean keypoint velocity (normalised px/frame)

# ── Multi-camera fusion ───────────────────────────────────────────────────────
_CAMERA_STALE_AFTER_S: float = 10.0
"""Seconds after which a camera's last-seen posture score is dropped from fusion.

When a person walks out of a camera's field of view, their last score from that
camera expires after this many seconds so it does not continue influencing the
fused result indefinitely.
"""

_DEPTH_WEIGHT: float = 0.15
"""Floor weight given to depth-only cameras (keypoint_confidence == 0.0).

Prevents depth estimates from being completely ignored when keypoint cameras are
present, while still giving keypoint cameras proportionally higher weight.
"""


# ── Primitive helpers ─────────────────────────────────────────────────────────


def _midpoint(a: Keypoint, b: Keypoint) -> tuple[float, float]:
    return (a.x + b.x) / 2.0, (a.y + b.y) / 2.0


def _visible(*keypoints: Keypoint) -> bool:
    return all(k.score >= _SCORE_FLOOR for k in keypoints)


def _min_score(*keypoints: Keypoint) -> float:
    return min(k.score for k in keypoints) if keypoints else 0.0


# ── Raw geometric measures ────────────────────────────────────────────────────


def _torso_angle_deg(pose: PoseResult) -> float | None:
    """Angle of the torso vector (shoulder-midpoint → hip-midpoint) from vertical.

    0° = vertical, 90° = horizontal.  Uses ``abs(dx)`` so forward and backward
    lean both read as tilt away from vertical — the classifier cares about
    magnitude, not direction.
    """
    ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
    lh, rh = pose.get("left_hip"), pose.get("right_hip")
    if not _visible(ls, rs, lh, rh):
        return None
    sx, sy = _midpoint(ls, rs)
    hx, hy = _midpoint(lh, rh)
    dx, dy = hx - sx, hy - sy
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _knee_angle_deg(pose: PoseResult, side: str) -> float | None:
    """Angle at the knee joint: 0° = leg straight, ~90° = right-angle bend."""
    hip = pose.get(f"{side}_hip")
    knee = pose.get(f"{side}_knee")
    ankle = pose.get(f"{side}_ankle")
    if not _visible(hip, knee, ankle):
        return None
    tx, ty = knee.x - hip.x, knee.y - hip.y
    sx, sy = ankle.x - knee.x, ankle.y - knee.y
    dot = tx * sx + ty * sy
    norm_t = math.hypot(tx, ty)
    norm_s = math.hypot(sx, sy)
    if norm_t == 0 or norm_s == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (norm_t * norm_s)))))


def _best_knee_angle_deg(pose: PoseResult) -> tuple[float, float] | None:
    """(min_knee_angle_deg, keypoint_confidence) for the most bent visible leg."""
    left = _knee_angle_deg(pose, "left")
    right = _knee_angle_deg(pose, "right")

    def _score(side: str) -> float:
        return _min_score(
            pose.get(f"{side}_hip"), pose.get(f"{side}_knee"), pose.get(f"{side}_ankle")
        )

    if left is not None and right is not None:
        if left <= right:
            return (left, _score("left"))
        return (right, _score("right"))
    if left is not None:
        return (left, _score("left"))
    if right is not None:
        return (right, _score("right"))
    return None


def _head_torso_deviation(pose: PoseResult) -> float | None:
    """Lateral displacement of the nose from the torso midline, normalised by torso length.

    A lying person's head is roughly in line with the torso (small value).
    A standing or sitting person's head is well above the shoulders (large value).
    """
    nose = pose.get("nose")
    ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
    lh, rh = pose.get("left_hip"), pose.get("right_hip")
    if not _visible(nose, ls, rs, lh, rh):
        return None
    sx, sy = _midpoint(ls, rs)
    hx, hy = _midpoint(lh, rh)
    torso_len = math.hypot(hx - sx, hy - sy)
    if torso_len == 0:
        return None
    mid_y = (sy + hy) / 2.0
    return abs(nose.y - mid_y) / torso_len


# ── Feature dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PostureFeatures:
    """Scale-normalised geometric features derived from one skeleton frame.

    All distance features are expressed in torso-length units (shoulder-to-hip
    distance = 1.0), making the thresholds in the scorer functions independent
    of camera distance and subject size.
    """

    torso_angle_deg: float | None
    """0° = vertical spine, 90° = fully horizontal.  None when shoulders/hips invisible."""

    min_knee_angle_deg: float | None
    """Angle at the least-bent visible knee (straightest leg).  None when neither leg is visible."""

    max_knee_angle_deg: float | None
    """Angle at the most-bent visible knee.  None when neither leg is visible."""

    knee_confidence: float
    """Min keypoint score of the best knee triplet; 0.0 when no knees are visible."""

    norm_knee_dy: float | None
    """(knee_mid_y - hip_mid_y) / torso_len.  Positive = knees below hips (image coords)."""

    norm_knee_dx: float | None
    """abs(knee_mid_x - hip_mid_x) / torso_len.  Large when thighs spread laterally (chair
    sitting from front), even when ankles are occluded.  None when hips/knees invisible."""

    norm_ankle_dy: float | None
    """(ankle_mid_y - hip_mid_y) / torso_len.  Combined with norm_knee_dy, encodes the
    shin-drop geometry that distinguishes sitting (knees near hips, ankles hanging below)
    from standing (both well below) and lying (both near hips)."""

    head_spine_deviation: float | None
    """Vertical displacement of nose from torso centre, normalised by torso length."""

    kinematic_ordering: bool
    """True when knee_y > hip_y (and ankle_y > knee_y when ankles are visible)."""


# COCO-17 keypoint names used for mean confidence calculation.
_ALL_KP_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _mean_visible_confidence(pose: PoseResult) -> float:
    """Mean score of keypoints that clear the confidence floor.

    Returns 0.0 when no keypoint is visible above _SCORE_FLOOR.
    """
    scores = [
        kp.score
        for name in _ALL_KP_NAMES
        if (kp := pose.get(name)) is not None and kp.score >= _SCORE_FLOOR
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _score_upper_torso_lying(pose: PoseResult) -> float:
    """Lying evidence from upper body only, for use when hips/legs are occluded.

    Detects the 'visible shoulders, occluded hips' pattern which occurs when a
    person lies under bed sheets. Returns a conservative score in [0, 0.6] so
    it does not override a standing or sitting signal from cameras with full
    body visibility.

    Signals used:
    1. Both shoulders visible, both hips invisible — the sheet-occlusion hallmark.
    2. Shoulder line is horizontal: abs(left_shoulder.y - right_shoulder.y) is small
       relative to the inter-shoulder distance. A standing/sitting person has a
       roughly horizontal shoulder line too, but their hips are also visible.
    3. Head (nose/eyes) is roughly level with the shoulders in image y-coordinates.
       A standing person's head is clearly above the shoulders.

    Returns 0.0 when hips are visible (the full-body scorer handles this case)
    or when shoulder keypoints are unavailable.
    """
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    lh = pose.get("left_hip")
    rh = pose.get("right_hip")

    # Hips visible → use full-body scorer, not this one.
    if _visible(lh, rh):
        return 0.0

    # Shoulders must both be visible.
    if not _visible(ls, rs):
        return 0.0

    score = 0.0

    # Signal 1: shoulder line is horizontal.
    inter_shoulder_dist = math.hypot(rs.x - ls.x, rs.y - ls.y)
    if inter_shoulder_dist > 0:
        vertical_lean = abs(ls.y - rs.y) / inter_shoulder_dist
        # vertical_lean near 0 → nearly horizontal shoulder line → lying signal.
        score += 0.3 * max(0.0, 1.0 - vertical_lean / 0.3)

    # Signal 2: head (nose) is at approximately the same image height as shoulders.
    nose = pose.get("nose")
    if nose is not None and nose.score >= _SCORE_FLOOR and inter_shoulder_dist > 0:
        shoulder_y = (ls.y + rs.y) / 2.0
        head_above_shoulders = (shoulder_y - nose.y) / inter_shoulder_dist
        # head_above_shoulders near 0 → head is not above shoulders → lying signal.
        # For a standing person, the head is clearly above the shoulders (> 0.4 units).
        head_signal = max(0.0, 1.0 - head_above_shoulders / 0.4)
        score += 0.3 * head_signal

    return min(0.6, score)


def _extract_features(pose: PoseResult) -> PostureFeatures:
    """Derive ``PostureFeatures`` from raw COCO-17 keypoints."""
    torso_angle = _torso_angle_deg(pose)

    # Torso-length normalisation for knee and ankle heights.
    ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
    lh, rh = pose.get("left_hip"), pose.get("right_hip")
    norm_knee_dy: float | None = None
    norm_knee_dx: float | None = None
    norm_ankle_dy: float | None = None
    if _visible(ls, rs, lh, rh):
        sx, sy = _midpoint(ls, rs)
        hx, hy = _midpoint(lh, rh)
        torso_len = math.hypot(hx - sx, hy - sy)
        if torso_len > 0:
            lk, rk = pose.get("left_knee"), pose.get("right_knee")
            if _visible(lk, rk):
                norm_knee_dy = ((lk.y + rk.y) / 2.0 - hy) / torso_len
                # Lateral spread: half the inter-knee x-distance, normalised by torso length.
                # Wide lateral spread with knees near hip height = thighs horizontal
                # (chair-sitting from front camera), visible even when ankles are occluded.
                norm_knee_dx = abs(lk.x - rk.x) / 2.0 / torso_len
            la, ra = pose.get("left_ankle"), pose.get("right_ankle")
            if _visible(la, ra):
                norm_ankle_dy = ((la.y + ra.y) / 2.0 - hy) / torso_len

    # Knee angles.
    left_angle = _knee_angle_deg(pose, "left")
    right_angle = _knee_angle_deg(pose, "right")

    def _side_conf(side: str) -> float:
        return _min_score(
            pose.get(f"{side}_hip"), pose.get(f"{side}_knee"), pose.get(f"{side}_ankle")
        )

    left_conf = _side_conf("left") if left_angle is not None else 0.0
    right_conf = _side_conf("right") if right_angle is not None else 0.0

    min_knee_angle_deg: float | None = None
    max_knee_angle_deg: float | None = None
    knee_confidence: float = 0.0

    if left_angle is not None and right_angle is not None:
        if left_angle <= right_angle:
            min_knee_angle_deg = left_angle
            max_knee_angle_deg = right_angle
            knee_confidence = right_conf
        else:
            min_knee_angle_deg = right_angle
            max_knee_angle_deg = left_angle
            knee_confidence = left_conf
    elif left_angle is not None:
        min_knee_angle_deg = left_angle
        max_knee_angle_deg = left_angle
        knee_confidence = left_conf
    elif right_angle is not None:
        min_knee_angle_deg = right_angle
        max_knee_angle_deg = right_angle
        knee_confidence = right_conf

    # Head-spine deviation (lying detection).
    head_dev = _head_torso_deviation(pose)

    # Kinematic ordering: ankle > knee > hip in image y (downward).
    lk2, rk2 = pose.get("left_knee"), pose.get("right_knee")
    la, ra = pose.get("left_ankle"), pose.get("right_ankle")
    lh2, rh2 = pose.get("left_hip"), pose.get("right_hip")
    kinematic_ordering = False
    if _visible(lh2, rh2, lk2, rk2):
        hip_y = (lh2.y + rh2.y) / 2.0
        knee_y = (lk2.y + rk2.y) / 2.0
        if knee_y > hip_y:
            kinematic_ordering = (la.y + ra.y) / 2.0 > knee_y if _visible(la, ra) else True

    return PostureFeatures(
        torso_angle_deg=torso_angle,
        min_knee_angle_deg=min_knee_angle_deg,
        max_knee_angle_deg=max_knee_angle_deg,
        knee_confidence=knee_confidence,
        norm_knee_dy=norm_knee_dy,
        norm_knee_dx=norm_knee_dx,
        norm_ankle_dy=norm_ankle_dy,
        head_spine_deviation=head_dev,
        kinematic_ordering=kinematic_ordering,
    )


@dataclass(frozen=True)
class PostureScores:
    """Soft evidence for each posture class, normalised to [0, 1].

    Produced by a PostureStrategy and consumed by GlobalPostureTracker for
    multi-camera fusion. Preserves continuous evidence so fusion can
    accumulate across cameras rather than picking a single winner early.
    """

    lying: float
    """Evidence that the person is lying. 0.0 = no evidence, 1.0 = certain."""

    sitting: float
    """Evidence that the person is sitting."""

    standing_walking: float
    """Combined evidence for standing or walking (motion_energy separates them
    later during fusion)."""

    keypoint_confidence: float = 0.0
    """Mean confidence of visible COCO-17 keypoints. 0.0 when no keypoints are
    available (e.g., depth-only estimate). Used by quality-weighted fusion."""


@dataclass(frozen=True)
class _CameraSnapshot:
    """A single camera's posture score contribution at a point in time."""

    lying: float
    sitting: float
    standing_walking: float
    keypoint_confidence: float
    captured_at: datetime


# ── Soft evidence scorers ─────────────────────────────────────────────────────


def _score_lying(feats: PostureFeatures) -> float:
    """Evidence that the person is lying down.

    Requires the torso to be substantially horizontal *and* the head to be
    roughly in line with the torso (not elevated above the shoulders as in all
    upright postures).  Both signals must co-occur — a tilted torso alone (e.g.
    someone reclined in a chair) is suppressed by a large head-spine deviation.
    """
    if feats.torso_angle_deg is None:
        return 0.0
    # Grows from 0 at the lying onset threshold to 1.0 at fully horizontal.
    horizontal = max(
        0.0,
        (feats.torso_angle_deg - _TORSO_HORIZONTAL_ONSET_DEG)
        / (90.0 - _TORSO_HORIZONTAL_ONSET_DEG),
    )
    if horizontal == 0.0:
        return 0.0
    if feats.head_spine_deviation is None:
        head_aligned = 0.5  # uncertain — give partial credit
    else:
        head_aligned = max(0.0, 1.0 - feats.head_spine_deviation / _HEAD_TORSO_DEVIATION_MAX)
    return horizontal * head_aligned


def _score_sitting(feats: PostureFeatures) -> float:
    """Evidence that the person is sitting.

    Five signals contribute additively:

    **Torso tilt** — tilting the trunk forward increases sitting evidence.

    **Knee bend** — a bent knee (55-150°) adds evidence.  We look at the most
    bent knee (max_knee_angle_deg) to handle cases where one leg is extended.
    Higher confidence is required when the torso provides no tilt signal.

    **Horizontal thigh / Knee proximity** — when knees are near hip height vertically,
    combined with knee bend, indicating the thigh is flat. This resolves sitting
    for upright sitters when ankles are occluded.

    **Lateral knee spread** — ankles-free signal: knees far apart laterally with
    knees near hip height indicates thighs horizontal (front-facing chair sit).
    Requires only visible hips + knees, no ankle visibility needed.

    **Shin drop** — the most camera-angle-invariant sitting cue: knees near hip
    height *and* ankles hanging below.
    """
    torso_tilt = feats.torso_angle_deg or 0.0
    score = 0.0

    # 1. Torso-tilt contribution: 0 at 30°, caps at 0.4 at 60°.
    tilt = max(0.0, min(0.4, 0.4 * (torso_tilt - _SEATED_TORSO_ANGLE_MIN_DEG) / 30.0))
    score += tilt

    # 2. Knee-bend contribution (using the most bent knee).
    if feats.max_knee_angle_deg is not None:
        ka = feats.max_knee_angle_deg
        if 55.0 <= ka <= _KNEE_ANGLE_SITTING_MAX_DEG:
            # Require stronger confidence when torso evidence is absent.
            knee_conf_threshold = 0.3 if torso_tilt > _SEATED_TORSO_ANGLE_MIN_DEG else 0.5
            score += 0.4 * min(1.0, feats.knee_confidence / knee_conf_threshold)

    # 3. Horizontal thighs: knees near hip height + bent knee angle.
    if (
        feats.norm_knee_dy is not None
        and feats.norm_knee_dy < _KNEE_HIP_PROXIMITY_MAX
        and feats.max_knee_angle_deg is not None
        and feats.max_knee_angle_deg >= 55.0
    ):
        score += 0.4 * min(1.0, feats.knee_confidence / 0.4)

    # 4. Lateral knee spread: knees far apart horizontally + knees near hip height.
    # Captures front-facing chair sit even when ankles are fully occluded.
    # norm_knee_dx > 0.6 means knees are at least 0.6 torso-lengths from hip midpoint.
    if (
        feats.norm_knee_dy is not None
        and feats.norm_knee_dy < _KNEE_HIP_PROXIMITY_MAX
        and feats.norm_knee_dx is not None
        and feats.norm_knee_dx > 0.6
    ):
        score += 0.5 * min(1.0, feats.norm_knee_dx / 1.0)

    # 5. Shin-drop leg geometry: knees near hip height, ankles hanging below.
    if (
        feats.knee_confidence >= 0.4
        and feats.norm_knee_dy is not None
        and feats.norm_knee_dy < _KNEE_HIP_PROXIMITY_MAX
        and feats.norm_ankle_dy is not None
        and feats.norm_ankle_dy > feats.norm_knee_dy + 0.4
    ):
        shin_drop = feats.norm_ankle_dy - feats.norm_knee_dy
        score += 0.5 * min(1.0, shin_drop / 0.6)

    # 6. Weak corroborating signal: knees at hip height when torso is already tilted.
    if (
        feats.norm_knee_dy is not None
        and abs(feats.norm_knee_dy) < _KNEE_HIP_PROXIMITY_MAX
        and torso_tilt > _SEATED_TORSO_ANGLE_MIN_DEG
    ):
        tilt_factor = (torso_tilt - _SEATED_TORSO_ANGLE_MIN_DEG) / 60.0
        score += 0.15 * min(1.0, tilt_factor)

    return min(1.0, score)


def _score_standing_or_walking(feats: PostureFeatures) -> float:
    """Evidence that the person is standing (or walking).

    Two hard vetos suppress this class:

    **Bent knees** — uses the conservative ``_KNEE_BENT_GUARD_MAX_DEG`` (130°).
    Vetoes standing if BOTH legs are bent (which indicates sitting or squatting).
    Since min_knee_angle_deg is the straightest leg, if min_knee_angle_deg >= 55.0,
    then both legs are bent.

    **Knees near hips** — norm_knee_dy < 0.55 means the knees are close to the
    hips vertically, which is physically incompatible with upright standing.
    """
    knees_bent = (
        feats.min_knee_angle_deg is not None
        and 55.0 <= feats.min_knee_angle_deg <= _KNEE_BENT_GUARD_MAX_DEG
    )
    knees_near_hips = feats.norm_knee_dy is not None and feats.norm_knee_dy < 0.55
    if knees_bent or knees_near_hips:
        return 0.0

    score = 0.0
    if feats.torso_angle_deg is not None and feats.torso_angle_deg < _SEATED_TORSO_ANGLE_MIN_DEG:
        score += 0.5
    if feats.kinematic_ordering:
        score += 0.5
    return min(1.0, score)


# ── Public classifier ─────────────────────────────────────────────────────────


def score_posture(
    pose: PoseResult,
    bedroom_prior: float = 0.0,
) -> PostureScores:
    """Compute soft evidence scores from COCO-17 keypoints.

    Args:
        pose:           COCO-17 keypoints with (x, y, score) per point.
        bedroom_prior:  Additional prior weight added to the lying score. Use
                        to implement a context-sensitive bedroom lying prior
                        in GlobalPostureTracker when the room is a bedroom and
                        the body is heavily occluded. Default 0.0 (no prior).
    """
    feats = _extract_features(pose)
    lying = _score_lying(feats)
    sitting = _score_sitting(feats)
    standing_walking = _score_standing_or_walking(feats)

    # Upper-torso-only path: when full-body lying cannot be scored because hips are
    # occluded, use the shoulder-only signal. Cap it at 0.6 so it never overrides a
    # full-body signal from another camera.
    if lying == 0.0 and standing_walking == 0.0:
        partial_lying = _score_upper_torso_lying(pose)
        lying = min(1.0, lying + partial_lying + bedroom_prior)
    else:
        # Full body is visible — the bedroom prior still adds small context evidence.
        lying = min(1.0, lying + bedroom_prior * 0.3)

    return PostureScores(
        lying=lying,
        sitting=sitting,
        standing_walking=standing_walking,
        keypoint_confidence=_mean_visible_confidence(pose),
    )


def classify_posture(
    pose: PoseResult,
    bbox: BoundingBox,
    motion_energy: float | None = None,
) -> PostureType:
    """Classify posture from COCO-17 keypoints and optional motion energy.

    Delegates to score_posture() and resolves to a PostureType label.
    Public API preserved for callers that need a label directly.
    """
    scores = score_posture(pose)
    best = max(scores.lying, scores.sitting, scores.standing_walking)
    if best < _MIN_EVIDENCE:
        return "unknown"
    if scores.lying >= scores.sitting and scores.lying >= scores.standing_walking:
        return "lying"
    if scores.sitting >= scores.standing_walking:
        return "sitting"
    if motion_energy is not None and motion_energy > _WALKING_VELOCITY_THRESHOLD:
        return "walking"
    return "standing"


# ── Temporal smoother ─────────────────────────────────────────────────────────


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
                self._state[track_id] = (raw, raw, count)
                return raw
            self._state[track_id] = (committed, candidate, count)
            return committed
        else:
            self._state[track_id] = (committed, raw, 1)
            return committed

    def evict(self, track_id: str) -> None:
        """Remove state for a closed track."""
        self._state.pop(track_id, None)


class GlobalPostureTracker:
    """Stateful posture tracker that aggregates evidence across cameras and smooths transitions.

    Maintains:
    1. Per-camera posture score snapshots with timestamps, so stale cameras are
       automatically excluded from fusion.
    2. Quality-weighted fusion: cameras with higher keypoint confidence contribute
       proportionally more to the fused score.
    3. Hysteresis state stored directly (no per-GT PostureHysteresis objects) to
       avoid the dict-of-objects-with-one-entry indirection.
    """

    def __init__(
        self,
        required_consecutive: int = 2,
        camera_stale_after_s: float = _CAMERA_STALE_AFTER_S,
        depth_weight: float = _DEPTH_WEIGHT,
    ) -> None:
        # global_track_id → camera_id → _CameraSnapshot
        self._snapshots: dict[str, dict[str, _CameraSnapshot]] = {}
        # global_track_id → (committed, candidate, consecutive_count)
        self._hysteresis_state: dict[str, tuple[PostureType, PostureType, int]] = {}
        self._required_consecutive = required_consecutive
        self._camera_stale_after_s = camera_stale_after_s
        self._depth_weight = depth_weight

    def update(
        self,
        global_track_id: str,
        camera_id: str,
        scores: PostureScores,
        active_camera_ids: list[str] | tuple[str, ...] | set[str],
        motion_energy: float | None = None,
    ) -> PostureType:
        """Update posture scores for a track on one camera; return smoothed global posture.

        Args:
            global_track_id:   The global track being updated.
            camera_id:         The camera this frame came from.
            scores:            Soft posture evidence from PostureStrategy.score().
            active_camera_ids: All cameras currently associated with this track.
            motion_energy:     Normalised keypoint velocity. Used to distinguish
                               walking from standing at resolve time.
        """
        now = datetime.now(UTC)

        # 1. Store this camera's latest snapshot.
        if global_track_id not in self._snapshots:
            self._snapshots[global_track_id] = {}
        self._snapshots[global_track_id][camera_id] = _CameraSnapshot(
            lying=scores.lying,
            sitting=scores.sitting,
            standing_walking=scores.standing_walking,
            keypoint_confidence=scores.keypoint_confidence,
            captured_at=now,
        )
        _metrics.metrics.cts_posture_camera_contributions_total.labels(
            camera_id=camera_id,
        ).inc()

        # 2. Quality-weighted fusion across active cameras.
        fused = self._fuse(global_track_id, active_camera_ids, now)

        # 3. Resolve the winner using clinical priority rules.
        raw_posture = self._resolve(fused, motion_energy)

        # 4. Apply hysteresis (state stored directly — no PostureHysteresis indirection).
        return self._apply_hysteresis(global_track_id, raw_posture)

    def _fuse(
        self,
        global_track_id: str,
        active_camera_ids: list[str] | tuple[str, ...] | set[str],
        now: datetime,
    ) -> dict[str, float]:
        """Compute quality-weighted average posture scores across active cameras.

        Only cameras whose snapshot is fresh (within camera_stale_after_s) and are
        in active_camera_ids contribute. Cameras with keypoint_confidence == 0.0
        (e.g., depth-only) receive a floor weight of _depth_weight.
        """
        total_weight = 0.0
        acc = {"lying": 0.0, "sitting": 0.0, "standing_walking": 0.0}
        gt_snapshots = self._snapshots.get(global_track_id, {})

        for cam in active_camera_ids:
            snap = gt_snapshots.get(cam)
            if snap is None:
                continue
            age_s = (now - snap.captured_at).total_seconds()
            if age_s > self._camera_stale_after_s:
                continue
            # Floor so depth cameras (keypoint_confidence == 0) still contribute.
            weight = max(snap.keypoint_confidence, self._depth_weight)
            acc["lying"] += weight * snap.lying
            acc["sitting"] += weight * snap.sitting
            acc["standing_walking"] += weight * snap.standing_walking
            total_weight += weight

        active_count = sum(
            1
            for cam in active_camera_ids
            if gt_snapshots.get(cam) is not None
            and (now - gt_snapshots[cam].captured_at).total_seconds() <= self._camera_stale_after_s
        )
        _metrics.metrics.cts_posture_cameras_fused.observe(active_count)

        if total_weight == 0.0:
            return {"lying": 0.0, "sitting": 0.0, "standing_walking": 0.0}

        return {c: acc[c] / total_weight for c in acc}

    def _resolve(
        self,
        fused: dict[str, float],
        motion_energy: float | None,
    ) -> PostureType:
        """Map fused soft scores to a PostureType label using clinical priority."""
        best = max(fused["lying"], fused["sitting"], fused["standing_walking"])
        if best < _MIN_EVIDENCE:
            raw: PostureType = "unknown"
        elif fused["lying"] >= fused["sitting"] and fused["lying"] >= fused["standing_walking"]:
            raw = "lying"
        elif fused["sitting"] >= fused["standing_walking"]:
            raw = "sitting"
        elif motion_energy is not None and motion_energy > _WALKING_VELOCITY_THRESHOLD:
            raw = "walking"
        else:
            raw = "standing"
        _metrics.metrics.cts_posture_fused_class_total.labels(posture=raw).inc()
        return raw

    def _apply_hysteresis(
        self,
        global_track_id: str,
        raw: PostureType,
    ) -> PostureType:
        """Apply N-consecutive-frame smoothing.

        State stored directly as (committed, candidate, count) tuple — no per-GT
        PostureHysteresis object needed.
        """
        entry = self._hysteresis_state.get(global_track_id)
        if entry is None:
            self._hysteresis_state[global_track_id] = (raw, raw, 1)
            return raw

        committed, candidate, count = entry
        if raw == candidate:
            count += 1
            if count >= self._required_consecutive:
                self._hysteresis_state[global_track_id] = (raw, raw, count)
                if committed != raw:
                    _metrics.metrics.cts_posture_hysteresis_flips_total.labels(
                        camera_id="global",
                    ).inc()
                return raw
            self._hysteresis_state[global_track_id] = (committed, candidate, count)
            return committed

        self._hysteresis_state[global_track_id] = (committed, raw, 1)
        return committed

    def committed_posture(self, global_track_id: str) -> PostureType | None:
        """Return the currently committed posture for a global track, or None if unknown."""
        entry = self._hysteresis_state.get(global_track_id)
        return entry[0] if entry is not None else None

    def evict_track(self, global_track_id: str) -> None:
        """Evict all state for a closed track. No-op if the track is unknown."""
        self._snapshots.pop(global_track_id, None)
        self._hysteresis_state.pop(global_track_id, None)

    def clear(self) -> None:
        """Clear all tracking state (used in tests and on pipeline restart)."""
        self._snapshots.clear()
        self._hysteresis_state.clear()
