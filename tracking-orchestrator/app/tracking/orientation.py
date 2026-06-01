"""Body orientation estimation from RTMPose COCO-17 keypoints.

Pure functions — no I/O, no DB, no model.  Orientation is a geometric
classifier over 2D keypoint positions and visibility scores, designed to
be fast (< 10 µs per call) and trivially testable.

Orientation bins
----------------
- FRONT  — person facing the camera (high facial-keypoint confidence).
- BACK   — person facing away (low facial-keypoint confidence, shoulders
           wide enough to rule out profile).
- LEFT   — left profile (one ear visible, shoulders near-colinear).
- RIGHT  — right profile.
- UNKNOWN — too few keypoints visible for a reliable estimate.

The geometry rules:

1. **Shoulder colinearity gate**: when the horizontal shoulder span
   ``|x_L - x_R|`` is below a fraction of the bbox width, the person is
   in profile (LEFT or RIGHT).  The signed order ``x_L - x_R``
   disambiguates: positive means the left shoulder is farther right
   (person's left side toward camera = LEFT profile).

2. **Front vs back**: when shoulders are wide enough to rule out profile,
   facial-keypoint confidence (nose, eyes, ears) determines FRONT vs BACK.
   High mean confidence → FRONT; low → BACK.

3. **Profile disambiguation**: the ear with higher visibility wins.
   Left ear visible ∧ right ear invisible → LEFT profile; the reverse
   → RIGHT profile.  When both ears have similar visibility, the signed
   shoulder order breaks the tie.

4. **Confidence**: the mean visibility score of the keypoints used in the
   classification.  When the score is below a minimum threshold the bin
   is UNKNOWN.

COCO-17 keypoint indices used
-----------------------------
- 0  nose
- 1  left_eye
- 2  right_eye
- 3  left_ear
- 4  right_ear
- 5  left_shoulder
- 6  right_shoulder
- 11 left_hip
- 12 right_hip
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..domain import OrientationBin, ViewPrototype

if TYPE_CHECKING:
    from ..inference.schemas import Keypoint


# ---------------------------------------------------------------------------
# Tunable constants (module-level, not config — these are geometry thresholds)
# ---------------------------------------------------------------------------

# Minimum mean keypoint visibility to return any orientation.
_MIN_MEAN_VISIBILITY = 0.15

# Shoulder horizontal span (in normalised coords [0,1]) below which the
# person is considered to be in profile.
_PROFILE_SHOULDER_SPAN = 0.06

# Mean facial-keypoint confidence above which a non-profile person is
# classified as FRONT (rather than BACK).
_FRONT_FACE_CONFIDENCE = 0.25

# Minimum ear visibility difference to disambiguate LEFT vs RIGHT profile.
_EAR_CONFIDENCE_DELTA = 0.05

# ---------------------------------------------------------------------------
# Keypoint index constants (COCO-17)
# ---------------------------------------------------------------------------

_KP_NOSE = 0
_KP_LEFT_EYE = 1
_KP_RIGHT_EYE = 2
_KP_LEFT_EAR = 3
_KP_RIGHT_EAR = 4
_KP_LEFT_SHOULDER = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_HIP = 11
_KP_RIGHT_HIP = 12

_FACIAL_INDICES = (_KP_NOSE, _KP_LEFT_EYE, _KP_RIGHT_EYE, _KP_LEFT_EAR, _KP_RIGHT_EAR)


def estimate_body_orientation(
    keypoints: tuple[Keypoint, ...],
    bbox_width: float,
) -> tuple[OrientationBin, float]:
    """Estimate body orientation from COCO-17 keypoints and bbox width.

    Args:
        keypoints: 17 COCO keypoints from RTMPose, each with normalised
            [0,1] x/y coordinates and a visibility score in [0,1].
        bbox_width: Width of the detection bounding box in normalised
            image coordinates [0,1].  Used as the reference scale for
            the shoulder-colinearity gate.

    Returns:
        (orientation, confidence) where orientation is the estimated
        OrientationBin and confidence is the mean visibility of the
        keypoints used.
    """
    if len(keypoints) < 17:
        return OrientationBin.UNKNOWN, 0.0

    # Gather the keypoints we need.
    ls = keypoints[_KP_LEFT_SHOULDER]
    rs = keypoints[_KP_RIGHT_SHOULDER]
    lh = keypoints[_KP_LEFT_HIP]
    rh = keypoints[_KP_RIGHT_HIP]

    shoulder_visible = ls.score > 0.1 and rs.score > 0.1
    hip_visible = lh.score > 0.1 and rh.score > 0.1

    if not shoulder_visible and not hip_visible:
        # Not enough structural keypoints to classify.
        return OrientationBin.UNKNOWN, 0.0

    # Use whichever pair is more confident.
    if shoulder_visible:
        left_x, left_score = ls.x, ls.score
        right_x, right_score = rs.x, rs.score
    else:
        left_x, left_score = lh.x, lh.score
        right_x, right_score = rh.x, rh.score

    shoulder_span = abs(left_x - right_x)
    signed_diff = left_x - right_x  # positive → left kp is farther right in image

    # Scale the shoulder-span threshold by bbox width so it works for
    # both tight and loose crops.  Floor at 0.01 to avoid division noise.
    profile_threshold = max(_PROFILE_SHOULDER_SPAN, bbox_width * 0.06)
    is_profile = shoulder_span < profile_threshold

    # Facial keypoint visibility for front/back disambiguation.
    face_scores = [keypoints[i].score for i in _FACIAL_INDICES]
    mean_face_conf = float(np.mean(face_scores)) if face_scores else 0.0

    if is_profile:
        # Profile: disambiguate LEFT vs RIGHT.
        left_ear_score = keypoints[_KP_LEFT_EAR].score
        right_ear_score = keypoints[_KP_RIGHT_EAR].score
        ear_delta = left_ear_score - right_ear_score

        if ear_delta > _EAR_CONFIDENCE_DELTA:
            orientation = OrientationBin.LEFT
        elif ear_delta < -_EAR_CONFIDENCE_DELTA:
            orientation = OrientationBin.RIGHT
        elif signed_diff > 0:
            # Left shoulder farther right → left side toward camera.
            orientation = OrientationBin.LEFT
        else:
            orientation = OrientationBin.RIGHT

        # Confidence: mean of the shoulder/hip and ear scores.
        key_scores = [left_score, right_score, left_ear_score, right_ear_score]
        confidence = float(np.mean(key_scores))
    else:
        # Non-profile: disambiguate FRONT vs BACK via facial keypoints.
        if mean_face_conf >= _FRONT_FACE_CONFIDENCE:
            orientation = OrientationBin.FRONT
        else:
            orientation = OrientationBin.BACK

        # Confidence: mean of shoulder/hip and facial scores.
        key_scores = [left_score, right_score, *face_scores]
        confidence = float(np.mean(key_scores))

    if confidence < _MIN_MEAN_VISIBILITY:
        return OrientationBin.UNKNOWN, confidence

    return orientation, confidence


# ---------------------------------------------------------------------------
# View prototype update helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gallery seeding helpers
# ---------------------------------------------------------------------------

# Default EMA floor so the prototype stays adaptive even after many observations.
_DEFAULT_EMA_FLOOR = 0.1

# Per-(identity, orientation) gallery seeding cap.
_DEFAULT_MAX_ENTRIES_PER_ORIENTATION = 10


def update_view_prototypes(
    existing: tuple[ViewPrototype, ...],
    orientation: OrientationBin,
    embedding: list[float] | None,
    orientation_confidence: float,
    *,
    min_confidence: float = 0.3,
    ema_floor: float = _DEFAULT_EMA_FLOOR,
    max_prototypes: int = 5,  # one per OrientationBin value
) -> tuple[ViewPrototype, ...]:
    """Update view prototypes with a new observation.

    If a prototype already exists for *orientation*, EMA the new embedding
    into it and increment its count.  Otherwise, if *orientation_confidence*
    clears *min_confidence*, create a new prototype.

    The total number of prototypes is capped at *max_prototypes* (defaults
    to the number of known OrientationBin values).  When the cap would be
    exceeded, the prototype with the lowest count is evicted.

    Args:
        existing: Current prototypes (may be empty).
        orientation: The estimated OrientationBin for this observation.
        embedding: The SOLIDER-REID embedding (768-dim float list).  When
            None, prototypes are returned unchanged.
        orientation_confidence: Confidence of the orientation estimate.
        min_confidence: Minimum confidence to create a new prototype.
        ema_floor: Minimum EMA alpha (prevents prototypes from freezing).
        max_prototypes: Maximum number of prototypes to retain.

    Returns:
        Updated tuple of ViewPrototype instances.
    """
    if embedding is None:
        return existing

    emb_tuple = tuple(float(v) for v in embedding)
    prototypes = list(existing)

    # Find matching prototype.
    for i, p in enumerate(prototypes):
        if p.orientation == orientation:
            # EMA: alpha = 1/(count+1), capped at ema_floor.
            alpha = max(ema_floor, 1.0 / (p.count + 1))
            old = np.asarray(p.embedding, dtype=np.float32)
            new = np.asarray(emb_tuple, dtype=np.float32)
            updated = (1.0 - alpha) * old + alpha * new
            # Re-normalise to unit length.
            norm = float(np.linalg.norm(updated))
            if norm > 1e-8:
                updated = updated / norm
            prototypes[i] = ViewPrototype(
                orientation=orientation,
                embedding=tuple(float(v) for v in updated.tolist()),
                count=p.count + 1,
            )
            return tuple(prototypes)

    # No matching prototype — create one if confidence is sufficient.
    if orientation_confidence < min_confidence or orientation == OrientationBin.UNKNOWN:
        return existing

    if len(prototypes) >= max_prototypes:
        # Evict the lowest-count prototype.
        prototypes.sort(key=lambda p: p.count)
        prototypes.pop(0)

    prototypes.append(
        ViewPrototype(
            orientation=orientation,
            embedding=emb_tuple,
            count=1,
        )
    )
    return tuple(prototypes)
