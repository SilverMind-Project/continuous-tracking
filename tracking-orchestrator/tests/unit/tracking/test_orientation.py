"""Orientation estimation and view prototype update tests."""

from __future__ import annotations

from app.domain import OrientationBin, ViewPrototype
from app.inference.schemas import Keypoint, PoseResult
from app.tracking.orientation import estimate_body_orientation, update_view_prototypes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kp(x: float, y: float, score: float) -> Keypoint:
    return Keypoint(x=x, y=y, score=score)


def _pose_result(keypoints: list[tuple[float, float, float]]) -> PoseResult:
    """Build a PoseResult from a list of (x, y, score) tuples (17 required)."""
    assert len(keypoints) == 17
    return PoseResult(keypoints=tuple(Keypoint(x=x, y=y, score=s) for x, y, s in keypoints))


# COCO-17 keypoint order:
# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle

_HIDDEN = (0.0, 0.0, 0.0)
_FRONT_FACE = [
    (0.50, 0.30, 0.8),  # nose
    (0.48, 0.28, 0.7),  # left_eye
    (0.52, 0.28, 0.7),  # right_eye
    (0.46, 0.30, 0.6),  # left_ear
    (0.54, 0.30, 0.6),  # right_ear
    (0.40, 0.50, 0.9),  # left_shoulder
    (0.60, 0.50, 0.9),  # right_shoulder
]
_BODY = [
    _HIDDEN,  # left_elbow
    _HIDDEN,  # right_elbow
    _HIDDEN,  # left_wrist
    _HIDDEN,  # right_wrist
    (0.42, 0.70, 0.8),  # left_hip
    (0.58, 0.70, 0.8),  # right_hip
    _HIDDEN,  # left_knee
    _HIDDEN,  # right_knee
    _HIDDEN,  # left_ankle
    _HIDDEN,  # right_ankle
]


# ---------------------------------------------------------------------------
# estimate_body_orientation tests
# ---------------------------------------------------------------------------


def test_front_facing() -> None:
    """Clear front: high facial kp confidence, wide shoulders."""
    kps = _FRONT_FACE + _BODY
    ori, conf = estimate_body_orientation(_pose_result(kps).keypoints, 0.3)
    assert ori == OrientationBin.FRONT
    assert conf > 0.5


def test_back_facing() -> None:
    """Clear back: low facial kp confidence, wide shoulders."""
    # Same shoulder positions but all facial kps invisible.
    face = [
        (0.50, 0.30, 0.05),  # nose (low)
        (0.48, 0.28, 0.05),  # left_eye
        (0.52, 0.28, 0.05),  # right_eye
        (0.46, 0.30, 0.05),  # left_ear
        (0.54, 0.30, 0.05),  # right_ear
        (0.40, 0.50, 0.9),  # left_shoulder
        (0.60, 0.50, 0.9),  # right_shoulder
    ]
    kps = face + _BODY
    ori, conf = estimate_body_orientation(_pose_result(kps).keypoints, 0.3)
    assert ori == OrientationBin.BACK
    assert conf > 0.2  # lower because facial kp scores are low


def test_left_profile() -> None:
    """Left profile: shoulders near-colinear, left ear visible."""
    face = [
        (0.48, 0.30, 0.05),  # nose (low)
        (0.46, 0.28, 0.05),  # left_eye
        (0.48, 0.28, 0.05),  # right_eye
        (0.45, 0.30, 0.8),  # left_ear (visible)
        (0.49, 0.30, 0.05),  # right_ear (invisible)
        (0.47, 0.50, 0.9),  # left_shoulder (near-colinear)
        (0.48, 0.50, 0.9),  # right_shoulder
    ]
    kps = face + _BODY
    ori, conf = estimate_body_orientation(_pose_result(kps).keypoints, 0.3)
    assert ori == OrientationBin.LEFT
    assert conf > 0.4


def test_right_profile() -> None:
    """Right profile: shoulders near-colinear, right ear visible."""
    face = [
        (0.52, 0.30, 0.05),  # nose
        (0.50, 0.28, 0.05),  # left_eye
        (0.52, 0.28, 0.05),  # right_eye
        (0.49, 0.30, 0.05),  # left_ear (invisible)
        (0.55, 0.30, 0.8),  # right_ear (visible)
        (0.52, 0.50, 0.9),  # left_shoulder
        (0.53, 0.50, 0.9),  # right_shoulder
    ]
    kps = face + _BODY
    ori, conf = estimate_body_orientation(_pose_result(kps).keypoints, 0.3)
    assert ori == OrientationBin.RIGHT
    assert conf > 0.4


def test_too_few_keypoints_unknown() -> None:
    """All keypoints invisible → UNKNOWN with low confidence."""
    all_hidden = [_HIDDEN] * 17
    ori, conf = estimate_body_orientation(_pose_result(all_hidden).keypoints, 0.3)
    assert ori == OrientationBin.UNKNOWN
    assert conf < 0.3


def test_profile_signed_shoulder_order_left() -> None:
    """When both ears have similar visibility, signed shoulder order
    disambiguates: left shoulder farther right → LEFT profile."""
    face = [
        (0.50, 0.30, 0.05),  # nose
        _HIDDEN,
        _HIDDEN,  # eyes
        (0.47, 0.30, 0.3),  # left_ear (moderate)
        (0.47, 0.30, 0.3),  # right_ear (similar — delta < 0.05)
        (0.52, 0.50, 0.9),  # left_shoulder (farther right → LEFT)
        (0.48, 0.50, 0.9),  # right_shoulder
    ]
    kps = face + _BODY
    ori, _ = estimate_body_orientation(_pose_result(kps).keypoints, 0.3)
    assert ori == OrientationBin.LEFT


# ---------------------------------------------------------------------------
# update_view_prototypes tests
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING: list[float] = [0.0] * 768
_FAKE_EMBEDDING[0] = 1.0  # unit vector for simplicity


def _normalize(emb: list[float]) -> list[float]:
    import numpy as np

    arr = np.asarray(emb, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr.tolist()


def test_create_new_prototype() -> None:
    """First observation with sufficient confidence creates a prototype."""
    emb = _normalize(_FAKE_EMBEDDING)
    result = update_view_prototypes(
        (),
        OrientationBin.FRONT,
        emb,
        orientation_confidence=0.8,
    )
    assert len(result) == 1
    assert result[0].orientation == OrientationBin.FRONT
    assert result[0].count == 1


def test_ema_into_existing_bin() -> None:
    """Second observation of same orientation EMAs into existing prototype."""
    emb1 = _normalize([1.0] + [0.0] * 767)
    emb2 = _normalize([0.0, 1.0] + [0.0] * 766)

    proto1 = update_view_prototypes((), OrientationBin.FRONT, emb1, orientation_confidence=0.8)
    proto2 = update_view_prototypes(proto1, OrientationBin.FRONT, emb2, orientation_confidence=0.8)

    assert len(proto2) == 1
    assert proto2[0].orientation == OrientationBin.FRONT
    assert proto2[0].count == 2
    # The EMA should be between the two embeddings.
    import numpy as np

    emb_result = np.asarray(proto2[0].embedding, dtype=np.float32)
    emb_avg = np.asarray(_normalize([0.5, 0.5] + [0.0] * 766), dtype=np.float32)
    # With alpha=0.5 (1/(1+1)) and renormalization, should be between.
    sim = float(np.dot(emb_result, emb_avg))
    assert sim > 0.9


def test_ignore_low_confidence_orientation() -> None:
    """Low-confidence orientation does NOT create a new prototype."""
    emb = _normalize(_FAKE_EMBEDDING)
    result = update_view_prototypes(
        (),
        OrientationBin.BACK,
        emb,
        orientation_confidence=0.1,
        min_confidence=0.3,
    )
    assert len(result) == 0


def test_ignore_unknown_orientation() -> None:
    """UNKNOWN orientation never creates a prototype."""
    emb = _normalize(_FAKE_EMBEDDING)
    result = update_view_prototypes(
        (),
        OrientationBin.UNKNOWN,
        emb,
        orientation_confidence=0.9,
    )
    assert len(result) == 0


def test_respects_max_prototypes_cap() -> None:
    """When the cap is reached, the lowest-count prototype is evicted."""
    emb = _normalize(_FAKE_EMBEDDING)
    prototypes: tuple[ViewPrototype, ...] = ()
    # Create FRONT prototype with count=5.
    for _ in range(5):
        prototypes = update_view_prototypes(
            prototypes, OrientationBin.FRONT, emb, orientation_confidence=0.8
        )
    assert prototypes[0].count == 5

    # Create BACK prototype with count=1 (will be evicted later).
    emb2 = _normalize([0.0, 1.0] + [0.0] * 766)
    prototypes = update_view_prototypes(
        prototypes, OrientationBin.BACK, emb2, orientation_confidence=0.8
    )
    assert len(prototypes) == 2

    # Create LEFT prototype with max_prototypes=2 → BACK (count=1) evicted.
    emb3 = _normalize([0.0, 0.0, 1.0] + [0.0] * 765)
    prototypes = update_view_prototypes(
        prototypes,
        OrientationBin.LEFT,
        emb3,
        orientation_confidence=0.8,
        max_prototypes=2,
    )
    assert len(prototypes) == 2
    orientations = {p.orientation for p in prototypes}
    assert OrientationBin.FRONT in orientations
    assert OrientationBin.LEFT in orientations  # BACK was evicted


def test_none_embedding_is_noop() -> None:
    """None embedding returns existing prototypes unchanged."""
    emb = _normalize(_FAKE_EMBEDDING)
    proto1 = update_view_prototypes((), OrientationBin.FRONT, emb, orientation_confidence=0.8)
    proto2 = update_view_prototypes(proto1, OrientationBin.FRONT, None, orientation_confidence=0.8)
    assert proto1 == proto2
