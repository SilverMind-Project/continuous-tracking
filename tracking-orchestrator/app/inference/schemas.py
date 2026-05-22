"""Typed input/output schemas for Triton-backed inference models.

These types sit at the inference layer boundary.  They are distinct from the
domain types in app.domain — they carry raw model outputs that services
translate into domain objects.

``DetectionBox`` is re-exported from ``triton_shared.inference.schemas``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from triton_shared.inference.schemas import DetectionBox

# ---------------------------------------------------------------------------
# SOLIDER-REID output
# ---------------------------------------------------------------------------

#: 768-dim L2-normalised appearance embedding from SOLIDER-REID Swin-Tiny
#: (384x128 input, MSMT17 fine-tuned). Shape: (768,), dtype float32.
Embedding = npt.NDArray[np.float32]

EMBEDDING_DIM = 768

# ---------------------------------------------------------------------------
# RTMPose-m output
# ---------------------------------------------------------------------------

#: COCO 17 keypoint names ordered by index.
COCO_KEYPOINTS: tuple[str, ...] = (
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

NUM_KEYPOINTS = 17


@dataclass(frozen=True)
class Keypoint:
    """Single skeleton keypoint in normalised image coordinates [0, 1]."""

    x: float  # horizontal, left=0 right=1
    y: float  # vertical, top=0 bottom=1
    score: float  # visibility confidence [0, 1]


__all__ = [
    "COCO_KEYPOINTS",
    "EMBEDDING_DIM",
    "NUM_KEYPOINTS",
    "DetectionBox",
    "Embedding",
    "Keypoint",
    "PoseResult",
]


@dataclass(frozen=True)
class PoseResult:
    """17 COCO keypoints for one person crop."""

    keypoints: tuple[Keypoint, ...]  # always len == NUM_KEYPOINTS (17)

    def __post_init__(self) -> None:
        if len(self.keypoints) != NUM_KEYPOINTS:
            raise ValueError(
                f"PoseResult requires {NUM_KEYPOINTS} keypoints, got {len(self.keypoints)}"
            )

    def get(self, name: str) -> Keypoint:
        """Return keypoint by COCO name."""
        idx = COCO_KEYPOINTS.index(name)
        return self.keypoints[idx]
