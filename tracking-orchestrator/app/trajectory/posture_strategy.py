"""PostureStrategy protocol — pluggable posture detection interface.

All posture detectors implement this protocol. The pipeline uses them through
this interface so new detection modalities (depth, thermal, mmWave) can be
added without changing pipeline code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..domain import Detection, PostureType
from ..inference.schemas import PoseResult
from .posture import PostureScores, classify_posture, score_posture


@runtime_checkable
class PostureStrategy(Protocol):
    """Infer posture from a single detected person."""

    async def infer(
        self,
        frame: npt.NDArray[np.uint8],
        detection: Detection,
        pose_result: PoseResult | None = None,
    ) -> PostureType:
        """Return a PostureLabel for this detection.

        Must never raise — return "unknown" on any inference failure.
        """
        ...

    async def score(
        self,
        frame: npt.NDArray[np.uint8],
        detection: Detection,
        pose_result: PoseResult | None = None,
    ) -> PostureScores:
        """Return soft evidence scores for this detection.

        Must never raise — return PostureScores(0.0, 0.0, 0.0) on any failure.
        """
        ...

    def evict_tracklet(self, tracklet_id: str) -> None:
        """Called when a tracklet closes so strategies can free cache state."""
        ...

    @property
    def name(self) -> str:
        """Short identifier for logging and metrics, e.g. 'rtmpose', 'depth'."""
        ...


class RTMPosePostureStrategy:
    """Wraps the existing keypoint-based posture classifier.

    This is the fast-path strategy — runs every frame.
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "rtmpose"

    async def infer(
        self,
        frame: npt.NDArray[np.uint8],
        detection: Detection,
        pose_result: PoseResult | None = None,
    ) -> PostureType:
        try:
            if pose_result is None:
                return "unknown"
            return classify_posture(pose_result)
        except Exception:  # noqa: BLE001
            return "unknown"

    async def score(
        self,
        frame: npt.NDArray[np.uint8],
        detection: Detection,
        pose_result: PoseResult | None = None,
    ) -> PostureScores:
        try:
            if pose_result is None:
                return PostureScores(lying=0.0, sitting=0.0, standing_walking=0.0)
            return score_posture(pose_result)
        except Exception:  # noqa: BLE001
            return PostureScores(lying=0.0, sitting=0.0, standing_walking=0.0)

    def evict_tracklet(self, tracklet_id: str) -> None:
        """No-op — RTMPose has no per-tracklet cache."""
        pass
