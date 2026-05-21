"""Fused posture strategy: combines fast-path (RTMPose) and slow-path (Depth).

The fast-path runs every frame. The slow-path runs at most once per
``slow_path_min_interval_s`` seconds per tracklet. The most recent slow-path
result is cached per tracklet.

Fusion rule:
- If fast-path returns anything other than 'unknown', use fast-path.
- If fast-path returns 'unknown' and a slow-path result is cached and
  fresh (within slow_path_max_age_s), use slow-path.
- Otherwise return 'unknown'.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

from ..domain import Detection, PostureType
from ..inference.schemas import PoseResult
from .posture_strategy import PostureStrategy


@dataclass(frozen=True)
class _SlowPathCache:
    label: PostureType
    computed_at: datetime


class FusedPostureStrategy:
    def __init__(
        self,
        fast: PostureStrategy,
        slow: PostureStrategy,
        slow_path_min_interval_s: float = 15.0,
        slow_path_max_age_s: float = 60.0,
    ) -> None:
        self._fast = fast
        self._slow = slow
        self._interval = slow_path_min_interval_s
        self._max_age = slow_path_max_age_s
        self._cache: dict[str, _SlowPathCache] = {}
        self._last_slow: dict[str, datetime] = {}

    @property
    def name(self) -> str:
        return f"fused({self._fast.name}+{self._slow.name})"

    async def infer(
        self,
        frame: npt.NDArray[np.uint8],
        detection: Detection,
        pose_result: PoseResult | None = None,
    ) -> PostureType:
        fast_result = await self._fast.infer(frame, detection, pose_result)

        if fast_result != "unknown":
            return fast_result

        # Fast-path failed — check if slow-path should run.
        cache_key = detection.tracklet_id or detection.detection_id
        now = datetime.now(UTC)

        should_run_slow = (
            cache_key not in self._last_slow
            or (now - self._last_slow[cache_key]).total_seconds() >= self._interval
        )

        if should_run_slow:
            slow_result = await self._slow.infer(frame, detection, None)
            self._last_slow[cache_key] = now
            self._cache[cache_key] = _SlowPathCache(
                label=slow_result, computed_at=now
            )

        cached = self._cache.get(cache_key)
        if cached is None:
            return "unknown"

        age_s = (now - cached.computed_at).total_seconds()
        if age_s > self._max_age:
            return "unknown"

        return cached.label

    def evict_tracklet(self, tracklet_id: str) -> None:
        """Call when a tracklet closes to free memory."""
        self._cache.pop(tracklet_id, None)
        self._last_slow.pop(tracklet_id, None)
