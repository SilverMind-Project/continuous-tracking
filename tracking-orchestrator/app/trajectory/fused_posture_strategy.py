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
from .posture import PostureScores
from .posture_strategy import PostureStrategy


@dataclass(frozen=True)
class _SlowPathCache:
    label: PostureType
    computed_at: datetime


@dataclass(frozen=True)
class _SlowPathScoresCache:
    scores: PostureScores
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
        self._scores_cache: dict[str, _SlowPathScoresCache] = {}

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
        cache_key = detection.tracklet_id if detection.tracklet_id != "" else detection.detection_id
        now = datetime.now(UTC)

        should_run_slow = (
            cache_key not in self._last_slow
            or (now - self._last_slow[cache_key]).total_seconds() >= self._interval
        )

        if should_run_slow:
            slow_result = await self._slow.infer(frame, detection, None)
            self._last_slow[cache_key] = now
            self._cache[cache_key] = _SlowPathCache(label=slow_result, computed_at=now)

        cached = self._cache.get(cache_key)
        if cached is None:
            return "unknown"

        age_s = (now - cached.computed_at).total_seconds()
        if age_s > self._max_age:
            return "unknown"

        return cached.label

    async def score(
        self,
        frame: npt.NDArray[np.uint8],
        detection: Detection,
        pose_result: PoseResult | None = None,
    ) -> PostureScores:
        """Return soft evidence scores using fast path; fall back to slow path on unknown.

        Uses the same cache and interval logic as infer(). When the fast path
        produces non-zero scores, the slow path is skipped entirely.
        """
        fast_scores = await self._fast.score(frame, detection, pose_result)

        # If fast path has any evidence, return it immediately.
        fast_max = max(fast_scores.lying, fast_scores.sitting, fast_scores.standing_walking)
        if fast_max > 0.0:
            return fast_scores

        # Fast path has no evidence — check slow path.
        cache_key = detection.tracklet_id if detection.tracklet_id != "" else detection.detection_id
        now = datetime.now(UTC)

        should_run_slow = (
            cache_key not in self._last_slow
            or (now - self._last_slow[cache_key]).total_seconds() >= self._interval
        )

        if should_run_slow:
            slow_scores = await self._slow.score(frame, detection, None)
            self._last_slow[cache_key] = now
            self._scores_cache[cache_key] = _SlowPathScoresCache(
                scores=slow_scores, computed_at=now
            )

        cached = self._scores_cache.get(cache_key)
        if cached is None:
            return PostureScores(lying=0.0, sitting=0.0, standing_walking=0.0)

        age_s = (now - cached.computed_at).total_seconds()
        if age_s > self._max_age:
            return PostureScores(lying=0.0, sitting=0.0, standing_walking=0.0)

        return cached.scores

    def evict_tracklet(self, tracklet_id: str) -> None:
        """Call when a tracklet closes to free memory."""
        self._cache.pop(tracklet_id, None)
        self._last_slow.pop(tracklet_id, None)
        self._scores_cache.pop(tracklet_id, None)
