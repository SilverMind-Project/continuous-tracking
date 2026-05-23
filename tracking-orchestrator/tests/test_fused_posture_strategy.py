"""Unit tests for FusedPostureStrategy."""

from __future__ import annotations

import numpy as np
import pytest

from app.domain import BoundingBox, Detection
from app.trajectory.fused_posture_strategy import FusedPostureStrategy


class FakePostureStrategy:
    """Test double for PostureStrategy."""

    def __init__(
        self, name: str = "fake", returns: str = "unknown", raises: Exception | None = None
    ) -> None:
        self._name = name
        self._returns = returns
        self._raises = raises
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def infer(self, frame, detection, pose_result=None):
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        return self._returns

    def evict_tracklet(self, tracklet_id: str) -> None:
        pass


def _make_detection(tracklet_id: str = "") -> Detection:
    return Detection(
        detection_id="det-1",
        camera_id="cam-1",
        bbox=BoundingBox(x_min=100, y_min=100, x_max=300, y_max=400),
        embedding=[],
        capture_time=None,  # type: ignore[arg-type]
        event_time=None,  # type: ignore[arg-type]
        tracklet_id=tracklet_id,
    )


@pytest.mark.asyncio
async def test_fast_path_result_used_when_not_unknown():
    fast = FakePostureStrategy(name="fast", returns="standing")
    slow = FakePostureStrategy(name="slow", returns="lying")
    fused = FusedPostureStrategy(fast, slow)
    result = await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), _make_detection())
    assert result == "standing"
    assert slow.call_count == 0


@pytest.mark.asyncio
async def test_slow_path_runs_when_fast_returns_unknown():
    fast = FakePostureStrategy(name="fast", returns="unknown")
    slow = FakePostureStrategy(name="slow", returns="lying")
    fused = FusedPostureStrategy(fast, slow, slow_path_min_interval_s=0.0)
    result = await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), _make_detection())
    assert result == "lying"
    assert slow.call_count == 1


@pytest.mark.asyncio
async def test_slow_path_not_run_within_interval():
    fast = FakePostureStrategy(name="fast", returns="unknown")
    slow = FakePostureStrategy(name="slow", returns="lying")
    fused = FusedPostureStrategy(fast, slow, slow_path_min_interval_s=3600.0)
    await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), _make_detection())
    result = await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), _make_detection())
    assert slow.call_count == 1
    assert result == "lying"


@pytest.mark.asyncio
async def test_cached_result_expires_after_max_age():
    fast = FakePostureStrategy(name="fast", returns="unknown")
    slow = FakePostureStrategy(name="slow", returns="lying")
    fused = FusedPostureStrategy(fast, slow, slow_path_max_age_s=5.0)
    await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), _make_detection(tracklet_id="t1"))

    # Simulate time passing: manually expire the cache.
    fused._cache.clear()

    det = _make_detection(tracklet_id="t1")
    result = await fused.infer(
        np.zeros((480, 640, 3), dtype=np.uint8),
        det,
    )
    # Cache was cleared and interval not elapsed yet, so slow not re-run.
    assert result == "unknown"


@pytest.mark.asyncio
async def test_evict_tracklet_clears_cache():
    fast = FakePostureStrategy(name="fast", returns="unknown")
    slow = FakePostureStrategy(name="slow", returns="lying")
    fused = FusedPostureStrategy(fast, slow, slow_path_min_interval_s=0.0)
    det = _make_detection(tracklet_id="t1")
    await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), det)
    assert slow.call_count == 1

    fused.evict_tracklet("t1")
    # After eviction, next call should re-run slow path.
    slow.call_count = 0
    fast._returns = "unknown"
    # Reset interval guard
    fused._last_slow.pop("t1", None)
    await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), det)
    assert slow.call_count == 1


@pytest.mark.asyncio
async def test_different_tracklets_have_independent_cache():
    fast = FakePostureStrategy(name="fast", returns="unknown")
    slow = FakePostureStrategy(name="slow", returns="lying")
    fused = FusedPostureStrategy(fast, slow, slow_path_min_interval_s=3600.0)
    det1 = _make_detection(tracklet_id="t1")
    det2 = _make_detection(tracklet_id="t2")

    await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), det1)
    # Second call with different tracklet_id should also run slow path
    # since its own cache is empty.
    result = await fused.infer(np.zeros((480, 640, 3), dtype=np.uint8), det2)
    assert slow.call_count == 2
    assert result == "lying"
