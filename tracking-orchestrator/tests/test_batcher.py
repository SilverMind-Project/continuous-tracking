"""Tests for the frame batcher (frame batching for parallel cross-camera processing)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from app.pipeline.batcher import FrameBatcher

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeFrame:
    """Minimal protobuf-like frame for testing."""

    def __init__(self, camera_id: str, frame_index: int) -> None:
        self.camera_id = camera_id
        self.frame_index = frame_index


@pytest.fixture()
def fake_frame() -> Callable[[str, int], _FakeFrame]:
    """Factory for fake frames."""
    _idx = 0

    def _make(camera_id: str = "cam1", frame_index: int | None = None) -> _FakeFrame:
        nonlocal _idx
        if frame_index is None:
            frame_index = _idx
            _idx += 1
        return _FakeFrame(camera_id=camera_id, frame_index=frame_index)

    return _make


# ---------------------------------------------------------------------------
# FrameBatcher tests
# ---------------------------------------------------------------------------


class TestFrameBatcher:
    """Tests for the FrameBatcher class."""

    @pytest.fixture()
    async def batcher(
        self,
    ) -> Callable[[], FrameBatcher]:
        """Factory that creates a FrameBatcher and cleans it up."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        batcher = FrameBatcher(
            batch_window_s=0.1,
            max_batch_size=4,
            handler=handler,
        )
        yield batcher
        await batcher.close()

    @pytest.mark.asyncio
    async def test_immediate_flush_on_max_batch_size(
        self, fake_frame: Callable[[], _FakeFrame]
    ) -> None:
        """When total frames >= max_batch_size, flush should happen immediately."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        batcher = FrameBatcher(
            batch_window_s=2.0,  # Long window so timer doesn't fire
            max_batch_size=3,
            handler=handler,
        )

        # Push 3 frames from different cameras — should trigger immediate flush.
        f1 = fake_frame("cam1", 1)
        f2 = fake_frame("cam2", 2)
        f3 = fake_frame("cam3", 3)
        await batcher.push(f1)
        await batcher.push(f2)
        await batcher.push(f3)

        # Give the flush task a chance to run.
        await asyncio.sleep(0.05)

        assert len(handlers) == 3
        # Each camera should have exactly 1 frame.
        handler_cameras = {h[0] for h in handlers}
        assert handler_cameras == {"cam1", "cam2", "cam3"}

        await batcher.close()

    @pytest.mark.asyncio
    async def test_timer_flush(self, fake_frame: Callable[[], _FakeFrame]) -> None:
        """When batch_window_s elapses, buffered frames should flush."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        batcher = FrameBatcher(
            batch_window_s=0.05,  # Short window
            max_batch_size=10,  # Large so timer fires first
            handler=handler,
        )

        f1 = fake_frame("cam1", 1)
        f2 = fake_frame("cam2", 2)
        await batcher.push(f1)
        await batcher.push(f2)

        # Wait for the timer to fire.
        await asyncio.sleep(0.15)

        assert len(handlers) == 2
        handler_cameras = {h[0] for h in handlers}
        assert handler_cameras == {"cam1", "cam2"}

        await batcher.close()

    @pytest.mark.asyncio
    async def test_per_camera_sequential_ordering(self, fake_frame: Callable[[], _FakeFrame]) -> (
        None
    ):
        """Frames from the same camera should be processed in frame_index order."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        batcher = FrameBatcher(
            batch_window_s=0.05,
            max_batch_size=10,
            handler=handler,
        )

        # Push frames out of order for cam1.
        await batcher.push(fake_frame("cam1", 3))
        await batcher.push(fake_frame("cam1", 1))
        await batcher.push(fake_frame("cam1", 2))

        await asyncio.sleep(0.15)

        assert len(handlers) == 1
        assert handlers[0][0] == "cam1"
        indices = [f.frame_index for f in handlers[0][1]]
        assert indices == [1, 2, 3]

        await batcher.close()

    @pytest.mark.asyncio
    async def test_cross_camera_parallel_via_gather(self, fake_frame: Callable[[], _FakeFrame]) -> (
        None
    ):
        """Different cameras should be processed concurrently (via asyncio.gather)."""
        order: list[str] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            order.append(camera_id)
            # Simulate some work.
            await asyncio.sleep(0.05)
            order.append(f"{camera_id}-done")

        batcher = FrameBatcher(
            batch_window_s=2.0,  # Long window
            max_batch_size=3,  # Flush immediately on 3 frames
            handler=handler,
        )

        f1 = fake_frame("cam1", 1)
        f2 = fake_frame("cam2", 2)
        f3 = fake_frame("cam3", 3)
        await batcher.push(f1)
        await batcher.push(f2)
        await batcher.push(f3)

        await asyncio.sleep(0.3)

        # Check that all cameras started before any finished (parallelism).
        cam1_done = order.index("cam1-done") if "cam1-done" in order else len(order)
        cam2_done = order.index("cam2-done") if "cam2-done" in order else len(order)
        cam3_done = order.index("cam3-done") if "cam3-done" in order else len(order)
        # At least one camera should have started before all finished.
        first_start = min(order.index(c) for c in ["cam1", "cam2", "cam3"])
        min_done = min(cam1_done, cam2_done, cam3_done)
        assert first_start < min_done, "Cameras should have run in parallel"

        await batcher.close()

    @pytest.mark.asyncio
    async def test_close_flushes_remaining(self, fake_frame: Callable[[], _FakeFrame]) -> None:
        """Closing the batcher should flush any remaining buffered frames."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        batcher = FrameBatcher(
            batch_window_s=10.0,  # Very long window
            max_batch_size=100,  # Large
            handler=handler,
        )

        await batcher.push(fake_frame("cam1", 1))
        await batcher.push(fake_frame("cam2", 2))

        # Close should flush.
        await batcher.close()

        assert len(handlers) == 2

    @pytest.mark.asyncio
    async def test_push_after_close_is_noop(self, fake_frame: Callable[[], _FakeFrame]) -> None:
        """Pushing to a closed batcher should be a no-op."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        batcher = FrameBatcher(
            batch_window_s=0.05,
            max_batch_size=2,
            handler=handler,
        )

        await batcher.push(fake_frame("cam1", 1))
        await asyncio.sleep(0.15)  # Let timer fire.
        await batcher.close()

        # Push after close should be ignored.
        await batcher.push(fake_frame("cam1", 2))
        old_len = len(handlers)
        await asyncio.sleep(0.1)
        assert len(handlers) == old_len

    @pytest.mark.asyncio
    async def test_batch_window_clamping(self) -> None:
        """batch_window_s should be clamped to [0.1, 2.0]."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        # Below minimum.
        batcher = FrameBatcher(
            batch_window_s=0.01,
            max_batch_size=100,
            handler=handler,
        )
        assert batcher._batch_window_s == 0.1

        # Above maximum.
        batcher2 = FrameBatcher(
            batch_window_s=10.0,
            max_batch_size=100,
            handler=handler,
        )
        assert batcher2._batch_window_s == 2.0

        await batcher.close()
        await batcher2.close()

    @pytest.mark.asyncio
    async def test_max_batch_size_clamping(self) -> None:
        """max_batch_size should be clamped to [1, 16]."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        # Below minimum.
        batcher = FrameBatcher(
            batch_window_s=0.05,
            max_batch_size=0,
            handler=handler,
        )
        assert batcher._max_batch_size == 1

        # Above maximum.
        batcher2 = FrameBatcher(
            batch_window_s=0.05,
            max_batch_size=100,
            handler=handler,
        )
        assert batcher2._max_batch_size == 16

        await batcher.close()
        await batcher2.close()

    @pytest.mark.asyncio
    async def test_delayed_flush_cancellation(self, fake_frame: Callable[[], _FakeFrame]) -> (
        None
    ):
        """A new push should cancel any pending delayed flush."""
        handlers: list[tuple[str, list[_FakeFrame]]] = []

        async def handler(camera_id: str, frames: list[_FakeFrame]) -> None:
            handlers.append((camera_id, list(frames)))

        batcher = FrameBatcher(
            batch_window_s=0.2,  # Long enough to test cancellation
            max_batch_size=10,
            handler=handler,
        )

        # Push frame 1 — starts a delayed flush timer.
        await batcher.push(fake_frame("cam1", 1))

        # Push frame 2 — should cancel the previous delayed flush
        # and start a new one.
        await batcher.push(fake_frame("cam2", 2))

        # Wait longer than the first timer but shorter than the second.
        await asyncio.sleep(0.35)

        # Should have flushed once (from the second push).
        assert len(handlers) == 2

        await batcher.close()
