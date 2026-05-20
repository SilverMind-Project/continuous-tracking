"""Frame batching — buffer frames, group by camera, flush for parallel processing.

The tracker state machine is per-camera and must see frames in order.
This batcher buffers incoming frames for a configurable time window,
groups them by camera_id, then flushes each camera's frames sequentially
while running *different* cameras concurrently via ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..observability import metrics

if TYPE_CHECKING:
    from typing import Callable, Coroutine

    from ..transport.redis_streams import FrameReady


class FrameBatcherProtocol(Protocol):
    """Abstract interface for frame batching behaviour."""

    async def push(self, frame: "FrameReady") -> None:
        """Push a frame into the batcher. May flush internally."""

    async def flush(self) -> None:
        """Flush any remaining buffered frames."""


@dataclass
class _CameraBuffer:
    """Mutable buffer for a single camera's frames."""

    frames: list["FrameReady"] = field(default_factory=list)
    last_update: float = field(default_factory=time.monotonic)


@dataclass
class FrameBatcher:
    """Buffer frames for *batch_window_s*, group by camera, flush.

    Parameters
    ----------
    batch_window_s:
        Time window (seconds) to accumulate frames before flushing.
        Must be in ``[0.1, 2.0]``.
    max_batch_size:
        Maximum total frames to accumulate before forcing a flush.
        Must be in ``[1, 16]``.
    handler:
        Async callable invoked per camera with its batch of frames.
        Receives ``(camera_id, frames)`` where *frames* is a list
        sorted by ``frame_index`` (ascending).
    """

    handler: "Callable[[str, list[FrameReady]], Coroutine[None, None, None]]"

    batch_window_s: float = 0.5
    max_batch_size: int = 4

    _cameras: dict[str, _CameraBuffer] = field(default_factory=dict)
    _closed: bool = False
    _flush_task: asyncio.Task[None] | None = field(default=None)

    def __post_init__(self) -> None:
        # Clamp / validate config.
        self._batch_window_s = max(0.1, min(2.0, float(self.batch_window_s)))
        self._max_batch_size = max(1, min(16, int(self.max_batch_size)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push(self, frame: "FrameReady") -> None:
        """Push a frame into the batcher.

        If the batch exceeds ``max_batch_size`` after this push, the
        batcher flushes *immediately* (no need to wait for the timer).
        """
        if self._closed:
            return

        buf = self._cameras.get(frame.camera_id)
        if buf is None:
            buf = _CameraBuffer(frames=[], last_update=time.monotonic())
            self._cameras[frame.camera_id] = buf
        buf.frames.append(frame)
        buf.last_update = time.monotonic()

        total = sum(len(b.frames) for b in self._cameras.values())
        metrics.metrics.batch_size_metric.observe(total)

        if total >= self._max_batch_size:
            # Flush immediately — don't wait for the timer.
            await self.flush()
        else:
            # Defer flush so more frames can accumulate.
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._delayed_flush())

    async def flush(self) -> None:
        """Flush all buffered frames.

        Each camera's frames are processed sequentially (by camera),
        but *different* cameras are processed concurrently via
        ``asyncio.gather``.
        """
        if self._closed:
            return

        # Cancel any pending delayed flush.
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None

        cameras = dict(self._cameras)
        if not cameras:
            return

        # Clear buffers immediately so new pushes start fresh.
        self._cameras.clear()

        # Sort each camera's frames by frame_index.
        batches: list[tuple[str, list["FrameReady"]]] = []
        for cam_id, buf in cameras.items():
            buf.frames.sort(key=lambda f: f.frame_index)
            batches.append((cam_id, buf.frames))

        if not batches:
            return

        # Run each camera's handler concurrently.
        coros = [
            self.handler(cam_id, frames) for cam_id, frames in batches
        ]
        await asyncio.gather(*coros, return_exceptions=True)

    async def close(self) -> None:
        """Flush remaining frames and mark the batcher as closed."""
        if self._cameras:
            await self.flush()
        self._closed = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _delayed_flush(self) -> None:
        """Wait for the batch window, then flush."""
        try:
            await asyncio.sleep(self._batch_window_s)
        except asyncio.CancelledError:
            return
        await self.flush()
