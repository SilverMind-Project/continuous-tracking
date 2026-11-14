"""Stage contract and runner for the frame processing pipeline."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from structlog import get_logger

from ...observability import metrics as _metrics
from ..frame_context import FrameContext

logger = get_logger(__name__)


@contextmanager
def timed_phase(stage_name: str, camera_id: str) -> Iterator[None]:
    """Observe ``cts_stage_latency_ms`` for a pipeline stage or direct phase call."""
    start = time.monotonic()
    try:
        yield
    finally:
        latency_ms = (time.monotonic() - start) * 1000.0
        _metrics.metrics.stage_latency_ms.labels(stage=stage_name, camera_id=camera_id).observe(
            latency_ms
        )


class FrameStage(Protocol):
    """A single stage in the frame processing pipeline."""

    name: str

    async def run(self, ctx: FrameContext) -> None:
        """Execute this stage against *ctx*, mutating it in place."""
        ...


class StageRunner:
    """Runs a list of stages sequentially, recording per-stage latency."""

    def __init__(self, stages: list[FrameStage]) -> None:
        self._stages = stages

    async def run(self, ctx: FrameContext) -> None:
        for stage in self._stages:
            start = time.monotonic()
            try:
                with timed_phase(stage.name, ctx.frame.camera_id):
                    await stage.run(ctx)
            except Exception:
                latency_ms = (time.monotonic() - start) * 1000.0
                logger.exception(
                    "Stage failed",
                    stage=stage.name,
                    camera_id=ctx.frame.camera_id,
                    frame_index=ctx.frame.frame_index,
                    capture_time_unix_ns=ctx.frame.capture_time_unix_ns,
                    latency_ms=round(latency_ms, 3),
                )
                raise
