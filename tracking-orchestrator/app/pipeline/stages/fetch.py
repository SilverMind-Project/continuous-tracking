"""Fetch stage: loads the RGB frame image from object storage."""

from __future__ import annotations

import numpy as np
from structlog import get_logger

from ..frame_context import FrameContext
from ..types import FrameImageFetcher
from .base import FrameStage

logger = get_logger(__name__)


class FetchStage(FrameStage):
    name = "fetch"

    def __init__(self, frame_fetcher: FrameImageFetcher | None = None) -> None:
        self._frame_fetcher = frame_fetcher

    async def run(self, ctx: FrameContext) -> None:
        if self._frame_fetcher is not None:
            ctx.image = await self._frame_fetcher.fetch_rgb(ctx.frame.minio_key)
        else:
            ctx.image = np.zeros(
                (max(ctx.frame.height, 1), max(ctx.frame.width, 1), 3), dtype=np.uint8
            )
        img_h, img_w = ctx.image.shape[:2]
        ctx.effective_width = ctx.frame.width
        ctx.effective_height = ctx.frame.height
        if img_w != ctx.frame.width or img_h != ctx.frame.height:
            logger.warning(
                "frame_dimension_mismatch",
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
                minio_shape=f"{img_h}x{img_w}",
                reported_shape=f"{ctx.frame.height}x{ctx.frame.width}",
            )
            ctx.effective_width = img_w
            ctx.effective_height = img_h
