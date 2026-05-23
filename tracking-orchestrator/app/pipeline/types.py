"""Shared pipeline types imported by both ``frame_pipeline`` and ``stages/``.

These live in their own module to avoid a circular import between the
pipeline orchestrator and the individual stage classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..inference.schemas import Embedding


class FrameImageFetcher(Protocol):
    """Loads an RGB frame image from object storage."""

    async def fetch_rgb(self, minio_key: str) -> npt.NDArray[np.uint8]:
        """Return an RGB uint8 image for the object key."""


class ReidEmbedderProtocol(Protocol):
    """Appearance embedding boundary used by the pipeline."""

    async def embed_batch(
        self,
        crops: list[npt.NDArray[np.uint8]],
    ) -> list[Embedding]:
        """Return one ReID embedding per crop."""


@dataclass(frozen=True)
class FaceIdCameraConfig:
    """Per-camera face identification configuration.

    If *enabled* is False, face identification is skipped entirely for
    this camera (e.g. top-down surveillance cameras where faces are
    never visible).  If *min_confidence* is not None it overrides the
    global ``face_id_min_confidence`` for this camera.
    """

    enabled: bool = True
    min_confidence: float | None = None
