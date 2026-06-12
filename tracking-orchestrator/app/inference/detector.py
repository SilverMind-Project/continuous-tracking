"""YOLO26L person detector backed by Triton Inference Server.

Input images must be RGB uint8 numpy arrays, shape (H, W, 3).
Preprocessing and decode logic are shared via ``triton_shared.inference.detection``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from triton_shared.client import TritonClientProtocol
from triton_shared.inference.detection import (
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_INPUT_SIZE,
    DETECTOR_MODEL_NAME,
    decode_output,
    letterbox_preprocess,
)
from triton_shared.inference.schemas import DetectionBox

# CTS detector exports are static batch-8 by default. Set this to 0 only
# for a truly dynamic-batch export.
_DEFAULT_DETECTOR_STATIC_BATCH_SIZE = 8


class PersonDetector:
    """Async person detector using YOLO26L on Triton."""

    def __init__(
        self,
        client: TritonClientProtocol,
        model_name: str = DETECTOR_MODEL_NAME,
        conf_threshold: float = DETECTOR_CONF_THRESHOLD,
        static_batch_size: int = _DEFAULT_DETECTOR_STATIC_BATCH_SIZE,
        dynamic_batch: bool = False,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._conf = conf_threshold
        self._static_batch_size = static_batch_size
        # When True: send the actual frame count to a variable-batch ONNX export
        # (person-detector-dynamic). No padding is added; static_batch_size is
        # ignored. When False (default): pad to static_batch_size as before.
        self._dynamic_batch = dynamic_batch

    async def detect_batch(
        self,
        images: list[npt.NDArray[np.uint8]],
    ) -> list[list[DetectionBox]]:
        """Detect persons in a batch of RGB images (H,W,3) uint8."""
        if not images:
            return []

        if (
            not self._dynamic_batch
            and self._static_batch_size > 0
            and len(images) > self._static_batch_size
        ):
            results: list[list[DetectionBox]] = []
            for start in range(0, len(images), self._static_batch_size):
                chunk = images[start : start + self._static_batch_size]
                results.extend(await self._detect_model_batch(chunk))
            return results

        return await self._detect_model_batch(images)

    async def _detect_model_batch(
        self,
        images: list[npt.NDArray[np.uint8]],
    ) -> list[list[DetectionBox]]:
        preprocessed: list[npt.NDArray[np.float32]] = []
        meta: list[tuple[int, int, int, int, float]] = []
        for img in images:
            tensor, px, py, scale = letterbox_preprocess(img, DETECTOR_INPUT_SIZE)
            preprocessed.append(tensor)
            meta.append((img.shape[0], img.shape[1], px, py, scale))

        n = len(preprocessed)
        if not self._dynamic_batch and self._static_batch_size > 0:
            pad = self._static_batch_size - n
            if pad > 0:
                preprocessed.extend([preprocessed[0]] * pad)

        batch = np.stack(preprocessed)
        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[("images", batch)],
            output_names=["output0"],
        )
        raw_batch = outputs["output0"]  # (N, 300, 6) — YOLO26L NMS-free format

        return [decode_output(raw_batch[i], *meta[i], conf_threshold=self._conf) for i in range(n)]

    @property
    def conf_threshold(self) -> float:
        """The primary confidence threshold this detector was configured with."""
        return self._conf

    async def detect(self, image: npt.NDArray[np.uint8]) -> list[DetectionBox]:
        """Detect persons in a single RGB image."""
        return (await self.detect_batch([image]))[0]

    async def detect_batch_at_threshold(
        self,
        images: list[npt.NDArray[np.uint8]],
        threshold: float,
    ) -> list[list[DetectionBox]]:
        """Run detection at *threshold* in one Triton call.

        Returns all boxes with confidence >= *threshold*.  The caller is
        responsible for partitioning into high/low bands and applying
        cross-band IoU dedup.
        """
        if not images:
            return []
        if (
            not self._dynamic_batch
            and self._static_batch_size > 0
            and len(images) > self._static_batch_size
        ):
            results: list[list[DetectionBox]] = []
            for start in range(0, len(images), self._static_batch_size):
                chunk = images[start : start + self._static_batch_size]
                results.extend(await self._detect_model_batch_at(chunk, threshold))
            return results
        return await self._detect_model_batch_at(images, threshold)

    async def _detect_model_batch_at(
        self,
        images: list[npt.NDArray[np.uint8]],
        threshold: float,
    ) -> list[list[DetectionBox]]:
        preprocessed: list[npt.NDArray[np.float32]] = []
        meta: list[tuple[int, int, int, int, float]] = []
        for img in images:
            tensor, px, py, scale = letterbox_preprocess(img, DETECTOR_INPUT_SIZE)
            preprocessed.append(tensor)
            meta.append((img.shape[0], img.shape[1], px, py, scale))

        n = len(preprocessed)
        if not self._dynamic_batch and self._static_batch_size > 0:
            pad = self._static_batch_size - n
            if pad > 0:
                preprocessed.extend([preprocessed[0]] * pad)

        batch = np.stack(preprocessed)
        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[("images", batch)],
            output_names=["output0"],
        )
        raw_batch = outputs["output0"]
        return [decode_output(raw_batch[i], *meta[i], conf_threshold=threshold) for i in range(n)]
