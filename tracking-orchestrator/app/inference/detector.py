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

# person-detector/config.pbtxt dims are [16,3,640,640] — static batch baked into the ONNX export.
_DETECTOR_BATCH_SIZE = 16


class PersonDetector:
    """Async person detector using YOLO26L on Triton."""

    def __init__(
        self,
        client: TritonClientProtocol,
        model_name: str = DETECTOR_MODEL_NAME,
        conf_threshold: float = DETECTOR_CONF_THRESHOLD,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._conf = conf_threshold

    async def detect_batch(
        self,
        images: list[npt.NDArray[np.uint8]],
    ) -> list[list[DetectionBox]]:
        """Detect persons in a batch of RGB images (H,W,3) uint8."""
        if not images:
            return []

        preprocessed: list[npt.NDArray[np.float32]] = []
        meta: list[tuple[int, int, int, int, float]] = []
        for img in images:
            tensor, px, py, scale = letterbox_preprocess(img, DETECTOR_INPUT_SIZE)
            preprocessed.append(tensor)
            meta.append((img.shape[0], img.shape[1], px, py, scale))

        n = len(preprocessed)
        pad = _DETECTOR_BATCH_SIZE - n
        if pad > 0:
            preprocessed.extend([preprocessed[0]] * pad)

        batch = np.stack(preprocessed)  # (_DETECTOR_BATCH_SIZE, 3, 640, 640)
        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[("images", batch)],
            output_names=["output0"],
        )
        raw_batch = outputs["output0"]  # (_DETECTOR_BATCH_SIZE, 300, 6) — YOLO26L NMS-free format

        return [decode_output(raw_batch[i], *meta[i], conf_threshold=self._conf) for i in range(n)]

    async def detect(self, image: npt.NDArray[np.uint8]) -> list[DetectionBox]:
        """Detect persons in a single RGB image."""
        return (await self.detect_batch([image]))[0]
