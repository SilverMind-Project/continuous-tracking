"""RTMPose-m 2D pose estimator backed by Triton Inference Server.

Input: person crop as RGB uint8 numpy array (H, W, 3).
Preprocessing: letterbox resize to 256x192 (HxW), ImageNet normalise, CHW.

Triton model "pose-rtmpose" outputs:
  simcc_x: [batch, 17, 384]  x-axis SimCC logits
  simcc_y: [batch, 17, 512]  y-axis SimCC logits

SimCC decoding (simcc_split_ratio = 2.0):
  x_pixel = argmax(simcc_x[k]) / 2.0   (in crop pixel space)
  y_pixel = argmax(simcc_y[k]) / 2.0   (in crop pixel space)

Visibility score: min(max_logit_x, max_logit_y). The ONNX export produces
small-scale logits (~0.3-0.6 for visible keypoints, ~0.0 for occluded), so
the raw max logit is used directly rather than softmax-peak probability.

Keypoints are returned in original-crop normalised coordinates [0, 1].
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from app.inference.schemas import (
    NUM_KEYPOINTS,
    Keypoint,
    PoseResult,
)
from app.inference.triton_client import TritonClientProtocol

_MODEL_NAME = "pose-rtmpose"
_INPUT_H = 256
_INPUT_W = 192
_SIMCC_SPLIT_RATIO = 2.0
_X_BINS = int(_INPUT_W * _SIMCC_SPLIT_RATIO)  # 384
_Y_BINS = int(_INPUT_H * _SIMCC_SPLIT_RATIO)  # 512
_POSE_MAX_BATCH = 8

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess(
    crop: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.float32], int, int, float]:
    """Letterbox resize to 256x192, ImageNet-normalise, return (chw, pad_x, pad_y, scale)."""
    h, w = crop.shape[:2]
    scale = min(_INPUT_H / h, _INPUT_W / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_y = (_INPUT_H - new_h) // 2
    pad_x = (_INPUT_W - new_w) // 2
    canvas = np.full((_INPUT_H, _INPUT_W, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    chw = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (chw - _MEAN) / _STD, pad_x, pad_y, scale


def _decode_simcc(
    simcc_x: npt.NDArray[np.float32],  # (17, 384)
    simcc_y: npt.NDArray[np.float32],  # (17, 512)
    orig_h: int,
    orig_w: int,
    pad_x: int,
    pad_y: int,
    scale: float,
) -> PoseResult:
    """Decode SimCC logits → PoseResult in original-crop normalised coords."""
    kpts: list[Keypoint] = []
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    for k in range(NUM_KEYPOINTS):
        x_bin = int(np.argmax(simcc_x[k]))
        y_bin = int(np.argmax(simcc_y[k]))

        # Bin → pixel in input space, then unpad + unnormalise
        x_input = x_bin / _SIMCC_SPLIT_RATIO  # pixel in 256x192 crop
        y_input = y_bin / _SIMCC_SPLIT_RATIO

        x_norm = float(np.clip((x_input - pad_x) / new_w, 0.0, 1.0))
        y_norm = float(np.clip((y_input - pad_y) / new_h, 0.0, 1.0))
        # The exported ONNX model produces small-scale logits (~0.3-0.6 peak
        # vs ~0.0 background). Softmax-peak scoring over 384/512 bins dilutes
        # valid peaks to ~0.004, below any useful threshold. Use the raw max
        # logit directly — visible keypoints score >0.2, occluded ones <0.1.
        score = float(min(float(simcc_x[k].max()), float(simcc_y[k].max())))

        kpts.append(Keypoint(x=x_norm, y=y_norm, score=score))

    return PoseResult(keypoints=tuple(kpts))


class PoseEstimator:
    """Async 2D pose estimator using RTMPose-m on Triton."""

    def __init__(
        self,
        client: TritonClientProtocol,
        model_name: str = _MODEL_NAME,
    ) -> None:
        self._client = client
        self._model_name = model_name

    async def infer_batch(
        self,
        crops: list[npt.NDArray[np.uint8]],
    ) -> list[PoseResult]:
        """Estimate pose for a batch of RGB person crops."""
        if not crops:
            return []

        if len(crops) > _POSE_MAX_BATCH:
            results: list[PoseResult] = []
            for start in range(0, len(crops), _POSE_MAX_BATCH):
                results.extend(await self.infer_batch(crops[start : start + _POSE_MAX_BATCH]))
            return results

        preprocessed: list[npt.NDArray[np.float32]] = []
        meta: list[tuple[int, int, int, int, float]] = []
        for crop in crops:
            tensor, px, py, scale = _preprocess(crop)
            preprocessed.append(tensor)
            meta.append((crop.shape[0], crop.shape[1], px, py, scale))

        batch = np.stack(preprocessed)  # (N, 3, 256, 192)
        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[("input", batch)],
            output_names=["simcc_x", "simcc_y"],
        )
        sx: npt.NDArray[np.float32] = outputs["simcc_x"]  # (N, 17, 384)
        sy: npt.NDArray[np.float32] = outputs["simcc_y"]  # (N, 17, 512)

        return [_decode_simcc(sx[i], sy[i], *meta[i]) for i in range(len(crops))]

    async def infer(self, crop: npt.NDArray[np.uint8]) -> PoseResult:
        """Estimate pose for a single RGB person crop."""
        return (await self.infer_batch([crop]))[0]
