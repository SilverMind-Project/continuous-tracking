"""Depth Anything v2 ViT-S Metric Indoor depth estimator backed by Triton.

Input: RGB image as ``uint8`` numpy array ``(H, W, 3)``.
Preprocessing: letterbox resize to 518x518, ImageNet normalise, CHW layout.

Triton model ``depth-anything-v2`` output tensor ``depth`` shape ``[1, 518, 518]``
(or ``[1, 1, 518, 518]`` depending on export) carrying **absolute metric depth
in metres** for each pixel.  The output is cropped back to the unpadded region
and bilinearly resized to the original ``(H, W)`` resolution before returning.

Export command (run from the Depth Anything v2 repo root):

    python export_onnx.py \\
        --encoder vits \\
        --dataset indoor \\
        --model_path checkpoints/depth_anything_v2_metric_vits.pth \\
        --output_path triton-models/depth-anything-v2/1/model.onnx \\
        --height 518 --width 518

Verify tensor names after export::

    python -c "import onnx; m = onnx.load('1/model.onnx'); \\
        print([i.name for i in m.graph.input], [o.name for o in m.graph.output])"

Expected: inputs ``['pixel_values']``, outputs ``['predicted_depth']`` (or ``['depth']``).
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from app.inference.triton_client import TritonClientProtocol

_MODEL_NAME = "depth-anything-v2"
_INPUT_H = 518
_INPUT_W = 518

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess(
    image: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.float32], int, int, int, int]:
    """Letterbox-resize to 518x518, ImageNet-normalise, return (chw, pad_t, pad_b, pad_l, pad_r).

    Returns CHW float32 tensor ready for batching, plus the letterbox padding
    (in pixels within the 518x518 canvas) so the caller can recover the
    unpadded region after inference.
    """
    h, w = image.shape[:2]
    scale = min(_INPUT_H / h, _INPUT_W / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (_INPUT_H - new_h) // 2
    pad_bottom = _INPUT_H - new_h - pad_top
    pad_left = (_INPUT_W - new_w) // 2
    pad_right = _INPUT_W - new_w - pad_left

    canvas = np.full((_INPUT_H, _INPUT_W, 3), 114, dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    chw: npt.NDArray[np.float32] = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    chw = (chw - _MEAN) / _STD
    return chw, pad_top, pad_bottom, pad_left, pad_right


def _postprocess(
    raw: npt.NDArray[np.float32],
    orig_h: int,
    orig_w: int,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
) -> npt.NDArray[np.float32]:
    """Strip letterbox padding and resize depth map to (orig_h, orig_w)."""
    # raw may be (1, H, W) or (H, W)
    depth = raw[0] if raw.ndim == 3 else raw

    # Crop out the padding region.
    h_end = _INPUT_H - pad_bottom if pad_bottom > 0 else _INPUT_H
    w_end = _INPUT_W - pad_right if pad_right > 0 else _INPUT_W
    cropped = depth[pad_top:h_end, pad_left:w_end]

    # Resize to original image resolution.
    resized: npt.NDArray[np.float32] = np.asarray(
        cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR), dtype=np.float32
    )
    return resized


class DepthEstimator:
    """Async metric depth estimator using Depth Anything v2 on Triton.

    Returns an absolute depth map in metres with the same spatial resolution
    as the input image.  Only processes one image at a time (no batching) since
    auto-calibration is a low-frequency operation.
    """

    def __init__(
        self,
        client: TritonClientProtocol,
        model_name: str = _MODEL_NAME,
    ) -> None:
        self._client = client
        self._model_name = model_name

    async def estimate(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Estimate per-pixel depth (metres) for one RGB image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB array.

        Returns:
            ``(H, W)`` float32 array with absolute depth values in metres.
            Invalid (unreliable) pixels are marked with ``0.0``.
        """
        orig_h, orig_w = image.shape[:2]
        chw, pad_top, pad_bottom, pad_left, pad_right = _preprocess(image)

        # Triton expects a batch dimension: (1, 3, 518, 518).
        batch = chw[np.newaxis]  # (1, 3, 518, 518)

        # The ONNX export may name the input "pixel_values" or "image".
        # We try "pixel_values" first (standard HuggingFace export name).
        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[("pixel_values", batch)],
            output_names=["predicted_depth"],
        )
        raw: npt.NDArray[np.float32] = outputs["predicted_depth"]  # (1, H, W) or (1, 1, H, W)

        # Some exports add a leading batch+channel dim: (1, 1, H, W).
        if raw.ndim == 4:
            raw = raw[0]  # → (1, H, W)

        depth = _postprocess(raw, orig_h, orig_w, pad_top, pad_bottom, pad_left, pad_right)
        # Clamp negatives (metric model should not produce them, but guard anyway).
        np.clip(depth, 0.0, None, out=depth)
        return depth
