"""YOLO26L person detector backed by Triton Inference Server.

Input images must be RGB uint8 numpy arrays, shape (H, W, 3).
Preprocessing: letterbox resize to 640x640, divide by 255, CHW layout.

YOLO26L uses a NMS-Free (end-to-end) architecture. Output tensor "output0"
shape [batch, 300, 6]:
  dim 0: batch
  dim 1: 300 maximum detections per image (post NMS baked into the model)
  dim 2: 6 = x1, y1, x2, y2 (letterbox pixel space), confidence, class_id

Only class 0 (person) is retained; all other classes are discarded.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from app.inference.schemas import DetectionBox
from app.inference.triton_client import TritonClientProtocol

# YOLO26L input/output constants matching person-detector/config.pbtxt
_MODEL_NAME = "person-detector"
_INPUT_SIZE = 640
_PERSON_CLASS = 0
_CONF_THRESHOLD = 0.25


def _resize_letterbox(
    image: npt.NDArray[np.uint8],
    target: int,
) -> tuple[npt.NDArray[np.float32], int, int, float]:
    """Letterbox-resize image to target x target.

    Returns (padded_float32_chw, pad_x, pad_y, scale) where pad_x/pad_y are
    the number of pixels added on each side and scale is orig→input scaling.
    Uses bilinear interpolation for higher-quality resizing.
    """
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    # Bilinear resize via index broadcasting (per-channel to avoid 3-D index
    # ambiguity in numpy advanced indexing).
    src_float = image.astype(np.float32).transpose(2, 0, 1)  # (3, h, w)
    row_f = np.arange(new_h) * h / new_h
    col_f = np.arange(new_w) * w / new_w

    r0 = np.floor(row_f).astype(np.intp)
    c0 = np.floor(col_f).astype(np.intp)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)

    wr = (row_f - np.floor(row_f)).astype(np.float32)
    wc = (col_f - np.floor(col_f)).astype(np.float32)

    # (new_h, new_w) per channel
    w00 = (1 - wr)[:, None] * (1 - wc)[None, :]
    w10 = wr[:, None] * (1 - wc)[None, :]
    w01 = (1 - wr)[:, None] * wc[None, :]
    w11 = wr[:, None] * wc[None, :]

    bilinear = np.stack(
        [
            src_float[c, r0[:, None], c0[None, :]] * w00
            + src_float[c, r1[:, None], c0[None, :]] * w10
            + src_float[c, r0[:, None], c1[None, :]] * w01
            + src_float[c, r1[:, None], c1[None, :]] * w11
            for c in range(3)
        ],
        axis=2,
    )  # (new_h, new_w, 3)

    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = np.asarray(bilinear, dtype=np.uint8)

    float_chw = np.asarray(canvas, dtype=np.float32) / 255.0
    float_chw = np.transpose(float_chw, (2, 0, 1))  # HWC → CHW
    return float_chw, pad_x, pad_y, scale


def _decode_output(
    raw: npt.NDArray[np.float32],
    orig_h: int,
    orig_w: int,
    pad_x: int,
    pad_y: int,
    scale: float,
    conf_threshold: float = _CONF_THRESHOLD,
) -> list[DetectionBox]:
    """Decode YOLO26L NMS-free output0 tensor (300, 6) → DetectionBox list.

    YOLO26L bakes NMS into the model, so no post-processing NMS is needed.
    Columns: x1, y1, x2, y2 (letterbox pixel space), confidence, class_id.
    Converts letterbox pixel coords to normalised original-image coordinates.
    """
    conf_col = raw[:, 4]
    class_col = raw[:, 5]
    mask = (conf_col > conf_threshold) & (class_col.round().astype(np.int32) == _PERSON_CLASS)
    filtered = raw[mask]

    if filtered.shape[0] == 0:
        return []

    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    x1 = np.clip((filtered[:, 0] - pad_x) / new_w, 0.0, 1.0)
    y1 = np.clip((filtered[:, 1] - pad_y) / new_h, 0.0, 1.0)
    x2 = np.clip((filtered[:, 2] - pad_x) / new_w, 0.0, 1.0)
    y2 = np.clip((filtered[:, 3] - pad_y) / new_h, 0.0, 1.0)
    scores = filtered[:, 4]

    return [
        DetectionBox(
            x1=float(x1[i]),
            y1=float(y1[i]),
            x2=float(x2[i]),
            y2=float(y2[i]),
            confidence=float(scores[i]),
        )
        for i in range(filtered.shape[0])
    ]


class PersonDetector:
    """Async person detector using YOLO26L on Triton."""

    def __init__(
        self,
        client: TritonClientProtocol,
        model_name: str = _MODEL_NAME,
        conf_threshold: float = _CONF_THRESHOLD,
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
            tensor, px, py, scale = _resize_letterbox(img, _INPUT_SIZE)
            preprocessed.append(tensor)
            meta.append((img.shape[0], img.shape[1], px, py, scale))

        batch = np.stack(preprocessed)  # (N, 3, 640, 640)
        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[("images", batch)],
            output_names=["output0"],
        )
        raw_batch = outputs["output0"]  # (N, 300, 6) — YOLO26L NMS-free format

        return [
            _decode_output(raw_batch[i], *meta[i], conf_threshold=self._conf)
            for i in range(len(images))
        ]

    async def detect(self, image: npt.NDArray[np.uint8]) -> list[DetectionBox]:
        """Detect persons in a single RGB image."""
        return (await self.detect_batch([image]))[0]
