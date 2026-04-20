"""YOLO11m person detector backed by Triton Inference Server.

Input images must be RGB uint8 numpy arrays, shape (H, W, 3).
Preprocessing: letterbox resize to 640x640, divide by 255, CHW layout.

YOLO11m output tensor "output0" shape [batch, 84, 8400]:
  dim 0: batch
  dim 1: 84 = 4 (cx, cy, w, h in input-pixel space) + 80 COCO class scores
  dim 2: 8400 anchor positions across three grid scales

Only class 0 (person) is used; all other classes are discarded.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from app.inference.schemas import DetectionBox
from app.inference.triton_client import TritonClientProtocol

# YOLO11m input/output constants matching person-detector/config.pbtxt
_MODEL_NAME = "person-detector"
_INPUT_SIZE = 640
_PERSON_CLASS = 0
_CONF_THRESHOLD = 0.25
_IOU_THRESHOLD = 0.45

# ImageNet normalisation is NOT applied to YOLO — it expects [0, 1] input.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_letterbox(
    image: npt.NDArray[np.uint8],
    target: int,
) -> tuple[npt.NDArray[np.float32], int, int, float]:
    """Letterbox-resize image to target x target.

    Returns (padded_float32_chw, pad_x, pad_y, scale) where pad_x/pad_y are
    the number of pixels added on each side and scale is orig→input scaling.
    """
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    # Nearest-neighbour resize via index broadcasting
    row_idx = np.floor(np.arange(new_h) * h / new_h).astype(np.intp)
    col_idx = np.floor(np.arange(new_w) * w / new_w).astype(np.intp)
    resized = image[row_idx[:, None], col_idx[None, :]]  # (new_h, new_w, 3)

    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    float_chw = np.asarray(canvas, dtype=np.float32) / 255.0
    float_chw = np.transpose(float_chw, (2, 0, 1))  # HWC → CHW
    return float_chw, pad_x, pad_y, scale


def _nms(
    boxes: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    """Pure-numpy NMS.  boxes: (N, 4) x1,y1,x2,y2; returns kept indices."""
    if boxes.shape[0] == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_threshold]

    return keep


def _decode_output(
    raw: npt.NDArray[np.float32],
    orig_h: int,
    orig_w: int,
    pad_x: int,
    pad_y: int,
    scale: float,
) -> list[DetectionBox]:
    """Decode YOLO11m output0 tensor (84, 8400) → DetectionBox list."""
    preds = raw.T  # (8400, 84)
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    person_score = preds[:, 4 + _PERSON_CLASS]  # class 0

    mask = person_score > _CONF_THRESHOLD
    if not mask.any():
        return []

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    scores = person_score[mask]

    # cx,cy,w,h (input-pixel space) → x1,y1,x2,y2 (normalised original)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    x1 = np.clip((cx - bw / 2 - pad_x) / new_w, 0.0, 1.0)
    y1 = np.clip((cy - bh / 2 - pad_y) / new_h, 0.0, 1.0)
    x2 = np.clip((cx + bw / 2 - pad_x) / new_w, 0.0, 1.0)
    y2 = np.clip((cy + bh / 2 - pad_y) / new_h, 0.0, 1.0)

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    kept = _nms(boxes, scores, _IOU_THRESHOLD)

    return [
        DetectionBox(
            x1=float(boxes[i, 0]),
            y1=float(boxes[i, 1]),
            x2=float(boxes[i, 2]),
            y2=float(boxes[i, 3]),
            confidence=float(scores[i]),
        )
        for i in kept
    ]


class PersonDetector:
    """Async person detector using YOLO11m on Triton."""

    def __init__(
        self,
        client: TritonClientProtocol,
        model_name: str = _MODEL_NAME,
        conf_threshold: float = _CONF_THRESHOLD,
        iou_threshold: float = _IOU_THRESHOLD,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._conf = conf_threshold
        self._iou = iou_threshold

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
        raw_batch = outputs["output0"]  # (N, 84, 8400)

        return [_decode_output(raw_batch[i], *meta[i]) for i in range(len(images))]

    async def detect(self, image: npt.NDArray[np.uint8]) -> list[DetectionBox]:
        """Detect persons in a single RGB image."""
        return (await self.detect_batch([image]))[0]
