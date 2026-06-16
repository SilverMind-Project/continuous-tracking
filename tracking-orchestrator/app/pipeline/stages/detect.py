"""Detect stage: runs YOLO person detection and IoU dedup."""

from __future__ import annotations

import uuid

import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ...inference.detector import PersonDetector
from ...inference.schemas import DetectionBox
from ...observability import metrics as _metrics
from ..frame_context import FrameContext
from .base import FrameStage

logger = get_logger(__name__)

# Minimum bbox area as a fraction of the frame area. Very small bboxes
# (< 0.5% of frame) are almost always YOLO false positives — no real person
# in a home environment occupies less than 0.5% of a typical 1920x1080 frame.
_MIN_BBOX_AREA_FRACTION = 0.005


def _filter_small_bboxes(
    boxes: list[DetectionBox],
) -> list[DetectionBox]:
    """Drop detections whose normalised area is below the minimum threshold."""
    return [b for b in boxes if b.area >= _MIN_BBOX_AREA_FRACTION]


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_dedup_detections(
    boxes: list[DetectionBox],
    iou_threshold: float,
) -> list[DetectionBox]:
    if len(boxes) <= 1:
        return list(boxes)

    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[DetectionBox] = []
    suppressed_count = 0
    for box in sorted_boxes:
        b_coords = [box.x1, box.y1, box.x2, box.y2]
        if any(_bbox_iou(b_coords, [k.x1, k.y1, k.x2, k.y2]) > iou_threshold for k in kept):
            suppressed_count += 1
        else:
            kept.append(box)
    if suppressed_count > 0:
        _metrics.metrics.detections_suppressed_total.labels(stage="iou_dedup").inc(suppressed_count)
    return kept


class DetectStage(FrameStage):
    name = "detect"

    def __init__(
        self,
        detector: PersonDetector,
        iou_dedup_threshold: float = 0.55,
        enable_low_confidence_recovery: bool = False,
        low_confidence_floor: float = 0.25,
        high_threshold: float = 0.7,
    ) -> None:
        self._detector = detector
        self._iou_dedup_threshold = iou_dedup_threshold
        self._enable_low_confidence_recovery = enable_low_confidence_recovery
        self._low_confidence_floor = low_confidence_floor
        self._high_threshold = high_threshold

    async def run(self, ctx: FrameContext) -> None:
        await self.run_batch([ctx])

    async def run_batch(self, contexts: list[FrameContext]) -> None:
        if not contexts:
            return

        images: list[npt.NDArray[np.uint8]] = [ctx.require_image() for ctx in contexts]

        if self._enable_low_confidence_recovery:
            await self._run_batch_with_recovery(contexts, images)
        else:
            await self._run_batch_standard(contexts, images)

    async def _run_batch_standard(
        self,
        contexts: list[FrameContext],
        images: list[npt.NDArray[np.uint8]],
    ) -> None:
        """Standard path, byte-identical to the behavior before low-confidence recovery."""
        detections_by_frame = await self._detect_images(images)
        for ctx, image, detections in zip(contexts, images, detections_by_frame, strict=True):
            logger.debug(
                "detections_raw",
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
                count=len(detections),
                image_shape=f"{image.shape[0]}x{image.shape[1]}",
            )
            if detections:
                detections = _filter_small_bboxes(detections)
            if detections and self._iou_dedup_threshold < 1.0:
                detections = _iou_dedup_detections(detections, self._iou_dedup_threshold)
            ctx.raw_detections = detections
            ctx._detection_ids = {idx: str(uuid.uuid4()) for idx in range(len(detections))}

    async def _run_batch_with_recovery(
        self,
        contexts: list[FrameContext],
        images: list[npt.NDArray[np.uint8]],
    ) -> None:
        """Low-confidence recovery path: one Triton call at the band floor,
        then partition into high (>=high_threshold) and low (band) sets.

        High band: exactly today's raw_detections (only these seed gallery,
        ReID, face, and can spawn PHs).
        Low band: stored in ctx.low_band_detections for a second association
        pass in WorldTracker; they cannot spawn PHs or contribute identity
        evidence.
        """
        all_detections_by_frame = await self._detect_images_at_threshold(
            images, self._low_confidence_floor
        )
        for ctx, image, all_detections in zip(
            contexts, images, all_detections_by_frame, strict=True
        ):
            all_detections = _filter_small_bboxes(all_detections)

            high: list[DetectionBox] = [
                d for d in all_detections if d.confidence >= self._high_threshold
            ]
            low: list[DetectionBox] = [
                d
                for d in all_detections
                if self._low_confidence_floor <= d.confidence < self._high_threshold
            ]

            if high and self._iou_dedup_threshold < 1.0:
                high = _iou_dedup_detections(high, self._iou_dedup_threshold)
            if low and self._iou_dedup_threshold < 1.0:
                low = _iou_dedup_detections(low, self._iou_dedup_threshold)

            # Cross-band dedup: drop any low-band box overlapping a kept high-band box.
            if low and high:
                low = [
                    lb
                    for lb in low
                    if not any(
                        _bbox_iou(
                            [lb.x1, lb.y1, lb.x2, lb.y2],
                            [hb.x1, hb.y1, hb.x2, hb.y2],
                        )
                        > self._iou_dedup_threshold
                        for hb in high
                    )
                ]

            logger.debug(
                "detections_partitioned",
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
                high_count=len(high),
                low_band_count=len(low),
                image_shape=f"{image.shape[0]}x{image.shape[1]}",
            )

            ctx.raw_detections = high
            ctx._detection_ids = {idx: str(uuid.uuid4()) for idx in range(len(high))}
            ctx.low_band_detections = low

    async def _detect_images(
        self,
        images: list[npt.NDArray[np.uint8]],
    ) -> list[list[DetectionBox]]:
        detect_batch = getattr(self._detector, "detect_batch", None)
        if detect_batch is not None:
            detections_by_frame = await detect_batch(images)
            if isinstance(detections_by_frame, list) and len(detections_by_frame) == len(images):
                return detections_by_frame

        detect = getattr(self._detector, "detect", None)
        if detect is None:
            msg = "detector must provide detect_batch(images) or detect(image)"
            raise TypeError(msg)
        return [await detect(image) for image in images]

    async def _detect_images_at_threshold(
        self,
        images: list[npt.NDArray[np.uint8]],
        threshold: float,
    ) -> list[list[DetectionBox]]:
        detect_at = getattr(self._detector, "detect_batch_at_threshold", None)
        if detect_at is not None:
            result = await detect_at(images, threshold)
            if isinstance(result, list) and len(result) == len(images):
                return result
        # Fallback for test fakes that don't implement detect_batch_at_threshold.
        return await self._detect_images(images)
