"""Inference stage: crops detections, runs ReID embeddings and pose estimation.

Produces domain Detection objects alongside typed ``AppearanceEvidence``
and ``PoseEvidence`` records for downstream model provenance.
"""

from __future__ import annotations

import asyncio

import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ...domain import BoundingBox, Detection, FloorPoint
from ...inference.evidence import AppearanceEvidence, PersonDetectionEvidence, PoseEvidence
from ...inference.pose import PoseEstimator
from ...inference.schemas import DetectionBox, Embedding, PoseResult
from ...pipeline.crop_quality import CropQuality
from ...pipeline.crops import crop_detection, is_degenerate
from ..frame_context import FrameContext
from ..types import ReidEmbedderProtocol
from .base import FrameStage

logger = get_logger(__name__)


class InferenceStage(FrameStage):
    name = "inference"

    def __init__(
        self,
        reid_embedder: ReidEmbedderProtocol | None = None,
        pose_estimator: PoseEstimator | None = None,
        pose_enabled: bool = True,
        reid_model_version: str = "",
        pose_model_version: str = "",
        detector_model_version: str = "",
        detector_preprocess_w: int = 640,
        detector_preprocess_h: int = 640,
    ) -> None:
        self._reid_embedder = reid_embedder
        self._pose_estimator = pose_estimator
        self._pose_enabled = pose_enabled
        self._reid_model_version = reid_model_version
        self._pose_model_version = pose_model_version
        self._detector_model_version = detector_model_version
        self._detector_preprocess_w = detector_preprocess_w
        self._detector_preprocess_h = detector_preprocess_h

    async def run(self, ctx: FrameContext) -> None:
        detections = ctx.raw_detections
        if not detections:
            return

        image = ctx.require_image()
        crops = [crop_detection(image, det) for det in detections]
        ctx.crops = crops

        async def _do_reid() -> list[Embedding]:
            if self._reid_embedder is not None:
                return await self._reid_embedder.embed_batch(crops)
            return []

        async def _do_pose() -> list[PoseResult | None]:
            if self._pose_estimator is not None and self._pose_enabled:
                return await self._run_pose(crops, detections)
            return []

        embeddings, pose_results = await asyncio.gather(_do_reid(), _do_pose())
        ctx.embeddings = embeddings

        ew = ctx.effective_width
        eh = ctx.effective_height

        appearance_evidence: list[AppearanceEvidence] = []
        pose_evidence: list[PoseEvidence] = []
        detection_evidence: dict[int, PersonDetectionEvidence] = {}

        for det_idx, det in enumerate(detections):
            det_id = ctx._detection_ids.get(det_idx, "")
            bbox = BoundingBox(
                x_min=int(det.x1 * ew),
                y_min=int(det.y1 * eh),
                x_max=int(det.x2 * ew),
                y_max=int(det.y2 * eh),
            )

            # PersonDetectionEvidence — detector provenance.
            fp = ctx._floor_points_by_index.get(det_idx)
            detection_evidence[det_idx] = PersonDetectionEvidence(
                detection_id=det_id,
                camera_id=ctx.frame.camera_id,
                frame_index=ctx.frame.frame_index,
                bbox=bbox,
                confidence=det.confidence,
                floor_point=fp if fp is not None else FloorPoint(0, 0),
                model_version=self._detector_model_version,
                preprocessing_width=self._detector_preprocess_w,
                preprocessing_height=self._detector_preprocess_h,
                captured_at=ctx.event_time,
            )

            # AppearanceEvidence — ReID provenance.
            emb = embeddings[det_idx] if det_idx < len(embeddings) else None
            crop = crops[det_idx] if det_idx < len(crops) else None
            cq = _compute_crop_quality(
                det,
                bbox,
                crop,
                ew,
                eh,
                pose_results[det_idx] if det_idx < len(pose_results) else None,
            )
            if emb is not None:
                appearance_evidence.append(
                    AppearanceEvidence(
                        detection_id=det_id,
                        camera_id=ctx.frame.camera_id,
                        frame_index=ctx.frame.frame_index,
                        embedding=tuple(float(v) for v in emb),
                        crop_quality=cq.quality,
                        model_version=self._reid_model_version,
                        captured_at=ctx.event_time,
                    )
                )

            # PoseEvidence — RTMPose provenance.
            pose_result = pose_results[det_idx] if det_idx < len(pose_results) else None
            if pose_result is not None:
                kp_tuples = tuple((kp.x, kp.y, kp.score) for kp in pose_result.keypoints)
                visible = sum(1 for kp in pose_result.keypoints if kp.score > 0.2)
                mean_score = (
                    sum(kp.score for kp in pose_result.keypoints) / len(pose_result.keypoints)
                    if pose_result.keypoints
                    else 0.0
                )
                pose_evidence.append(
                    PoseEvidence(
                        detection_id=det_id,
                        camera_id=ctx.frame.camera_id,
                        frame_index=ctx.frame.frame_index,
                        keypoints=kp_tuples,
                        visible_keypoint_count=visible,
                        quality=mean_score,
                        model_version=self._pose_model_version,
                        captured_at=ctx.event_time,
                    )
                )

            # Domain Detection — carry crop_quality so WorldTrackingStage can set
            # WorldObservation.quality without recomputing the composite scorer.
            domain_det = Detection(
                detection_id=det_id,
                camera_id=ctx.frame.camera_id,
                bbox=bbox,
                embedding=emb.tolist() if emb is not None else [],
                capture_time=ctx.capture_time,
                event_time=ctx.event_time,
                confidence=det.confidence,
                floor_point=fp if fp is not None else FloorPoint(0, 0),
                crop_quality=cq.quality,
            )
            ctx.domain_detections.append(domain_det)
            if pose_result is not None:
                ctx.det_pose_result[domain_det.detection_id] = pose_result

        ctx._detection_evidence = detection_evidence
        ctx._appearance_evidence = appearance_evidence
        ctx._pose_evidence = pose_evidence

    async def _run_pose(
        self,
        crops: list[npt.NDArray[np.uint8]],
        detections: list[DetectionBox],
    ) -> list[PoseResult | None]:
        assert self._pose_estimator is not None
        valid_idxs: list[int] = []
        valid_crops: list[npt.NDArray[np.uint8]] = []
        results: list[PoseResult | None] = [None] * len(crops)
        for i, crop in enumerate(crops):
            if is_degenerate(crop):
                logger.debug(
                    "pose_skipped",
                    detection_index=i,
                    crop_width=crop.shape[1],
                    crop_height=crop.shape[0],
                )
                continue
            valid_idxs.append(i)
            valid_crops.append(crop)

        if valid_crops:
            batch_results = await self._pose_estimator.infer_batch(valid_crops)
            for vi, pr in zip(valid_idxs, batch_results, strict=True):
                results[vi] = pr
                visible = sum(1 for kp in pr.keypoints if kp.score > 0.2)
                logger.debug(
                    "pose_result",
                    detection_index=vi,
                    visible_keypoints=visible,
                    min_score=round(min(kp.score for kp in pr.keypoints), 3),
                    max_score=round(max(kp.score for kp in pr.keypoints), 3),
                )

        return results


def _compute_crop_quality(
    det: DetectionBox,
    bbox: BoundingBox,
    crop: npt.NDArray[np.uint8] | None,
    frame_w: int,
    frame_h: int,
    pose_result: PoseResult | None,
) -> CropQuality:
    """Compute CropQuality from detection and optional pose metadata."""
    area_fraction = (bbox.width * bbox.height) / max(frame_w * frame_h, 1)
    edge_truncated = (
        bbox.x_min <= 0 or bbox.y_min <= 0 or bbox.x_max >= frame_w or bbox.y_max >= frame_h
    )
    crop_w = crop.shape[1] if crop is not None else 0
    crop_h = crop.shape[0] if crop is not None else 0
    visible_kp = (
        sum(1 for kp in pose_result.keypoints if kp.score > 0.2) if pose_result is not None else 0
    )
    return CropQuality(
        area_fraction=area_fraction,
        detector_confidence=det.confidence,
        edge_truncated=edge_truncated,
        crop_width_px=crop_w,
        crop_height_px=crop_h,
        visible_keypoint_count=visible_kp,
    )
