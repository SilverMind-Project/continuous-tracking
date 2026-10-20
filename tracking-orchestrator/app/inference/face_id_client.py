"""Async HTTP client for the person-identification-service.

Used by the frame processing pipeline to obtain face-based identity
evidence (FaceAnchors) that feed into the Bayesian identity resolver.

Rate limiting is handled by the caller (``FaceIdentityStage``, per-track cooldown),
so this client is stateless: ``identify_crops`` sends one HTTP POST per crop.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import cv2
import httpx
import numpy as np
import numpy.typing as npt
from structlog import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FaceResult:
    """A single face identification result from person-identification-service."""

    person_id: str
    name: str
    confidence: float
    # Normalised bounding box [x1, y1, x2, y2] in [0, 1].
    bbox_normalized: list[float]
    # Three-valued recognition state.
    recognition_state: str = "recognized"
    # Nearest centroid person_id, even below threshold.
    best_candidate_id: str | None = None
    # Raw cosine similarity to best candidate (alias of similarity, contract).
    raw_similarity: float = 0.0
    # Raw cosine similarity (legacy field; kept for backward compatibility).
    similarity: float = 0.0
    # Head pose in degrees.
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    # SCRFD detection score.
    det_score: float = 0.0
    # Calibration fields. calibrated_confidence is None in any degraded state.
    calibrated_confidence: float | None = None
    calibration_status: str = "degraded_missing"
    arcface_model_version: str = ""
    preprocessing_version: str = ""
    # Face size in pixels of the submitted crop (equal to frame pixels; crops are not resized).
    face_width_px: float = 0.0
    face_height_px: float = 0.0
    # Eye-keypoint distance in crop pixels; None when the service could not measure it.
    interocular_px: float | None = None


class CropQueryStatus(StrEnum):
    """Outcome of one person-crop query against person-identification-service."""

    FACE_FOUND = "face_found"
    NO_FACE = "no_face"
    TOO_SMALL = "too_small"
    ERROR = "error"


@dataclass(frozen=True)
class CropQueryResult:
    """Result for one input crop. Every crop passed to ``identify_crops`` gets exactly one."""

    crop_index: int
    status: CropQueryStatus
    crop_width_px: int
    crop_height_px: int
    faces: tuple[FaceResult, ...] = ()


# Crops smaller than this on either side carry too few pixels for face detection.
_MIN_CROP_SIDE_PX = 40


class FaceIdentificationClient:
    """Async HTTP client for person-identification-service.

    Sends person crops to ``{base_url}/api/v1/identify`` and returns
    face detections with identity assignments.

    Graceful degradation: if the service is unreachable or returns an
    error, an empty list is returned and the error is logged.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 2.0,
        min_confidence: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
        jpeg_quality: int = 95,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self._jpeg_quality = jpeg_quality
        self._base_url = base_url.rstrip("/")
        # Tests inject httpx.MockTransport; production uses the default network transport.
        self._transport = transport
        self._timeout = timeout_s
        self._min_confidence = min_confidence
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={"Content-Type": "application/json"},
            transport=self._transport,
        )
        logger.info(
            "FaceIdentificationClient connected",
            base_url=self._base_url,
            timeout_s=self._timeout,
        )

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
    ) -> list[CropQueryResult]:
        """Identify faces in person crops at native crop resolution.

        Each crop is encoded as a standalone JPEG and sent to the
        person-id-service.  Face bboxes returned by the service are in
        the crop's pixel space; they are normalised to [0, 1] relative
        to the **original frame** using the corresponding crop bbox.
        No similarity filtering is applied here; the caller decides what
        becomes evidence.

        Args:
            crops: RGB uint8 person crops (one per detection).
            crop_bboxes_norm: Normalised person bboxes (x1, y1, x2, y2)
                in [0, 1] of the original frame, one per crop.

        Returns:
            One ``CropQueryResult`` per input crop, in input order, including
            crops that were too small, failed, or contained no face.
        """
        if self._client is None:
            logger.warning("FaceIdentificationClient not connected, skipping identify")
            return [
                CropQueryResult(
                    crop_index=idx,
                    status=CropQueryStatus.ERROR,
                    crop_width_px=int(crop.shape[1]),
                    crop_height_px=int(crop.shape[0]),
                )
                for idx, crop in enumerate(crops)
            ]

        client = self._client
        results: list[CropQueryResult] = []
        crop_with_bboxes = zip(crops, crop_bboxes_norm, strict=True)
        for idx, (crop, crop_bbox_norm) in enumerate(crop_with_bboxes):
            results.append(await self._identify_one(client, idx, crop, crop_bbox_norm))

        logger.debug(
            "Face identification complete (crops)",
            crop_count=len(crops),
            face_count=sum(len(r.faces) for r in results),
            identities=list({f.person_id for r in results for f in r.faces}),
        )
        return results

    async def _identify_one(
        self,
        client: httpx.AsyncClient,
        idx: int,
        crop: npt.NDArray[np.uint8],
        crop_bbox_norm: tuple[float, float, float, float],
    ) -> CropQueryResult:
        crop_h, crop_w = crop.shape[:2]

        def _result(status: CropQueryStatus, faces: tuple[FaceResult, ...] = ()) -> CropQueryResult:
            return CropQueryResult(
                crop_index=idx,
                status=status,
                crop_width_px=int(crop_w),
                crop_height_px=int(crop_h),
                faces=faces,
            )

        if crop_w < _MIN_CROP_SIDE_PX or crop_h < _MIN_CROP_SIDE_PX:
            return _result(CropQueryStatus.TOO_SMALL)

        try:
            crop_b64 = _encode_crop(crop, quality=self._jpeg_quality)
        except Exception:
            logger.exception("Failed to encode crop for face id", crop_index=idx)
            return _result(CropQueryStatus.ERROR)

        try:
            resp = await client.post(
                f"{self._base_url}/api/v1/identify",
                json={"image": crop_b64, "save_guest_images": False},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPError:
            logger.warning(
                "Face identification HTTP error for crop",
                crop_index=idx,
                exc_info=True,
            )
            return _result(CropQueryStatus.ERROR)
        except Exception:
            logger.exception("Face identification unexpected error for crop", crop_index=idx)
            return _result(CropQueryStatus.ERROR)

        faces = tuple(
            face
            for raw in data.get("faces", [])
            if (face := _parse_face(raw, crop_w, crop_h, crop_bbox_norm)) is not None
        )
        return _result(CropQueryStatus.FACE_FOUND if faces else CropQueryStatus.NO_FACE, faces)


def _parse_face(
    face: dict[str, Any],
    crop_w: int,
    crop_h: int,
    crop_bbox_norm: tuple[float, float, float, float],
) -> FaceResult | None:
    """Map one service face to a FaceResult with its bbox normalised to the original frame."""
    bbox_px: list[float] = face.get("bbox", [])
    if len(bbox_px) != 4:
        return None

    cx1, cy1, cx2, cy2 = crop_bbox_norm
    fx1, fy1, fx2, fy2 = bbox_px
    # Normalise face bbox from crop pixel space to crop [0, 1], then to original frame [0, 1].
    crop_w_norm = cx2 - cx1
    crop_h_norm = cy2 - cy1
    nx1 = cx1 + (fx1 / crop_w) * crop_w_norm
    ny1 = cy1 + (fy1 / crop_h) * crop_h_norm
    nx2 = cx1 + (fx2 / crop_w) * crop_w_norm
    ny2 = cy1 + (fy2 / crop_h) * crop_h_norm

    raw_sim = float(face.get("raw_similarity", face.get("similarity", 0)))
    cal_conf_raw = face.get("calibrated_confidence")
    interocular_raw = face.get("interocular_px")
    return FaceResult(
        person_id=face.get("person_id", "unknown"),
        name=face.get("name", "Unknown"),
        confidence=float(face.get("confidence", 0)),
        bbox_normalized=[nx1, ny1, nx2, ny2],
        recognition_state=str(face.get("recognition_state", "recognized")),
        best_candidate_id=face.get("best_candidate_id"),
        raw_similarity=raw_sim,
        similarity=raw_sim,
        yaw_deg=float(face.get("yaw_deg", 0)),
        pitch_deg=float(face.get("pitch_deg", 0)),
        roll_deg=float(face.get("roll_deg", 0)),
        det_score=float(face.get("det_score", 0)),
        calibrated_confidence=float(cal_conf_raw) if cal_conf_raw is not None else None,
        calibration_status=str(face.get("calibration_status", "degraded_missing")),
        arcface_model_version=str(face.get("arcface_model_version", "")),
        preprocessing_version=str(face.get("preprocessing_version", "")),
        face_width_px=float(face.get("face_width_px", 0.0)),
        face_height_px=float(face.get("face_height_px", 0.0)),
        interocular_px=float(interocular_raw) if interocular_raw is not None else None,
    )


def _encode_crop(
    crop: npt.NDArray[np.uint8],
    quality: int,
) -> str:
    """Encode a person crop as base64 JPEG (no downscaling)."""
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed for crop")
    return base64.b64encode(buf.tobytes()).decode("ascii")
