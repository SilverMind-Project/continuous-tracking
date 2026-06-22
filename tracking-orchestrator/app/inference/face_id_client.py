"""Async HTTP client for the person-identification-service.

Used by the frame processing pipeline to obtain face-based identity
evidence (FaceAnchors) that feed into the Bayesian identity resolver.

Rate-limiting is handled by the caller (frame_pipeline) via a per-camera
cooldown, so this client is stateless — each ``identify_crops`` call is a
single HTTP POST to the person-identification-service.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
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
    # Raw cosine similarity to best candidate (alias of similarity, M10 contract).
    raw_similarity: float = 0.0
    # Raw cosine similarity (legacy field; kept for backward compatibility).
    similarity: float = 0.0
    # Head pose in degrees.
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    # SCRFD detection score.
    det_score: float = 0.0
    # M10 calibration fields. calibrated_confidence is None in any degraded state.
    calibrated_confidence: float | None = None
    calibration_status: str = "degraded_missing"
    arcface_model_version: str = ""
    preprocessing_version: str = ""


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
    ) -> None:
        self._base_url = base_url.rstrip("/")
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
    ) -> list[tuple[int, list[FaceResult]]]:
        """Identify faces in person crops at native crop resolution.

        Each crop is encoded as a standalone JPEG and sent to the
        person-id-service.  Face bboxes returned by the service are in
        the crop's pixel space; they are normalised to [0, 1] relative
        to the **original frame** using the corresponding crop bbox.

        Args:
            crops: RGB uint8 person crops (one per detection).
            crop_bboxes_norm: Normalised person bboxes (x1, y1, x2, y2)
                in [0, 1] of the original frame, one per crop.

        Returns:
            List of ``(crop_index, [FaceResult, ...])`` pairs.  Only
            crops with at least one face above the confidence threshold
            are included.
        """
        if self._client is None:
            logger.warning("FaceIdentificationClient not connected, skipping identify")
            return []

        results: list[tuple[int, list[FaceResult]]] = []
        crop_with_bboxes = zip(crops, crop_bboxes_norm, strict=True)
        for idx, (crop, (cx1, cy1, cx2, cy2)) in enumerate(crop_with_bboxes):
            crop_h, crop_w = crop.shape[:2]
            if crop_w < 40 or crop_h < 40:
                continue  # too small for meaningful face detection

            try:
                crop_b64 = _encode_crop(crop)
            except Exception:
                logger.exception("Failed to encode crop for face id", crop_index=idx)
                continue

            try:
                resp = await self._client.post(
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
                continue
            except Exception:
                logger.exception("Face identification unexpected error for crop", crop_index=idx)
                continue

            crop_results: list[FaceResult] = []
            for face in data.get("faces", []):
                bbox_px: list[float] = face.get("bbox", [])
                if len(bbox_px) != 4:
                    continue

                fx1, fy1, fx2, fy2 = bbox_px
                # Normalise face bbox from crop pixel space → crop [0, 1].
                fnx1 = fx1 / crop_w
                fny1 = fy1 / crop_h
                fnx2 = fx2 / crop_w
                fny2 = fy2 / crop_h
                # Map from crop [0, 1] → original frame [0, 1].
                crop_w_norm = cx2 - cx1
                crop_h_norm = cy2 - cy1
                nx1 = cx1 + fnx1 * crop_w_norm
                ny1 = cy1 + fny1 * crop_h_norm
                nx2 = cx1 + fnx2 * crop_w_norm
                ny2 = cy1 + fny2 * crop_h_norm

                recognition_state = str(face.get("recognition_state", "recognized"))

                raw_sim = float(face.get("raw_similarity", face.get("similarity", 0)))
                cal_conf_raw = face.get("calibrated_confidence")
                cal_conf: float | None = float(cal_conf_raw) if cal_conf_raw is not None else None
                crop_results.append(
                    FaceResult(
                        person_id=face.get("person_id", "unknown"),
                        name=face.get("name", "Unknown"),
                        confidence=float(face.get("confidence", 0)),
                        bbox_normalized=[nx1, ny1, nx2, ny2],
                        recognition_state=recognition_state,
                        best_candidate_id=face.get("best_candidate_id"),
                        raw_similarity=raw_sim,
                        similarity=raw_sim,
                        yaw_deg=float(face.get("yaw_deg", 0)),
                        pitch_deg=float(face.get("pitch_deg", 0)),
                        roll_deg=float(face.get("roll_deg", 0)),
                        det_score=float(face.get("det_score", 0)),
                        calibrated_confidence=cal_conf,
                        calibration_status=str(face.get("calibration_status", "degraded_missing")),
                        arcface_model_version=str(face.get("arcface_model_version", "")),
                        preprocessing_version=str(face.get("preprocessing_version", "")),
                    )
                )

            if crop_results:
                results.append((idx, crop_results))

        total_faces = sum(len(r) for _, r in results)
        logger.debug(
            "Face identification complete (crops)",
            crop_count=len(crops),
            face_count=total_faces,
            identities=list({r.person_id for _, rr in results for r in rr}),
        )
        return results


def _encode_crop(
    crop: npt.NDArray[np.uint8],
    quality: int = 90,
) -> str:
    """Encode a person crop as base64 JPEG (no downscaling)."""
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed for crop")
    return base64.b64encode(buf.tobytes()).decode("ascii")
