"""Async HTTP client for the person-identification-service.

Used by the frame processing pipeline to obtain face-based identity
evidence (FaceAnchors) that feed into the Bayesian identity resolver.

Rate-limiting is handled by the caller (frame_pipeline) via a per-camera
cooldown, so this client is stateless — each ``identify`` call is a
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


class FaceIdentificationClient:
    """Async HTTP client for person-identification-service.

    Sends person crops (or a downscaled full frame) to
    ``{base_url}/api/v1/identify`` and returns face detections with
    identity assignments.

    Graceful degradation: if the service is unreachable or returns an
    error, an empty list is returned and the error is logged.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 2.0,
        min_confidence: float = 0.5,
        max_image_dim: int = 640,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._min_confidence = min_confidence
        self._max_dim = max_image_dim
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
                confidence = float(face.get("confidence", 0))
                if confidence < self._min_confidence:
                    continue
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

                crop_results.append(
                    FaceResult(
                        person_id=face.get("person_id", "unknown"),
                        name=face.get("name", "Unknown"),
                        confidence=confidence,
                        bbox_normalized=[nx1, ny1, nx2, ny2],
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

    async def identify(
        self,
        image: npt.NDArray[np.uint8],
        orig_width: int,
        orig_height: int,
    ) -> list[FaceResult]:
        """Identify faces in a full RGB frame (legacy path).

        Prefer :meth:`identify_crops` for better face resolution — this
        method downscales the full frame before sending, which may lose
        small faces.

        Args:
            image: RGB uint8 numpy array (H, W, 3).
            orig_width: Original frame width (for bbox normalisation).
            orig_height: Original frame height (for bbox normalisation).
        """
        if self._client is None:
            logger.warning("FaceIdentificationClient not connected, skipping identify")
            return []

        try:
            image_b64 = _encode_frame(image, self._max_dim)
        except Exception:
            logger.exception("Failed to encode frame for face identification")
            return []

        try:
            resp = await self._client.post(
                f"{self._base_url}/api/v1/identify",
                json={"image": image_b64, "save_guest_images": False},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPError:
            logger.warning("Face identification HTTP error", exc_info=True)
            return []
        except Exception:
            logger.exception("Face identification unexpected error")
            return []

        results: list[FaceResult] = []
        for face in data.get("faces", []):
            confidence = float(face.get("confidence", 0))
            if confidence < self._min_confidence:
                continue
            bbox_px: list[float] = face.get("bbox", [])
            if len(bbox_px) != 4:
                continue
            x1, y1, x2, y2 = bbox_px
            dw = self._max_dim
            dh = self._max_dim
            if orig_width > orig_height:
                dh = int(orig_height * self._max_dim / orig_width)
            else:
                dw = int(orig_width * self._max_dim / orig_height)
            nx1 = x1 / dw
            ny1 = y1 / dh
            nx2 = x2 / dw
            ny2 = y2 / dh
            results.append(
                FaceResult(
                    person_id=face.get("person_id", "unknown"),
                    name=face.get("name", "Unknown"),
                    confidence=confidence,
                    bbox_normalized=[nx1, ny1, nx2, ny2],
                )
            )

        logger.debug(
            "Face identification complete (full frame)",
            face_count=len(results),
            identities=[r.person_id for r in results],
        )
        return results


def _encode_frame(
    image: npt.NDArray[np.uint8],
    max_dim: int,
    quality: int = 85,
) -> str:
    """Downscale an RGB image so its longest side ≤ max_dim, encode as base64 JPEG."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    # OpenCV imencode expects BGR
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


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
