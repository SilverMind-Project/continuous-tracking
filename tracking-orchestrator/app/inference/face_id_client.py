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

    Sends a downscaled JPEG frame to ``{base_url}/api/v1/identify`` and
    returns face detections with identity assignments.

    Graceful degradation: if the service is unreachable or returns an
    error, an empty list is returned and the error is logged.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 2.0,
        min_confidence: float = 0.4,
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

    async def identify(
        self,
        image: npt.NDArray[np.uint8],
        orig_width: int,
        orig_height: int,
    ) -> list[FaceResult]:
        """Identify faces in an RGB image.

        Args:
            image: RGB uint8 numpy array (H, W, 3).
            orig_width: Original frame width (for bbox normalisation).
            orig_height: Original frame height (for bbox normalisation).

        Returns:
            List of FaceResult with normalised bboxes in [0, 1] relative
            to the original frame dimensions.
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
            # Person-id-service returns pixel bboxes in the downscaled image.
            # Normalise to [0,1] in the original frame coordinate space.
            bbox_px: list[float] = face.get("bbox", [])
            if len(bbox_px) != 4:
                continue
            # The downscaled image dimensions
            scale_h = orig_height / max(orig_height, orig_width) * self._max_dim
            scale_w = orig_width / max(orig_height, orig_width) * self._max_dim
            # Actually simpler: person-id-service receives the downscaled image,
            # so its bboxes are in that image's pixel space.  We normalise to
            # [0,1] using the original dimensions because our YOLO detections
            # are already normalised to the original frame.
            x1, y1, x2, y2 = bbox_px
            dw = self._max_dim
            dh = self._max_dim
            # Letterbox scaling used by _encode_frame
            if orig_width > orig_height:
                dh = int(orig_height * self._max_dim / orig_width)
            else:
                dw = int(orig_width * self._max_dim / orig_height)

            # Convert from downscaled pixel space → normalised original space
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
            "Face identification complete",
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
