"""Async HTTP client for the person-identification-service.

Used by the frame processing pipeline to obtain face-based identity
evidence (FaceAnchors) that feed into the Bayesian identity resolver.

Rate limiting is handled by the caller (``FaceIdentityStage``, per-track cooldown),
so this client is stateless: ``identify_crops`` sends one HTTP POST per crop.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import cv2
import httpx
import numpy as np
import numpy.typing as npt
from structlog import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FaceQualityReading:
    """Face image quality reported by person-identification-service, all components in [0, 1].

    The service computes it from the aligned face crop; CTS only carries it. ``version`` names the
    scorer, so a policy or calibration that depends on the score binds to that version.
    """

    score: float
    resolution: float
    sharpness: float
    detection: float
    pitch: float
    version: str


@dataclass(frozen=True)
class FaceResult:
    """A single face identification result from person-identification-service."""

    person_id: str
    name: str
    # Normalised bounding box [x1, y1, x2, y2] in [0, 1].
    bbox_normalized: list[float]
    # Three-valued recognition state.
    recognition_state: str = "recognized"
    # Nearest centroid person_id, even below threshold.
    best_candidate_id: str | None = None
    # Raw cosine similarity to the best candidate; the only similarity field.
    raw_similarity: float = 0.0
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
    # None for a service that does not score face quality, or a malformed reading.
    quality: FaceQualityReading | None = None
    # Profile of the calibration behind calibrated_confidence ("single_frame"); "" when none.
    calibration_profile: str = ""
    # The face's ArcFace embedding (unit length, 512 values) when the call asked for it, for face
    # templates. Biometric data: held in memory only, never logged (excluded from repr) or
    # persisted. None when not requested or malformed.
    embedding: npt.NDArray[np.float32] | None = field(default=None, repr=False, compare=False)
    embedding_model_version: str = ""


# The calibration profile for per-face (single frame) calibrated confidence.
SINGLE_FRAME_PROFILE = "single_frame"
# The calibration profile for aggregated face templates (/match-template).
TEMPLATE_PROFILE = "template"
EMBEDDING_DIM = 512
_F16_LE = np.dtype("<f2")


@dataclass(frozen=True)
class TemplateMember:
    """One face sent to ``/match-template``: its embedding (never logged) and its quality."""

    embedding: npt.NDArray[np.float32] = field(repr=False, compare=False)
    quality: FaceQualityReading


@dataclass(frozen=True)
class TemplateQualityReading:
    """Template quality features: person-id's means over the three best members by score."""

    score: float
    resolution: float
    sharpness: float


@dataclass(frozen=True)
class TemplateMatchResult:
    """person-identification-service's verdict on a face template (several faces of one PH)."""

    person_id: str
    recognition_state: str
    best_candidate_id: str | None
    # Raw cosine similarity of the template to the nearest / second-nearest centroid.
    raw_similarity: float
    second_best_similarity: float | None
    member_count: int
    aggregator_version: str
    quality: TemplateQualityReading
    # Face-quality scorer version of the members.
    quality_version: str
    # Calibrated probability from the template-profile artifact; None without one.
    calibrated_confidence: float | None
    # "template" when a template artifact is loaded, "" otherwise.
    calibration_profile: str
    calibration_status: str


@dataclass(frozen=True)
class CaptureContext:
    """Where a crop came from and how it was cut, sent with every identify call.

    person-identification-service samples faces into its calibration set from these calls, and the
    crop settings shape the similarity distribution it calibrates, so they travel with the crop.
    """

    camera_id: str
    # Frame capture time (timezone-aware), not wall clock.
    captured_at: datetime
    crop_pad_top_frac: float
    crop_pad_side_frac: float


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

    async def list_enrolled_members(self) -> list[tuple[str, str]] | None:
        """Fetch the active enrolled member roster from person-identification-service.

        Returns a list of ``(person_id, display_name)`` tuples on HTTP 200, or
        ``None`` when disconnected or when the upstream request fails so callers
        never confuse a transient outage with an empty enrollment roster.
        """
        if self._client is None:
            return None
        url = f"{self._base_url}/api/v1/members"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("face_id_members_fetch_failed", error=str(exc))
            return None
        raw_members = payload.get("members") if isinstance(payload, dict) else None
        if not isinstance(raw_members, list):
            return None
        members: list[tuple[str, str]] = []
        for item in raw_members:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("person_id") or "").strip()
            name = str(item.get("name") or pid).strip()
            if pid:
                members.append((pid, name))
        return members

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def identify_crops(
        self,
        crops: list[npt.NDArray[np.uint8]],
        crop_bboxes_norm: list[tuple[float, float, float, float]],
        context: CaptureContext,
        *,
        return_embedding: bool = False,
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
            context: Camera, capture time, and crop padding of these crops; sent as the
                request's ``capture_context`` together with this client's JPEG quality.
            return_embedding: Ask the service for each face's embedding (face templates);
                parsed into ``FaceResult.embedding`` and never logged.

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
        capture_context = {
            "camera_id": context.camera_id,
            "captured_at": context.captured_at.isoformat(),
            "crop_pad_top_frac": context.crop_pad_top_frac,
            "crop_pad_side_frac": context.crop_pad_side_frac,
            "crop_jpeg_quality": self._jpeg_quality,
        }
        results: list[CropQueryResult] = []
        crop_with_bboxes = zip(crops, crop_bboxes_norm, strict=True)
        for idx, (crop, crop_bbox_norm) in enumerate(crop_with_bboxes):
            results.append(
                await self._identify_one(
                    client, idx, crop, crop_bbox_norm, capture_context, return_embedding
                )
            )

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
        capture_context: dict[str, Any],
        return_embedding: bool,
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
                json={
                    "image": crop_b64,
                    "save_guest_images": False,
                    "capture_context": capture_context,
                    "return_embedding": return_embedding,
                },
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

    async def match_template(
        self, members: Sequence[TemplateMember], embedding_model_version: str
    ) -> TemplateMatchResult | None:
        """Match several faces of one person as a template (``POST /api/v1/match-template``).

        Returns None when disconnected, on any HTTP error (including a 422 contract rejection),
        or for a malformed response; the caller counts and moves on. No side effects upstream.
        """
        if self._client is None:
            return None
        body = {
            "members": [
                {
                    "embedding_f16_b64": _encode_embedding(m.embedding),
                    "quality": {
                        "score": m.quality.score,
                        "resolution": m.quality.resolution,
                        "sharpness": m.quality.sharpness,
                        "detection": m.quality.detection,
                        "pitch": m.quality.pitch,
                        "version": m.quality.version,
                    },
                }
                for m in members
            ],
            "embedding_model_version": embedding_model_version,
        }
        try:
            resp = await self._client.post(f"{self._base_url}/api/v1/match-template", json=body)
            resp.raise_for_status()
            return _parse_template_match(resp.json())
        except httpx.HTTPStatusError as exc:
            code = _error_code(exc.response)
            logger.warning(
                "face_template_match_rejected",
                status_code=exc.response.status_code,
                code=code,
                member_count=len(members),
            )
            return None
        except httpx.HTTPError:
            logger.warning("face_template_match_http_error", member_count=len(members))
            return None
        except (KeyError, TypeError, ValueError):
            logger.warning("face_template_match_malformed_response", member_count=len(members))
            return None


def _error_code(response: httpx.Response) -> str | None:
    """The structured ``detail.code`` of an error response, when it has one."""
    try:
        payload = response.json()
    except ValueError:
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return str(detail["code"])
    return None


def _parse_template_match(data: dict[str, Any]) -> TemplateMatchResult:
    """Map the service's response; raises KeyError/TypeError/ValueError when malformed."""
    quality = data["template_quality"]
    calibrated = data.get("calibrated_confidence")
    second = data.get("second_best_similarity")
    return TemplateMatchResult(
        person_id=str(data["person_id"]),
        recognition_state=str(data["recognition_state"]),
        best_candidate_id=data.get("best_candidate_id"),
        raw_similarity=float(data["raw_similarity"]),
        second_best_similarity=float(second) if second is not None else None,
        member_count=int(data["member_count"]),
        aggregator_version=str(data["aggregator_version"]),
        quality=TemplateQualityReading(
            score=float(quality["score"]),
            resolution=float(quality["resolution"]),
            sharpness=float(quality["sharpness"]),
        ),
        quality_version=str(data["quality_version"]),
        calibrated_confidence=float(calibrated) if calibrated is not None else None,
        calibration_profile=str(data.get("calibration_profile", "")),
        calibration_status=str(data.get("calibration_status", "degraded_missing")),
    )


def _encode_embedding(embedding: npt.NDArray[np.float32]) -> str:
    """The wire form person-id expects: 512 little-endian float16 values, base64."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(f"embedding must have {EMBEDDING_DIM} values")
    return base64.b64encode(vector.astype(_F16_LE).tobytes()).decode("ascii")


def _decode_embedding(encoded: Any) -> npt.NDArray[np.float32] | None:
    """Decode the service's embedding; None (logged without content) when malformed."""
    if not isinstance(encoded, str):
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        logger.warning("face_embedding_malformed", reason="base64")
        return None
    if len(raw) != EMBEDDING_DIM * _F16_LE.itemsize:
        logger.warning("face_embedding_malformed", reason="length")
        return None
    vector = np.frombuffer(raw, dtype=_F16_LE).astype(np.float32)
    if not np.all(np.isfinite(vector)):
        logger.warning("face_embedding_malformed", reason="non_finite")
        return None
    return vector


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

    # An older service sends only the legacy "similarity" alias.
    raw_sim = float(face.get("raw_similarity", face.get("similarity", 0)))
    cal_conf_raw = face.get("calibrated_confidence")
    interocular_raw = face.get("interocular_px")
    return FaceResult(
        person_id=face.get("person_id", "unknown"),
        name=face.get("name", "Unknown"),
        bbox_normalized=[nx1, ny1, nx2, ny2],
        recognition_state=str(face.get("recognition_state", "recognized")),
        best_candidate_id=face.get("best_candidate_id"),
        raw_similarity=raw_sim,
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
        quality=_parse_quality(face.get("quality")),
        calibration_profile=str(face.get("calibration_profile", "")),
        embedding=_decode_embedding(face.get("embedding_f16_b64")),
        embedding_model_version=str(face.get("embedding_model_version") or ""),
    )


_QUALITY_COMPONENTS = ("score", "resolution", "sharpness", "detection", "pitch")


def _parse_quality(raw: Any) -> FaceQualityReading | None:
    """Parse the service's quality object; anything malformed is treated as unmeasured."""
    if not isinstance(raw, dict):
        return None
    try:
        values = {name: float(raw[name]) for name in _QUALITY_COMPONENTS}
        version = str(raw["version"])
    except (KeyError, TypeError, ValueError):
        logger.warning("face_quality_malformed", keys=sorted(raw))
        return None
    if not version or not all(0.0 <= v <= 1.0 for v in values.values()):
        logger.warning("face_quality_out_of_range", version=version)
        return None
    return FaceQualityReading(version=version, **values)


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
