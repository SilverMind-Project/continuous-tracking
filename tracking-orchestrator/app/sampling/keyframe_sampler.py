"""KeyframeSampler: periodic and triggered keyframe selection.

Selects at most one keyframe per PH per keyframe_min_interval_s
(periodic) and forces a sample on trigger events (identity_changed,
hazard, dwell_start).

This module does NOT import from transport or pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ..domain import BboxAnnotation, TaggedKeyframe
from ..storage.base import KeyframeRepository


@dataclass(frozen=True)
class FrameBbox:
    """One detection's bounding box within a sampled frame.

    A keyframe image typically contains every person visible in the frame,
    so a keyframe carries one ``FrameBbox`` per detection -- not just the
    bbox of the PH that triggered the sample.
    """

    ph_id: str
    bbox: tuple[float, float, float, float]
    confidence: float
    identity_id: str | None = None


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the keyframe sampler."""

    # Minimum seconds between periodic samples per PH.
    keyframe_min_interval_s: float = 30.0

    # Retention for periodic samples (hours).
    periodic_expires_hours: int = 72

    # Retention for trigger-reason samples (days).
    trigger_expires_days: int = 30


class KeyframeSampler:
    """Selects and persists periodic and triggered keyframes.

    Usage::

        sampler = KeyframeSampler(repo=keyframe_repo)

        # Periodic sampling — returns None if the interval hasn't elapsed.
        frame = await sampler.maybe_sample(
            ph_id="ph-001",
            camera_id="cam-a",
            minio_key="frames/cam-a/001.jpg",
            captured_at=datetime.now(UTC),
            annotations={"bbox": [10, 20, 100, 200], "person_id": "alice"},
        )

        # Forced trigger sample (identity changed, hazard, dwell_start).
        frame = await sampler.trigger_sample(
            ph_id="ph-001",
            camera_id="cam-a",
            minio_key="frames/cam-a/002.jpg",
            captured_at=datetime.now(UTC),
            annotations={"bbox": [10, 20, 100, 200]},
            tag_reason="identity_changed",
        )
    """

    def __init__(
        self,
        repo: KeyframeRepository,
        config: SamplerConfig | None = None,
    ) -> None:
        self._repo = repo
        self._config = config or SamplerConfig()
        # Last time a periodic sample was taken per PH.
        self._last_sample: dict[str, datetime] = {}

    async def maybe_sample(
        self,
        *,
        ph_id: str = "",
        camera_id: str = "",
        minio_key: str = "",
        captured_at: datetime | None = None,
        annotations: dict[str, Any] | None = None,
        detection_bbox: tuple[float, float, float, float] | None = None,
        detection_confidence: float = 0.0,
        detection_frame_width: int = 0,
        detection_frame_height: int = 0,
        detection_identity_id: str | None = None,
        frame_bboxes: list[FrameBbox] | None = None,
    ) -> TaggedKeyframe | None:
        """Sample a periodic keyframe if the interval has elapsed.

        When ``frame_bboxes`` is provided it is the authoritative set of
        bounding boxes for the whole frame (one per detected person) and the
        single ``detection_*`` arguments are ignored. This lets the keyframe
        carry every person visible in the image rather than only the PH that
        triggered the sample.
        """
        if captured_at is None:
            captured_at = datetime.now(UTC)
        if annotations is None:
            annotations = {}
        entity_key = ph_id
        if not entity_key:
            raise ValueError("ph_id is required for keyframe sampling")

        last = self._last_sample.get(entity_key)
        if last is not None:
            elapsed = (captured_at - last).total_seconds()
            if elapsed < self._config.keyframe_min_interval_s:
                return None

        expires_at = captured_at + timedelta(hours=self._config.periodic_expires_hours)
        keyframe_id = str(uuid.uuid4())
        keyframe = TaggedKeyframe(
            keyframe_id=keyframe_id,
            ph_id=entity_key,
            camera_id=camera_id,
            minio_key=minio_key,
            captured_at=captured_at,
            annotations=annotations,
            tag_reason="periodic",
            expires_at=expires_at,
        )
        bbox_annotations = _bbox_annotations_for_sample(
            keyframe=keyframe,
            frame_bboxes=frame_bboxes,
            detection_bbox=detection_bbox,
            detection_confidence=detection_confidence,
            detection_frame_width=detection_frame_width,
            detection_frame_height=detection_frame_height,
            detection_identity_id=detection_identity_id,
        )
        await self._repo.save_keyframe_with_bbox_annotations(keyframe, bbox_annotations)
        self._last_sample[entity_key] = captured_at

        return keyframe

    async def trigger_sample(
        self,
        *,
        ph_id: str = "",
        camera_id: str = "",
        minio_key: str = "",
        captured_at: datetime | None = None,
        annotations: dict[str, Any] | None = None,
        tag_reason: str = "",
        detection_bbox: tuple[float, float, float, float] | None = None,
        detection_confidence: float = 0.0,
        detection_frame_width: int = 0,
        detection_frame_height: int = 0,
        detection_identity_id: str | None = None,
        frame_bboxes: list[FrameBbox] | None = None,
    ) -> TaggedKeyframe:
        """Force a keyframe sample outside the periodic schedule.

        Valid tag_reason values:
        'identity_changed', 'hazard', 'dwell_start'.

        When ``frame_bboxes`` is provided it is the authoritative set of
        bounding boxes for the whole frame (one per detected person) and the
        single ``detection_*`` arguments are ignored.
        """
        if captured_at is None:
            captured_at = datetime.now(UTC)
        if annotations is None:
            annotations = {}
        entity_key = ph_id
        if not entity_key:
            raise ValueError("ph_id is required for keyframe sampling")

        expires_at = captured_at + timedelta(days=self._config.trigger_expires_days)
        keyframe_id = str(uuid.uuid4())
        keyframe = TaggedKeyframe(
            keyframe_id=keyframe_id,
            ph_id=entity_key,
            camera_id=camera_id,
            minio_key=minio_key,
            captured_at=captured_at,
            annotations=annotations,
            tag_reason=tag_reason,  # type: ignore[arg-type]
            expires_at=expires_at,
        )
        bbox_annotations = _bbox_annotations_for_sample(
            keyframe=keyframe,
            frame_bboxes=frame_bboxes,
            detection_bbox=detection_bbox,
            detection_confidence=detection_confidence,
            detection_frame_width=detection_frame_width,
            detection_frame_height=detection_frame_height,
            detection_identity_id=detection_identity_id,
        )
        await self._repo.save_keyframe_with_bbox_annotations(keyframe, bbox_annotations)

        return keyframe

    def reset_ph(self, ph_id: str) -> None:
        """Remove the periodic timer for a terminated PH."""
        self._last_sample.pop(ph_id, None)


def _bbox_annotations_for_sample(
    *,
    keyframe: TaggedKeyframe,
    frame_bboxes: list[FrameBbox] | None,
    detection_bbox: tuple[float, float, float, float] | None,
    detection_confidence: float,
    detection_frame_width: int,
    detection_frame_height: int,
    detection_identity_id: str | None,
) -> list[BboxAnnotation]:
    """Build the bbox annotations to persist alongside a keyframe.

    When ``frame_bboxes`` is supplied, persist one annotation per detected
    person in the frame so the keyframe carries every visible identity.
    Otherwise fall back to the single triggering detection.
    """
    now = datetime.now(UTC)
    if frame_bboxes is not None:
        return [
            BboxAnnotation(
                keyframe_id=keyframe.keyframe_id,
                ph_id=fb.ph_id,
                camera_id=keyframe.camera_id,
                x1=fb.bbox[0],
                y1=fb.bbox[1],
                x2=fb.bbox[2],
                y2=fb.bbox[3],
                detection_confidence=fb.confidence,
                frame_width=detection_frame_width,
                frame_height=detection_frame_height,
                identity_id=fb.identity_id,
                created_at=now,
            )
            for fb in frame_bboxes
        ]
    if detection_bbox is None:
        return []
    return [
        BboxAnnotation(
            keyframe_id=keyframe.keyframe_id,
            ph_id=keyframe.ph_id,
            camera_id=keyframe.camera_id,
            x1=detection_bbox[0],
            y1=detection_bbox[1],
            x2=detection_bbox[2],
            y2=detection_bbox[3],
            detection_confidence=detection_confidence,
            frame_width=detection_frame_width,
            frame_height=detection_frame_height,
            identity_id=detection_identity_id,
            created_at=now,
        )
    ]
