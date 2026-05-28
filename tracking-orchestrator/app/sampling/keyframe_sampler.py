"""KeyframeSampler: periodic and triggered keyframe selection.

Selects at most one keyframe per tracklet per keyframe_min_interval_s
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
from ..storage.base import BboxAnnotationRepository, KeyframeRepository


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the keyframe sampler."""

    # Minimum seconds between periodic samples per tracklet.
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
            tracklet_id="tl-001",
            global_track_id="gt-001",
            camera_id="cam-a",
            minio_key="frames/cam-a/001.jpg",
            captured_at=datetime.now(UTC),
            annotations={"bbox": [10, 20, 100, 200], "person_id": "alice"},
        )

        # Forced trigger sample (identity changed, hazard, dwell_start).
        frame = await sampler.trigger_sample(
            tracklet_id="tl-001",
            global_track_id="gt-001",
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
        bbox_repo: BboxAnnotationRepository | None = None,
    ) -> None:
        self._repo = repo
        self._config = config or SamplerConfig()
        self._bbox_repo = bbox_repo
        # Last time a periodic sample was taken per tracklet_id.
        self._last_sample: dict[str, datetime] = {}

    async def maybe_sample(
        self,
        tracklet_id: str = "",
        global_track_id: str = "",
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
    ) -> TaggedKeyframe | None:
        """Sample a periodic keyframe if the interval has elapsed (WTR3).

        ``ph_id`` is the canonical key (supersedes ``tracklet_id``).
        Callers should pass ``ph_id``; ``tracklet_id`` is kept for
        backward compat.
        """
        if captured_at is None:
            captured_at = datetime.now(UTC)
        if annotations is None:
            annotations = {}
        entity_key = ph_id or tracklet_id

        last = self._last_sample.get(entity_key)
        if last is not None:
            elapsed = (captured_at - last).total_seconds()
            if elapsed < self._config.keyframe_min_interval_s:
                return None

        expires_at = captured_at + timedelta(hours=self._config.periodic_expires_hours)
        keyframe_id = str(uuid.uuid4())
        keyframe = TaggedKeyframe(
            keyframe_id=keyframe_id,
            tracklet_id=entity_key,
            global_track_id=global_track_id or entity_key,
            camera_id=camera_id,
            minio_key=minio_key,
            captured_at=captured_at,
            annotations=annotations,
            tag_reason="periodic",
            expires_at=expires_at,
        )
        await self._repo.save_keyframe(keyframe)
        self._last_sample[entity_key] = captured_at

        if self._bbox_repo is not None and detection_bbox is not None:
            await self._bbox_repo.save_bbox_annotations(
                [
                    BboxAnnotation(
                        keyframe_id=keyframe_id,
                        tracklet_id=entity_key,
                        camera_id=camera_id,
                        x1=detection_bbox[0],
                        y1=detection_bbox[1],
                        x2=detection_bbox[2],
                        y2=detection_bbox[3],
                        detection_confidence=detection_confidence,
                        frame_width=detection_frame_width,
                        frame_height=detection_frame_height,
                        identity_id=detection_identity_id,
                        created_at=datetime.now(UTC),
                    )
                ]
            )

        return keyframe

    async def trigger_sample(
        self,
        tracklet_id: str = "",
        global_track_id: str = "",
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
    ) -> TaggedKeyframe:
        """Force a keyframe sample outside the periodic schedule (WTR3).

        ``ph_id`` is the canonical key. Valid tag_reason values:
        'identity_changed', 'hazard', 'dwell_start'.
        """
        if captured_at is None:
            captured_at = datetime.now(UTC)
        if annotations is None:
            annotations = {}
        entity_key = ph_id or tracklet_id

        expires_at = captured_at + timedelta(days=self._config.trigger_expires_days)
        keyframe_id = str(uuid.uuid4())
        keyframe = TaggedKeyframe(
            keyframe_id=keyframe_id,
            tracklet_id=entity_key,
            global_track_id=global_track_id or entity_key,
            camera_id=camera_id,
            minio_key=minio_key,
            captured_at=captured_at,
            annotations=annotations,
            tag_reason=tag_reason,  # type: ignore[arg-type]
            expires_at=expires_at,
        )
        await self._repo.save_keyframe(keyframe)

        if self._bbox_repo is not None and detection_bbox is not None:
            await self._bbox_repo.save_bbox_annotations(
                [
                    BboxAnnotation(
                        keyframe_id=keyframe_id,
                        tracklet_id=entity_key,
                        camera_id=camera_id,
                        x1=detection_bbox[0],
                        y1=detection_bbox[1],
                        x2=detection_bbox[2],
                        y2=detection_bbox[3],
                        detection_confidence=detection_confidence,
                        frame_width=detection_frame_width,
                        frame_height=detection_frame_height,
                        identity_id=detection_identity_id,
                        created_at=datetime.now(UTC),
                    )
                ]
            )

        return keyframe

    def reset_tracklet(self, tracklet_id: str) -> None:
        """Remove the periodic timer for a terminated entity."""
        self._last_sample.pop(tracklet_id, None)

    def reset_ph(self, ph_id: str) -> None:
        """Remove the periodic timer for a terminated PH (WTR3)."""
        self._last_sample.pop(ph_id, None)
