"""KeyframeSampler: periodic and triggered keyframe selection.

Selects at most one keyframe per tracklet per keyframe_min_interval_s
(periodic) and forces a sample on trigger events (identity_changed,
hazard, dwell_start).

This module does NOT import from transport or pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..domain import TaggedKeyframe
from ..storage.base import KeyframeRepository


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
    ) -> None:
        self._repo = repo
        self._config = config or SamplerConfig()
        # Last time a periodic sample was taken per tracklet_id.
        self._last_sample: dict[str, datetime] = {}

    async def maybe_sample(
        self,
        tracklet_id: str,
        global_track_id: str,
        camera_id: str,
        minio_key: str,
        captured_at: datetime,
        annotations: dict[str, Any],
    ) -> TaggedKeyframe | None:
        """Sample a periodic keyframe if the interval has elapsed.

        Returns None if the minimum interval since the last sample has not
        elapsed. Returns and persists a TaggedKeyframe otherwise.
        """
        last = self._last_sample.get(tracklet_id)
        if last is not None:
            elapsed = (captured_at - last).total_seconds()
            if elapsed < self._config.keyframe_min_interval_s:
                return None

        expires_at = captured_at + timedelta(hours=self._config.periodic_expires_hours)
        keyframe = TaggedKeyframe(
            keyframe_id=str(uuid.uuid4()),
            tracklet_id=tracklet_id,
            global_track_id=global_track_id,
            camera_id=camera_id,
            minio_key=minio_key,
            captured_at=captured_at,
            annotations=annotations,
            tag_reason="periodic",
            expires_at=expires_at,
        )
        await self._repo.save_keyframe(keyframe)
        self._last_sample[tracklet_id] = captured_at
        return keyframe

    async def trigger_sample(
        self,
        tracklet_id: str,
        global_track_id: str,
        camera_id: str,
        minio_key: str,
        captured_at: datetime,
        annotations: dict[str, Any],
        tag_reason: str,
    ) -> TaggedKeyframe:
        """Force a keyframe sample outside the periodic schedule.

        Valid tag_reason values: 'identity_changed', 'hazard', 'dwell_start'.
        The sample is persisted regardless of the periodic interval.
        The periodic timer is NOT reset (the next periodic sample fires
        at the normal interval after the last periodic one).
        """
        expires_at = captured_at + timedelta(days=self._config.trigger_expires_days)
        keyframe = TaggedKeyframe(
            keyframe_id=str(uuid.uuid4()),
            tracklet_id=tracklet_id,
            global_track_id=global_track_id,
            camera_id=camera_id,
            minio_key=minio_key,
            captured_at=captured_at,
            annotations=annotations,
            tag_reason=tag_reason,  # type: ignore[arg-type]
            expires_at=expires_at,
        )
        await self._repo.save_keyframe(keyframe)
        return keyframe

    def reset_tracklet(self, tracklet_id: str) -> None:
        """Remove the periodic timer for a terminated tracklet."""
        self._last_sample.pop(tracklet_id, None)
