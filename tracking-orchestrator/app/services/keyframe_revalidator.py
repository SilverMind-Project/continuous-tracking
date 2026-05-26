"""Background task that drops historical low-confidence bbox annotations (M3).

Scheduled nightly via the orchestrator lifespan. Scans annotations written
in the last 7 days and removes any whose detection_confidence is below
the configured threshold.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from structlog import get_logger

logger = get_logger(__name__)

_REVALIDATE_INTERVAL_S = 86400  # 24 hours


class KeyframeRevalidator:
    """Drops low-confidence bbox annotations on a schedule."""

    def __init__(self, bbox_repo: Any, threshold: float = 0.5, window_days: int = 7) -> None:
        self._bbox_repo: Any = bbox_repo
        self._threshold = threshold
        self._window_days = window_days
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        """Run the revalidation loop until stopped."""
        logger.info(
            "keyframe_revalidator_started",
            threshold=self._threshold,
            window_days=self._window_days,
            interval_s=_REVALIDATE_INTERVAL_S,
        )
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("keyframe_revalidate_failed")
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=_REVALIDATE_INTERVAL_S
                )

    async def stop(self) -> None:
        self._stop_event.set()

    async def run_once(self) -> int:
        """Scan and drop low-confidence annotations. Returns count of deleted rows."""
        cutoff = datetime.now(UTC) - timedelta(days=self._window_days)
        n_dropped: int = await self._bbox_repo.delete_annotations_below_confidence(
            threshold=self._threshold,
            since=cutoff,
        )
        if n_dropped > 0:
            logger.info(
                "keyframe_revalidate_pass",
                dropped=n_dropped,
                threshold=self._threshold,
                window_days=self._window_days,
            )
        return n_dropped
