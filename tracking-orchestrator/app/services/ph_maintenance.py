"""Background maintenance for Person Hypotheses."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from structlog import get_logger

from ..storage.base import PHRepositoryProtocol

logger = get_logger(__name__)


@dataclass(frozen=True)
class PHUnknownPurgeConfig:
    enabled: bool
    older_than_days: int
    interval_s: float
    batch_size: int


class PHMaintenanceService:
    """Runs configured PH cleanup jobs.

    Unknown PH purge deletes only closed PHs with no committed identity whose
    last observation is older than the configured cutoff. Active unknown tracks
    are never purged by this job.
    """

    def __init__(self, repo: PHRepositoryProtocol, config: PHUnknownPurgeConfig) -> None:
        self._repo = repo
        self._config = config
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        if not self._config.enabled:
            logger.info("ph_maintenance_disabled")
            return

        logger.info(
            "ph_maintenance_started",
            older_than_days=self._config.older_than_days,
            interval_s=self._config.interval_s,
            batch_size=self._config.batch_size,
        )
        first_run = True
        while not self._stop_event.is_set():
            try:
                if first_run:
                    first_run = False
                else:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._config.interval_s,
                    )
                    continue
                await self.run_once()
            except TimeoutError:
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("ph_maintenance_tick_failed")

        logger.info("ph_maintenance_stopped")

    async def run_once(self) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=self._config.older_than_days)
        deleted = await self._repo.purge_unknown_older_than(cutoff, limit=self._config.batch_size)
        if deleted:
            logger.info(
                "ph_maintenance_unknown_purged",
                deleted=deleted,
                cutoff=cutoff.isoformat(),
            )
        return deleted

    async def stop(self) -> None:
        self._stop_event.set()
