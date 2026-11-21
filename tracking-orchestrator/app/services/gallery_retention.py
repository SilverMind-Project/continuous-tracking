"""Background retention service for governed ReID gallery rows.

Body-appearance embeddings and crops are kept only as long as they remain useful.
Because ``gallery_review_events.entry_id`` references ``reid_gallery(id)`` (NOT NULL),
deleting a reviewed row would either fail or erase its audit history. Retention instead
transitions aged rows to the terminal ``expired`` state via
:meth:`GalleryRepository.expire_past_retention`, nulling their embedding vector,
deleting their dedicated crop object, and recording a ``system:retention`` audit event.

``rejected`` rows are already tombstones (vector nulled, crop deleted) and are not
touched. In ``shadow`` mode, retention only counts how many rows would expire per state
and emits ``cts_gallery_retention_rows_total{state, mode="shadow"}`` without mutating
any row or deleting any crop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from structlog import get_logger

from app.observability import metrics as _metrics
from app.observability.metrics import GalleryRetentionModeLabel, GalleryRetentionState
from app.storage.gallery import GalleryRepository

logger = get_logger(__name__)


class CropDeleter(Protocol):
    """Minimal object-storage contract needed to delete expired gallery crops."""

    async def delete_object(self, key: str) -> None: ...


GalleryRetentionMode = Literal["off", "shadow", "enabled"]
_VALID_MODES: frozenset[str] = frozenset({"off", "shadow", "enabled"})


@dataclass(frozen=True)
class GalleryRetentionConfig:
    """Retention windows (days) and cadence for ``reid_gallery`` lifecycle expiration."""

    mode: GalleryRetentionMode = "enabled"
    auto_verified_days: int = 14
    pending_review_days: int = 30
    operator_verified_days: int = 90
    interval_s: float = 86400.0
    batch_size: int = 1000

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"reid_gallery.retention.mode must be one of {sorted(_VALID_MODES)}, "
                f"got {self.mode!r}"
            )
        if self.auto_verified_days < 1:
            raise ValueError("reid_gallery.retention.auto_verified_days must be >= 1")
        if self.pending_review_days < 1:
            raise ValueError("reid_gallery.retention.pending_review_days must be >= 1")
        if self.operator_verified_days < 1:
            raise ValueError("reid_gallery.retention.operator_verified_days must be >= 1")
        if self.interval_s <= 0.0:
            raise ValueError("reid_gallery.retention.interval_s must be > 0")
        if self.batch_size < 1:
            raise ValueError("reid_gallery.retention.batch_size must be >= 1")

    def state_windows(self) -> tuple[tuple[GalleryRetentionState, int], ...]:
        """Ordered ``(state, retention_days)`` pairs evaluated on each retention sweep."""
        return (
            (GalleryRetentionState.AUTO_VERIFIED, self.auto_verified_days),
            (GalleryRetentionState.PENDING_REVIEW, self.pending_review_days),
            (GalleryRetentionState.OPERATOR_VERIFIED, self.operator_verified_days),
        )


@dataclass(frozen=True)
class RetentionTickResult:
    """Counts recorded by one :meth:`GalleryRetentionService.run_once` execution."""

    mode: GalleryRetentionMode
    by_state: dict[str, int]
    crops_deleted: int
    crop_delete_failures: int


class GalleryRetentionService:
    """Periodic job that counts (shadow) or expires (enabled) aged gallery rows."""

    def __init__(
        self,
        *,
        gallery_repo: GalleryRepository,
        config: GalleryRetentionConfig,
        crop_storage: CropDeleter | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repo = gallery_repo
        self._config = config
        self._crop_storage = crop_storage
        self._clock = clock
        self._stop_event = asyncio.Event()
        # Crop keys whose MinIO delete failed on an earlier tick; retried on the next run.
        self._pending_crop_deletes: set[str] = set()
        self._warned_no_storage = False

    @property
    def pending_crop_deletes(self) -> frozenset[str]:
        """Crop keys awaiting retry after an earlier storage deletion failure."""
        return frozenset(self._pending_crop_deletes)

    async def run(self) -> None:
        if self._config.mode == "off":
            logger.info("gallery_retention_disabled")
            return

        logger.info(
            "gallery_retention_started",
            mode=self._config.mode,
            auto_verified_days=self._config.auto_verified_days,
            pending_review_days=self._config.pending_review_days,
            operator_verified_days=self._config.operator_verified_days,
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
                logger.exception("gallery_retention_tick_failed")

        logger.info("gallery_retention_stopped")

    async def run_once(self, *, now: datetime | None = None) -> RetentionTickResult:
        """Execute one retention sweep according to ``config.mode``."""
        mode = self._config.mode
        by_state: dict[str, int] = {}
        if mode == "off":
            return RetentionTickResult(
                mode=mode, by_state=by_state, crops_deleted=0, crop_delete_failures=0
            )

        ref_now = now if now is not None else self._clock()
        mode_label = (
            GalleryRetentionModeLabel.ENABLED
            if mode == "enabled"
            else GalleryRetentionModeLabel.SHADOW
        )

        crops_deleted = 0
        crop_delete_failures = 0

        if mode == "enabled" and self._pending_crop_deletes:
            retried_ok, retried_fail = await self._delete_crop_keys(
                sorted(self._pending_crop_deletes)
            )
            crops_deleted += retried_ok
            crop_delete_failures += retried_fail

        for state, retention_days in self._config.state_windows():
            cutoff = ref_now - timedelta(days=retention_days)
            if mode == "shadow":
                count = await self._repo.count_past_retention(
                    state=str(state),
                    seen_before=cutoff,
                )
                by_state[str(state)] = count
                if count > 0:
                    _metrics.metrics.cts_gallery_retention_rows_total.labels(
                        state=state, mode=mode_label
                    ).inc(count)
                    logger.info(
                        "gallery_retention_shadow_count",
                        state=str(state),
                        rows=count,
                        cutoff=cutoff.isoformat(),
                    )
                continue

            total_expired = 0
            while True:
                before = await self._repo.count_past_retention(
                    state=str(state),
                    seen_before=cutoff,
                )
                if before == 0:
                    break
                crop_keys = await self._repo.expire_past_retention(
                    state=str(state),
                    seen_before=cutoff,
                    limit=self._config.batch_size,
                )
                after = await self._repo.count_past_retention(
                    state=str(state),
                    seen_before=cutoff,
                )
                expired_in_batch = max(len(crop_keys), before - after)
                total_expired += expired_in_batch
                deleted_ok, deleted_fail = await self._delete_crop_keys(crop_keys)
                crops_deleted += deleted_ok
                crop_delete_failures += deleted_fail
                if expired_in_batch == 0 or after == 0:
                    break

            by_state[str(state)] = total_expired
            if total_expired > 0:
                _metrics.metrics.cts_gallery_retention_rows_total.labels(
                    state=state, mode=mode_label
                ).inc(total_expired)
                logger.info(
                    "gallery_retention_expired",
                    state=str(state),
                    rows=total_expired,
                    cutoff=cutoff.isoformat(),
                )

        return RetentionTickResult(
            mode=mode,
            by_state=by_state,
            crops_deleted=crops_deleted,
            crop_delete_failures=crop_delete_failures,
        )

    async def stop(self) -> None:
        self._stop_event.set()

    async def _delete_crop_keys(self, crop_keys: list[str]) -> tuple[int, int]:
        """Delete each non-empty crop key; retain failures for retry on the next tick."""
        non_empty = [k for k in crop_keys if k]
        if not non_empty:
            return (0, 0)
        if self._crop_storage is None:
            if not self._warned_no_storage:
                self._warned_no_storage = True
                logger.warning("gallery_retention_crop_storage_unconfigured")
            return (0, 0)

        deleted = 0
        failed = 0
        for key in non_empty:
            try:
                await self._crop_storage.delete_object(key)
                self._pending_crop_deletes.discard(key)
                deleted += 1
            except Exception:
                self._pending_crop_deletes.add(key)
                failed += 1
                _metrics.metrics.cts_gallery_retention_crop_delete_failures_total.inc()
                logger.warning(
                    "gallery_retention_crop_delete_failed",
                    crop_key=key,
                    exc_info=True,
                )
        return (deleted, failed)
