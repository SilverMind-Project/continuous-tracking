"""Per-identity standing height profile learner.

Learns robust per-identity height profiles (median + 1.4826 * MAD, floored at 0.03 m)
from standing height samples collected during high-trust intervals:
- Trajectory-grade, unrevoked, non-swap-suspected wearable associations.
- Direct-face-authority identity decisions (free of conflict).

Requires at least ``min_samples`` (200) accepted observations spanning at least
``min_days`` (2 distinct UTC days) and at least ``min_cameras`` (2 distinct cameras)
before promoting an identity to a ``learned`` profile.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Protocol

import asyncpg
from structlog import get_logger

from app.domain import IdentityHeightProfile
from app.storage.height_profiles import IdentityHeightProfileRepository

logger = get_logger(__name__)

_MAD_TO_SIGMA: float = 1.4826


@dataclass(frozen=True)
class AttestedHeightSample:
    """One standing height sample tied to a high-trust identity interval."""

    identity_id: str
    camera_id: str
    observed_at: datetime
    height_m: float
    height_sigma_m: float


class HeightSampleSource(Protocol):
    """Source of standing height samples attached to attested/authoritative identities."""

    async def list_attested_samples(
        self, *, since: datetime, until: datetime
    ) -> list[AttestedHeightSample]:
        """Return samples in ``[since, until)`` backed by trajectory wearables or direct face."""
        ...


class InMemoryHeightSampleSource:
    """In-memory :class:`HeightSampleSource` peer for unit tests."""

    def __init__(self, samples: Sequence[AttestedHeightSample] = ()) -> None:
        self._samples = list(samples)

    def add(self, sample: AttestedHeightSample) -> None:
        self._samples.append(sample)

    async def list_attested_samples(
        self, *, since: datetime, until: datetime
    ) -> list[AttestedHeightSample]:
        return [s for s in self._samples if since <= s.observed_at < until]


class PostgresHeightSampleSource:
    """Queries ``ph_height_samples`` joined to trajectory wearables and direct-face authority."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        direct_face_window_s: float = 30.0,
    ) -> None:
        self._pool = pool
        self._direct_face_window_s = direct_face_window_s

    async def list_attested_samples(
        self, *, since: datetime, until: datetime
    ) -> list[AttestedHeightSample]:
        sql = """
        WITH wearable_samples AS (
            SELECT
                wa.identity_id,
                s.camera_id,
                s.observed_at,
                s.height_m,
                s.height_sigma_m,
                s.ph_id
            FROM continuous_tracking.ph_height_samples s
            INNER JOIN continuous_tracking.wearable_associations wa
                ON wa.ph_id = s.ph_id
               AND wa.attestation_grade = 'trajectory'
               AND wa.revoked_at IS NULL
               AND wa.swap_suspected IS FALSE
               AND s.observed_at >= wa.started_at
               AND (wa.ended_at IS NULL OR s.observed_at <= wa.ended_at)
            INNER JOIN continuous_tracking.identities i
                ON i.identity_id = wa.identity_id
               AND i.is_active IS TRUE
            WHERE s.observed_at >= $1 AND s.observed_at < $2
        ),
        face_samples AS (
            SELECT
                d.effective_identity_id AS identity_id,
                s.camera_id,
                s.observed_at,
                s.height_m,
                s.height_sigma_m,
                s.ph_id
            FROM continuous_tracking.ph_height_samples s
            INNER JOIN continuous_tracking.identity_decisions d
                ON d.ph_id = s.ph_id
               AND d.authority = 'direct_face'
               AND d.effective_identity_id IS NOT NULL
               AND d.conflict_kind IS NULL
               AND ABS(EXTRACT(EPOCH FROM (s.observed_at - d.captured_at))) <= $3
            INNER JOIN continuous_tracking.identities i
                ON i.identity_id = d.effective_identity_id
               AND i.is_active IS TRUE
            WHERE s.observed_at >= $1 AND s.observed_at < $2
        ),
        unioned AS (
            SELECT * FROM wearable_samples
            UNION
            SELECT * FROM face_samples
        )
        SELECT DISTINCT ON (identity_id, ph_id, camera_id, observed_at)
            identity_id,
            camera_id,
            observed_at,
            height_m,
            height_sigma_m
        FROM unioned
        ORDER BY identity_id, ph_id, camera_id, observed_at ASC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, since, until, self._direct_face_window_s)
        return [
            AttestedHeightSample(
                identity_id=str(r["identity_id"]),
                camera_id=str(r["camera_id"]),
                observed_at=r["observed_at"],
                height_m=float(r["height_m"]),
                height_sigma_m=float(r["height_sigma_m"]),
            )
            for r in rows
        ]


@dataclass(frozen=True)
class HeightProfileLearnerConfig:
    """Thresholds and cadence for learning per-identity standing height profiles."""

    enabled: bool = True
    min_samples: int = 200
    min_days: int = 2
    min_cameras: int = 2
    min_sigma_m: float = 0.03
    lookback_days: int = 14
    interval_s: float = 3600.0

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if self.min_days < 1:
            raise ValueError("min_days must be >= 1")
        if self.min_cameras < 1:
            raise ValueError("min_cameras must be >= 1")
        if self.min_sigma_m <= 0.0:
            raise ValueError("min_sigma_m must be > 0")
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if self.interval_s <= 0.0:
            raise ValueError("interval_s must be > 0")


@dataclass(frozen=True)
class CameraHeightSummary:
    """Per-camera height statistics for one identity during a learning pass."""

    camera_id: str
    samples: int
    median_height_m: float


@dataclass(frozen=True)
class HeightLearnResult:
    """Outcome of one learning pass across all identities with attested samples."""

    updated_identities: tuple[str, ...]
    skipped_identities: dict[str, str]
    camera_spread_by_identity_m: dict[str, float]


def compute_robust_height_stats(
    heights_m: Sequence[float],
    *,
    min_sigma_m: float = 0.03,
) -> tuple[float, float]:
    """Return ``(median_m, sigma_m)`` where ``sigma_m = max(min_sigma_m, 1.4826 * MAD)``."""
    med = float(median(heights_m))
    abs_devs = [abs(h - med) for h in heights_m]
    mad = float(median(abs_devs))
    sigma_m = max(min_sigma_m, _MAD_TO_SIGMA * mad)
    return med, sigma_m


class HeightProfileLearner:
    """Periodically learns per-identity standing height profiles from attested samples."""

    def __init__(
        self,
        *,
        sample_source: HeightSampleSource,
        profile_repo: IdentityHeightProfileRepository,
        config: HeightProfileLearnerConfig | None = None,
        on_profiles_updated: Callable[[], None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._source = sample_source
        self._repo = profile_repo
        self._config = config or HeightProfileLearnerConfig()
        self._on_updated = on_profiles_updated
        self._clock = clock
        self._stop_event = asyncio.Event()

    async def run_once(self, *, now: datetime | None = None) -> HeightLearnResult:
        """Compute and upsert learned profiles for every identity meeting the bar."""
        as_of = now or self._clock()
        since = as_of - timedelta(days=self._config.lookback_days)
        samples = await self._source.list_attested_samples(since=since, until=as_of)

        by_identity: dict[str, list[AttestedHeightSample]] = defaultdict(list)
        for sample in samples:
            if 0.9 <= sample.height_m <= 2.2:
                by_identity[sample.identity_id].append(sample)

        updated: list[str] = []
        skipped: dict[str, str] = {}
        spreads: dict[str, float] = {}

        for identity_id in sorted(by_identity):
            id_samples = by_identity[identity_id]
            days = {s.observed_at.astimezone(UTC).date() for s in id_samples}
            by_cam: dict[str, list[float]] = defaultdict(list)
            for s in id_samples:
                by_cam[s.camera_id].append(s.height_m)

            if len(id_samples) < self._config.min_samples:
                skipped[identity_id] = "insufficient_samples"
                continue
            if len(days) < self._config.min_days:
                skipped[identity_id] = "insufficient_days"
                continue
            if len(by_cam) < self._config.min_cameras:
                skipped[identity_id] = "insufficient_cameras"
                continue

            cam_medians = {cam: float(median(vals)) for cam, vals in by_cam.items()}
            spread_m = max(cam_medians.values()) - min(cam_medians.values())
            spreads[identity_id] = spread_m

            med_m, sigma_m = compute_robust_height_stats(
                [s.height_m for s in id_samples],
                min_sigma_m=self._config.min_sigma_m,
            )
            profile = IdentityHeightProfile(
                identity_id=identity_id,
                height_m=med_m,
                height_sigma_m=sigma_m,
                source="learned",
                samples=len(id_samples),
                cameras=len(by_cam),
                updated_at=as_of,
            )
            await self._repo.upsert(profile)
            updated.append(identity_id)

            logger.info(
                "height_profile_learned",
                samples=len(id_samples),
                days=len(days),
                cameras=len(by_cam),
                height_m=round(med_m, 4),
                height_sigma_m=round(sigma_m, 4),
                camera_spread_m=round(spread_m, 4),
            )

        if updated and self._on_updated is not None:
            self._on_updated()

        return HeightLearnResult(
            updated_identities=tuple(updated),
            skipped_identities=skipped,
            camera_spread_by_identity_m=spreads,
        )

    async def run(self) -> None:
        """Background hourly scheduler loop (mirrors PHMaintenanceService)."""
        if not self._config.enabled:
            logger.info("height_profile_learner_disabled")
            return
        logger.info(
            "height_profile_learner_started",
            interval_s=self._config.interval_s,
            min_samples=self._config.min_samples,
            min_days=self._config.min_days,
            min_cameras=self._config.min_cameras,
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
                logger.exception("height_profile_learner_tick_failed")
        logger.info("height_profile_learner_stopped")

    async def stop(self) -> None:
        self._stop_event.set()


def summarize_camera_medians(
    samples: Sequence[AttestedHeightSample],
) -> list[CameraHeightSummary]:
    """Summarize per-camera sample count and median height (for diagnostics)."""
    by_cam: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        by_cam[s.camera_id].append(s.height_m)
    return [
        CameraHeightSummary(
            camera_id=cam,
            samples=len(vals),
            median_height_m=float(median(vals)),
        )
        for cam, vals in sorted(by_cam.items())
    ]
