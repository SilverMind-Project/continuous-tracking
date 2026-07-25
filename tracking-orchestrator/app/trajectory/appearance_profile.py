"""Daily appearance profile: pure quality-weighted centroid math (DL-M07).

SOLIDER re-identification embeddings persisted on ``PersonHypothesis.gallery_mean``
are clothing-dominated appearance vectors. A quality-weighted daily centroid,
built from every qualifying PH for one identity on one local day, is robust to
PH churn and to a handful of low-quality crops. Day-over-day cosine similarity
of two such centroids is the ``same_clothes_suspected`` evaluator's core signal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from structlog import get_logger

from ..domain import DementiaSignal, DementiaSignalSeverity, PersonHypothesis
from .dementia_signals import SignalHysteresis, _stable_signal_id
from .signal_specs import apply_algorithm_metadata

if TYPE_CHECKING:
    from ..storage.appearance import DailyAppearanceRepo
    from ..storage.base import PHRepositoryProtocol
    from ..storage.gallery import GalleryRepository
    from ..storage.trajectory import KeyframeRepository

logger = get_logger(__name__)

# PHs with fewer observations than this contribute a gallery_mean averaged over
# too few frames to be a reliable appearance sample (mirrors the "transient PH"
# threshold used elsewhere for the same reason: a couple of frames of churn,
# not a settled appearance estimate).
MIN_OBSERVATIONS_FOR_CENTROID = 3

# Quality floor: PHs below this mean_quality are noisy crops (motion blur, bad
# angle, occlusion) that would drag the centroid toward garbage rather than
# toward "what she is wearing".
DEFAULT_QUALITY_FLOOR = 0.3


@dataclass(frozen=True)
class AppearanceEntry:
    """One PH's contribution to a daily centroid."""

    embedding: tuple[float, ...]
    quality: float
    observation_count: int


@dataclass(frozen=True)
class DailyAppearanceProfile:
    """Quality-weighted appearance centroid for one identity on one local day."""

    identity_id: str
    day: date
    centroid: tuple[float, ...]
    sample_count: int
    mean_quality: float
    best_keyframe_objects: tuple[str, ...]
    created_at: datetime


def daily_centroid(
    entries: Sequence[AppearanceEntry],
    *,
    quality_floor: float = DEFAULT_QUALITY_FLOOR,
    min_observations: int = MIN_OBSERVATIONS_FOR_CENTROID,
) -> tuple[tuple[float, ...], int, float] | None:
    """Compute a quality-weighted, L2-normalised centroid from PH appearance entries.

    Filters out PHs below ``quality_floor`` or ``min_observations``, then
    averages the remaining embeddings weighted by ``quality`` and normalises the
    result to unit length. Returns ``(centroid, sample_count, mean_quality)``
    over the qualifying entries, or ``None`` when none qualify (fail-silent:
    insufficient evidence, not a zero-vector centroid).
    """
    qualifying = [
        e for e in entries if e.quality >= quality_floor and e.observation_count >= min_observations
    ]
    if not qualifying:
        return None

    dim = len(qualifying[0].embedding)
    weighted_sum = [0.0] * dim
    weight_total = 0.0
    for entry in qualifying:
        if len(entry.embedding) != dim:
            raise ValueError("all embeddings must share the same dimensionality")
        for i, value in enumerate(entry.embedding):
            weighted_sum[i] += value * entry.quality
        weight_total += entry.quality

    if weight_total <= 0:
        return None

    centroid = [value / weight_total for value in weighted_sum]
    norm = math.sqrt(sum(value * value for value in centroid))
    if norm > 0:
        centroid = [value / norm for value in centroid]

    mean_quality = sum(e.quality for e in qualifying) / len(qualifying)
    return tuple(centroid), len(qualifying), mean_quality


def compare(centroid_a: Sequence[float], centroid_b: Sequence[float]) -> float:
    """Cosine similarity between two centroids.

    Recomputes the norm rather than assuming unit-normalised input, so a
    malformed or all-zero centroid degrades to a similarity of ``0.0`` instead
    of raising or dividing by zero.
    """
    if len(centroid_a) != len(centroid_b):
        raise ValueError("centroid dimension mismatch")
    dot = sum(a * b for a, b in zip(centroid_a, centroid_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in centroid_a))
    norm_b = math.sqrt(sum(b * b for b in centroid_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _entries_from_phs(phs: Sequence[PersonHypothesis]) -> list[AppearanceEntry]:
    """Adapt PH rows to appearance entries, dropping PHs with no embedding."""
    return [
        AppearanceEntry(
            embedding=tuple(ph.gallery_mean),
            quality=ph.mean_quality,
            observation_count=ph.observation_count,
        )
        for ph in phs
        if ph.gallery_mean
    ]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

# Best-evidence keyframes kept per day: enough for CC to pick a usable crop
# across cameras without a second CTS query protocol, without turning
# context_json into an unbounded frame dump.
_MAX_BEST_KEYFRAMES = 3

# Restart-safety cooldown for the once-per-day emission: shorter than 24h so
# it never blocks the next day's legitimate evaluation, long enough that a
# process restart within the same evaluation day cannot double-alert.
_RESTART_SAFETY_COOLDOWN_MINUTES = 20 * 60


@dataclass(frozen=True)
class AppearanceSettings:
    """Configuration for the same_clothes_suspected evaluator."""

    enabled: bool = False
    similarity_threshold: float = 0.90
    min_samples_per_day: int = 5
    evaluate_local_hour: int = 11
    # Shared with SignalConfig.tz_name (app.timezone): "yesterday's clothes" is
    # a local-calendar concept, and CC's day-boundary logic must agree.
    tz_name: str = "America/New_York"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.tz_name)


class AppearanceEvaluator:
    """Computes the daily same_clothes_suspected signal from PH appearance data.

    Driven from the same scheduler loop as :class:`DementiaSignalWorker` and
    :class:`GaitAggregator` (mirrors the gait integration style: no additional
    asyncio task). Evaluates each identity at most once per local day, at or
    after ``evaluate_local_hour``; the per-identity "already evaluated today"
    check is an in-memory dict lookup, so calling ``run_once`` every worker
    tick before that hour costs nothing and after it costs one lookup per
    already-done identity (cheap enough that a separate due()-style gate,
    unlike GaitAggregator's hourly full-table scan, would not add anything).
    """

    def __init__(
        self,
        ph_repo: PHRepositoryProtocol,
        profile_repo: DailyAppearanceRepo,
        gallery_repo: GalleryRepository,
        keyframe_repo: KeyframeRepository | None = None,
        cfg: AppearanceSettings | None = None,
        hysteresis: SignalHysteresis | None = None,
    ) -> None:
        self._ph_repo = ph_repo
        self._profile_repo = profile_repo
        self._gallery_repo = gallery_repo
        self._keyframe_repo = keyframe_repo
        self._cfg = cfg or AppearanceSettings()
        # Same-clothes fires at most once per identity per local day. The
        # shared hysteresis default (min_consecutive=2, one onset debounce
        # cycle per *run*) would cost two calendar days at day-granularity,
        # delaying a same-day caregiver-actionable signal to tomorrow -- this
        # defeats decision 3's "the caregiver can still act today". This
        # evaluator therefore owns its own hysteresis with min_consecutive=1;
        # cooldown still applies so a restart mid-day cannot double-alert.
        self._hysteresis = hysteresis or SignalHysteresis(min_consecutive=1)
        self._last_evaluated_day: dict[str, date] = {}

    async def run_once(self, now: datetime) -> list[DementiaSignal]:
        """Evaluate every enrolled identity once, if past evaluate_local_hour today.

        Returns the list of emitted (or shadow-mode would-be-emitted) signals;
        in shadow mode (``cfg.enabled is False``) the list is always empty --
        candidates are logged and metered, never returned for publishing.
        """
        local_now = now.astimezone(self._cfg.tz)
        if local_now.hour < self._cfg.evaluate_local_hour:
            return []

        today = local_now.date()
        yesterday = today - timedelta(days=1)

        identities = await self._gallery_repo.list_identities(active_only=True)
        signals: list[DementiaSignal] = []
        for identity in identities:
            identity_id = identity.identity_id
            if self._last_evaluated_day.get(identity_id) == today:
                continue
            self._last_evaluated_day[identity_id] = today
            signal = await self._evaluate_identity(identity_id, today, yesterday, now)
            if signal is not None:
                signals.append(signal)
        return signals

    async def _evaluate_identity(
        self,
        identity_id: str,
        today: date,
        yesterday: date,
        now: datetime,
    ) -> DementiaSignal | None:
        yesterday_profile = await self._get_or_backfill_day(identity_id, yesterday)
        if (
            yesterday_profile is None
            or yesterday_profile.sample_count < self._cfg.min_samples_per_day
        ):
            return None

        today_result = await self._compute_day(identity_id, today, persist=False)
        if today_result is None:
            return None
        today_profile, _today_phs = today_result
        if today_profile.sample_count < self._cfg.min_samples_per_day:
            return None

        similarity = compare(yesterday_profile.centroid, today_profile.centroid)

        # Shadow-mode observability: always log and meter the candidate value,
        # even when disabled or below threshold, so the live distribution is
        # visible before any flip (DL10). This must run before the threshold
        # gate below or the backtest would only ever see already-filtered data.
        logger.info(
            "appearance_signal_candidate",
            identity_id=identity_id,
            similarity=round(similarity, 4),
            enabled=self._cfg.enabled,
            threshold=self._cfg.similarity_threshold,
            yesterday_day=str(yesterday),
            yesterday_sample_count=yesterday_profile.sample_count,
            today_sample_count=today_profile.sample_count,
        )
        try:
            from ..observability import metrics as m

            m.metrics.appearance_signal_candidates_total.inc()
        except Exception:  # noqa: BLE001  # metrics are non-required side-channel
            logger.warning("appearance_signal_candidate_metric_failed", exc_info=True)

        if not self._cfg.enabled:
            return None
        if similarity < self._cfg.similarity_threshold:
            return None

        # Experimental evidence grade (no clinical validation yet): severity
        # never escalates past info, mirroring the severity policy's cap for
        # experimental/trend kinds.
        severity: DementiaSignalSeverity = "info"
        if not self._hysteresis.should_emit(
            identity_id,
            "same_clothes_suspected",
            severity,
            now,
            _RESTART_SAFETY_COOLDOWN_MINUTES,
        ):
            return None

        window_start = datetime.combine(yesterday, time.min, tzinfo=self._cfg.tz).astimezone(UTC)
        signal = DementiaSignal(
            signal_id=_stable_signal_id(
                identity_id, "same_clothes_suspected", window_start, today.isoformat()
            ),
            identity_id=identity_id,
            signal_kind="same_clothes_suspected",
            severity=severity,
            value=round(similarity, 4),
            window_start=window_start,
            window_end=now,
            emitted_at=now,
            algorithm_version=1,
            context={
                "similarity": round(similarity, 4),
                "yesterday_day": str(yesterday),
                "today_day": str(today),
                "yesterday_sample_count": yesterday_profile.sample_count,
                "yesterday_mean_quality": round(yesterday_profile.mean_quality, 4),
                "today_sample_count": today_profile.sample_count,
                "today_mean_quality": round(today_profile.mean_quality, 4),
                "yesterday_best_keyframe_objects": list(yesterday_profile.best_keyframe_objects),
                "today_best_keyframe_objects": list(today_profile.best_keyframe_objects),
            },
        )
        return apply_algorithm_metadata(signal, "same_clothes_suspected")

    async def _get_or_backfill_day(
        self, identity_id: str, day: date
    ) -> DailyAppearanceProfile | None:
        """Return the persisted profile for ``day``, computing and storing it if absent."""
        existing = await self._profile_repo.get_profile(identity_id, day)
        if existing is not None:
            return existing
        result = await self._compute_day(identity_id, day, persist=True)
        return result[0] if result is not None else None

    async def _compute_day(
        self, identity_id: str, day: date, *, persist: bool
    ) -> tuple[DailyAppearanceProfile, list[PersonHypothesis]] | None:
        """Compute (and optionally persist) one identity's centroid for one local day."""
        day_start = datetime.combine(day, time.min, tzinfo=self._cfg.tz).astimezone(UTC)
        day_end = datetime.combine(day, time.max, tzinfo=self._cfg.tz).astimezone(UTC)

        phs = await self._ph_repo.list_overlapping_for_identity(identity_id, day_start, day_end)
        entries = _entries_from_phs(phs)
        result = daily_centroid(entries)
        if result is None:
            return None
        centroid, sample_count, mean_quality = result

        best_objects = await self._best_keyframe_objects(phs, day_start, day_end)
        profile = DailyAppearanceProfile(
            identity_id=identity_id,
            day=day,
            centroid=centroid,
            sample_count=sample_count,
            mean_quality=mean_quality,
            best_keyframe_objects=tuple(best_objects),
            created_at=datetime.now(UTC),
        )
        if persist:
            await self._profile_repo.upsert_profile(profile)
        return profile, phs

    async def _best_keyframe_objects(
        self, phs: Sequence[PersonHypothesis], day_start: datetime, day_end: datetime
    ) -> list[str]:
        """Return up to _MAX_BEST_KEYFRAMES minio_keys, highest detection confidence first."""
        if self._keyframe_repo is None:
            return []
        candidates: list[tuple[float, str]] = []
        for ph in phs:
            keyframes = await self._keyframe_repo.list_keyframes(
                ph_id=ph.ph_id, after=day_start, limit=50
            )
            for kf in keyframes:
                if kf.captured_at > day_end:
                    continue
                confidence = float(kf.annotations.get("confidence", 0.0) or 0.0)
                candidates.append((confidence, kf.minio_key))
        candidates.sort(key=lambda c: c[0], reverse=True)
        seen: set[str] = set()
        best: list[str] = []
        for _confidence, minio_key in candidates:
            if minio_key in seen:
                continue
            seen.add(minio_key)
            best.append(minio_key)
            if len(best) >= _MAX_BEST_KEYFRAMES:
                break
        return best
