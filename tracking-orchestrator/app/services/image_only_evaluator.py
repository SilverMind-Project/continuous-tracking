"""Scores image-only identity against wearable truth and keeps daily totals.

The resolver samples attested PH resolutions and decides each one twice without the wearable
channel (``ImageOnlyEvaluation``: a control arm without wearable-taught gallery rows, a
treatment arm with them). This service turns those decisions into seconds and outcomes:
Prometheus counters for the live view, and per identity, UTC day, arm, and grade totals in
``image_only_evaluation_daily`` so the result survives restarts and supports day-split claims
("taught on Monday, helps on Tuesday").

Time is measured, not counted: an evaluation of a PH contributes the capture-time gap since the
previous sampled evaluation of the same PH, capped at ``max_gap_s``. With sampling rate r the
gaps average round-duration / r, so the total equals round duration times sample weight in
expectation, independent of how often rounds run. Both arms of one evaluation get the same
seconds, which keeps the comparison paired.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from structlog import get_logger

from app.domain import ImageOnlyDailyTotal, ImageOnlyEvaluation, WearableAttestationGrade
from app.observability import metrics as _metrics
from app.observability.metrics import ImageOnlyArm, ImageOnlyOutcome
from app.storage.image_only_evaluation import ImageOnlyEvaluationRepository

logger = get_logger(__name__)


class ImageOnlyEvaluationSink(Protocol):
    """What the pipeline stage needs: one call per tracker round."""

    async def record_round(
        self, evaluations: Sequence[ImageOnlyEvaluation], *, captured_at: datetime
    ) -> None: ...


@dataclass
class _Pending:
    correct_s: float = 0.0
    wrong_s: float = 0.0
    unknown_s: float = 0.0


def classify(decided: str | None, truth: str) -> ImageOnlyOutcome:
    """Outcome of one arm's decision against the tag's member (``None`` is Unknown)."""
    if decided is None:
        return ImageOnlyOutcome.UNKNOWN
    return ImageOnlyOutcome.CORRECT if decided == truth else ImageOnlyOutcome.WRONG


@dataclass(frozen=True)
class DayRates:
    """One day's rates for one arm; a rate is None when its denominator is zero."""

    attribution: float | None
    wrong: float | None
    seconds: float


def daily_rates(total: ImageOnlyDailyTotal | None) -> DayRates:
    """Compute attribution and wrong rates from one day/arm total."""
    if total is None:
        return DayRates(attribution=None, wrong=None, seconds=0.0)
    seconds = total.correct_s + total.wrong_s + total.unknown_s
    named = total.correct_s + total.wrong_s
    return DayRates(
        attribution=total.correct_s / seconds if seconds > 0 else None,
        wrong=total.wrong_s / named if named > 0 else None,
        seconds=seconds,
    )


def summarize_attribution_difference(
    totals: Sequence[ImageOnlyDailyTotal],
) -> tuple[int, float | None]:
    """Return ``(distinct_paired_days, mean_treatment_minus_control_attribution)``.

    Matches the paired daily calculation in ``scripts/image_only_evaluation_report.py``:
    each ``(identity_id, attestation_grade, day)`` where both ``control`` and ``treatment``
    arms have positive total seconds contributes ``treatment.attribution - control.attribution``.
    """
    by_key: dict[tuple[str, str], dict[date, dict[str, ImageOnlyDailyTotal]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for total in totals:
        by_key[(total.identity_id, str(total.attestation_grade))][total.day][total.arm] = total

    paired_days: set[date] = set()
    diffs: list[float] = []
    for days in by_key.values():
        for day, by_arm in days.items():
            control = daily_rates(by_arm.get("control"))
            treatment = daily_rates(by_arm.get("treatment"))
            if control.attribution is not None and treatment.attribution is not None:
                paired_days.add(day)
                diffs.append(treatment.attribution - control.attribution)

    if not diffs:
        return 0, None
    return len(paired_days), sum(diffs) / len(diffs)


class ImageOnlyEvaluationService:
    """Accumulates image-only seconds and flushes them to the daily repository."""

    def __init__(
        self,
        repo: ImageOnlyEvaluationRepository,
        *,
        max_gap_s: float,
        flush_interval_s: float = 60.0,
    ) -> None:
        self._repo = repo
        self._max_gap = timedelta(seconds=max_gap_s)
        self._flush_interval = timedelta(seconds=flush_interval_s)
        self._last_eval_at: dict[str, datetime] = {}
        self._pending: dict[tuple[date, str, str, WearableAttestationGrade], _Pending] = {}
        self._last_flush_at: datetime | None = None

    async def record_round(
        self, evaluations: Sequence[ImageOnlyEvaluation], *, captured_at: datetime
    ) -> None:
        """Account one tracker round; flushes once ``flush_interval_s`` of capture time passed."""
        for evaluation in evaluations:
            self._account(evaluation)
        self._prune(captured_at)
        if self._last_flush_at is None:
            self._last_flush_at = captured_at
        elif captured_at - self._last_flush_at >= self._flush_interval:
            await self.flush()
            self._last_flush_at = captured_at

    async def flush(self) -> None:
        """Write the pending totals. A failed write keeps them for the next flush."""
        if not self._pending:
            return
        totals = [
            ImageOnlyDailyTotal(
                day=day,
                identity_id=identity_id,
                arm=arm,
                attestation_grade=grade,
                correct_s=p.correct_s,
                wrong_s=p.wrong_s,
                unknown_s=p.unknown_s,
            )
            for (day, identity_id, arm, grade), p in self._pending.items()
        ]
        try:
            await self._repo.add_totals(totals)
        except Exception:
            logger.warning("image_only_evaluation_flush_failed", rows=len(totals), exc_info=True)
            return
        self._pending.clear()

    async def purge_identity(self, identity_id: str) -> int:
        """Drop pending and stored totals of a purged identity."""
        doomed = [key for key in self._pending if key[1] == identity_id]
        for key in doomed:
            del self._pending[key]
        return len(doomed) + await self._repo.purge_identity(identity_id)

    def _account(self, evaluation: ImageOnlyEvaluation) -> None:
        previous = self._last_eval_at.get(evaluation.ph_id)
        self._last_eval_at[evaluation.ph_id] = evaluation.captured_at
        if previous is None:
            return
        gap = evaluation.captured_at - previous
        if gap <= timedelta(0) or gap > self._max_gap:
            return
        seconds = gap.total_seconds()
        grade = evaluation.attestation_grade
        day = evaluation.captured_at.date()
        truth = evaluation.truth_identity_id
        for arm, decided in (
            (ImageOnlyArm.CONTROL, evaluation.control_identity_id),
            (ImageOnlyArm.TREATMENT, evaluation.treatment_identity_id),
        ):
            outcome = classify(decided, truth)
            _metrics.metrics.cts_image_only_eval_seconds_total.labels(
                arm=arm, outcome=outcome, grade=str(grade)
            ).inc(seconds)
            pending = self._pending.setdefault((day, truth, str(arm), grade), _Pending())
            if outcome == ImageOnlyOutcome.CORRECT:
                pending.correct_s += seconds
            elif outcome == ImageOnlyOutcome.WRONG:
                pending.wrong_s += seconds
            else:
                pending.unknown_s += seconds

    def _prune(self, now: datetime) -> None:
        stale = [ph for ph, at in self._last_eval_at.items() if now - at > self._max_gap]
        for ph in stale:
            del self._last_eval_at[ph]
