#!/usr/bin/env python3
"""Backtest the same_clothes_suspected evaluator against historical PH data (DL-M07).

Computes, for every identity with `current_identity_id` set, a daily
quality-weighted appearance centroid for each of the last N days and the
day-over-day cosine similarity between consecutive days. Outputs a markdown
report: a per-consecutive-day similarity table, a distribution summary
(min/median/p90), the days a caregiver-supplied annotation file marks as
same-clothes days (if provided), and the candidate threshold's
would-have-fired dates.

This must be run and its report committed (DL-M07 Part D) before
`hygiene.same_clothes.enabled` is ever flipped to true (DL10: shadow before
authority).

Required env var: DATABASE_URL (postgresql+asyncpg://... or postgresql://...)

Usage::

    DATABASE_URL=postgresql://cts_user:...@localhost:5432/continuous_tracking \\
        python scripts/backtest_appearance.py --days 21 --output daily-living-m07-backtest-report.md
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import statistics
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import asyncpg

from app.trajectory.appearance_profile import (
    AppearanceEntry,
    DailyAppearanceProfile,
    compare,
    daily_centroid,
)

DEFAULT_TZ_NAME = "America/New_York"
DEFAULT_LOOKBACK_DAYS = 21
DEFAULT_SIMILARITY_THRESHOLD = 0.90
DEFAULT_MIN_SAMPLES_PER_DAY = 5


def _dsn_from_env() -> str:
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return raw.replace("postgresql+asyncpg://", "postgresql://")


@dataclass(frozen=True)
class _PHRow:
    identity_id: str
    born_at: datetime
    closed_at: datetime | None
    gallery_mean: list[float] | None
    mean_quality: float
    observation_count: int


async def _fetch_ph_rows(
    conn: asyncpg.Connection, since: datetime, until: datetime
) -> list[_PHRow]:
    rows = await conn.fetch(
        """
        SELECT current_identity_id, born_at, closed_at, gallery_mean,
               mean_quality, observation_count
        FROM continuous_tracking.person_hypotheses
        WHERE current_identity_id IS NOT NULL
          AND born_at <= $2
          AND (closed_at IS NULL OR closed_at >= $1)
        ORDER BY current_identity_id, born_at
        """,
        since,
        until,
    )
    return [
        _PHRow(
            identity_id=r["current_identity_id"],
            born_at=r["born_at"],
            closed_at=r["closed_at"],
            gallery_mean=list(r["gallery_mean"]) if r["gallery_mean"] else None,
            mean_quality=float(r["mean_quality"]),
            observation_count=int(r["observation_count"]),
        )
        for r in rows
    ]


def _day_bounds(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=tz).astimezone(UTC)
    end = datetime.combine(day, time.max, tzinfo=tz).astimezone(UTC)
    return start, end


def _profile_for_day(
    identity_id: str, day: date, rows: list[_PHRow], tz: ZoneInfo
) -> DailyAppearanceProfile | None:
    day_start, day_end = _day_bounds(day, tz)
    overlapping = [
        r
        for r in rows
        if r.born_at <= day_end and (r.closed_at is None or r.closed_at >= day_start)
    ]
    entries = [
        AppearanceEntry(
            embedding=tuple(r.gallery_mean),
            quality=r.mean_quality,
            observation_count=r.observation_count,
        )
        for r in overlapping
        if r.gallery_mean
    ]
    result = daily_centroid(entries)
    if result is None:
        return None
    centroid, sample_count, mean_quality = result
    return DailyAppearanceProfile(
        identity_id=identity_id,
        day=day,
        centroid=centroid,
        sample_count=sample_count,
        mean_quality=mean_quality,
        best_keyframe_objects=(),
        created_at=datetime.now(UTC),
    )


@dataclass(frozen=True)
class _DayPairResult:
    identity_id: str
    day_a: date
    day_b: date
    similarity: float
    sample_count_a: int
    sample_count_b: int


def _compute_day_pairs(
    identity_id: str,
    rows: list[_PHRow],
    days: list[date],
    tz: ZoneInfo,
    min_samples_per_day: int,
) -> list[_DayPairResult]:
    profiles: dict[date, DailyAppearanceProfile] = {}
    for day in days:
        profile = _profile_for_day(identity_id, day, rows, tz)
        if profile is not None and profile.sample_count >= min_samples_per_day:
            profiles[day] = profile

    results: list[_DayPairResult] = []
    sorted_days = sorted(profiles.keys())
    for prev_day, curr_day in itertools.pairwise(sorted_days):
        if (curr_day - prev_day).days != 1:
            continue  # only true consecutive-day pairs
        prev_profile = profiles[prev_day]
        curr_profile = profiles[curr_day]
        similarity = compare(prev_profile.centroid, curr_profile.centroid)
        results.append(
            _DayPairResult(
                identity_id=identity_id,
                day_a=prev_day,
                day_b=curr_day,
                similarity=similarity,
                sample_count_a=prev_profile.sample_count,
                sample_count_b=curr_profile.sample_count,
            )
        )
    return results


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _render_report(
    *,
    lookback_days: int,
    since: datetime,
    until: datetime,
    identity_count: int,
    ph_row_count: int,
    pairs: list[_DayPairResult],
    threshold: float,
    min_samples_per_day: int,
    annotations: dict[str, list[str]] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Daily Living M07 Backtest Report: same_clothes_suspected")
    lines.append("")
    lines.append(
        f"Generated {datetime.now(UTC).isoformat()} over the last {lookback_days} days "
        f"({since.date()} to {until.date()})."
    )
    lines.append("")
    lines.append(
        f"Identities with `current_identity_id` set in the window: {identity_count}. "
        f"Total qualifying PH rows: {ph_row_count}."
    )
    lines.append("")

    if not pairs:
        lines.append("## Result: insufficient data")
        lines.append("")
        lines.append(
            f"No identity in the lookback window had two consecutive local days each "
            f"with at least `min_samples_per_day={min_samples_per_day}` quality-weighted "
            f"PH samples. No day-over-day similarity could be computed."
        )
        lines.append("")
        lines.append(
            f"**Decision**: `hygiene.same_clothes.similarity_threshold` stays at its "
            f"documented pre-backtest default of {threshold:.2f}. The live flip "
            f"(`hygiene.same_clothes.enabled: true`) is blocked pending accumulation of "
            f"enough production data (repeated identity attribution across consecutive "
            f"days) to run a meaningful backtest. Shadow-mode logging "
            f"(`appearance_signal_candidate`) and the "
            f"`cts_appearance_signal_candidates_total` metric remain the path to observe "
            f"the live similarity distribution as data accumulates; re-run this script "
            f"once identity attribution is denser."
        )
        lines.append("")
        return "\n".join(lines)

    similarities = [p.similarity for p in pairs]
    lines.append("## Distribution summary")
    lines.append("")
    lines.append(f"- Consecutive-day pairs: {len(pairs)}")
    lines.append(f"- Min: {min(similarities):.4f}")
    lines.append(f"- Median: {statistics.median(similarities):.4f}")
    lines.append(f"- P90: {_percentile(similarities, 0.90):.4f}")
    lines.append(f"- Max: {max(similarities):.4f}")
    lines.append("")

    lines.append("## Per-consecutive-day similarity table")
    lines.append("")
    lines.append("| identity_id | day A | day B | similarity | samples A | samples B |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for p in pairs:
        lines.append(
            f"| {p.identity_id} | {p.day_a} | {p.day_b} | {p.similarity:.4f} "
            f"| {p.sample_count_a} | {p.sample_count_b} |"
        )
    lines.append("")

    would_fire = [p for p in pairs if p.similarity >= threshold]
    lines.append(f"## Would-have-fired dates at threshold {threshold:.2f}")
    lines.append("")
    if would_fire:
        for p in would_fire:
            annotated = ""
            if annotations is not None:
                known_days = set(annotations.get(p.identity_id, []))
                annotated = " (annotated same-clothes day)" if str(p.day_b) in known_days else ""
            lines.append(
                f"- {p.identity_id}: {p.day_a} -> {p.day_b} ({p.similarity:.4f}){annotated}"
            )
    else:
        lines.append("None.")
    lines.append("")

    if annotations is not None:
        lines.append("## Annotation cross-reference")
        lines.append("")
        total_known = sum(len(v) for v in annotations.values())
        caught = sum(
            1
            for p in pairs
            if str(p.day_b) in set(annotations.get(p.identity_id, [])) and p.similarity >= threshold
        )
        lines.append(f"- Annotated same-clothes days: {total_known}")
        lines.append(f"- Caught at threshold {threshold:.2f}: {caught}")
        lines.append("")

    lines.append(
        f"**Decision**: recorded threshold `hygiene.same_clothes.similarity_threshold = "
        f"{threshold:.2f}` per this report's distribution."
    )
    lines.append("")
    return "\n".join(lines)


async def run_backtest(
    *,
    days: int,
    threshold: float,
    min_samples_per_day: int,
    tz_name: str,
    annotations_path: Path | None,
) -> str:
    dsn = _dsn_from_env()
    tz = ZoneInfo(tz_name)
    now = datetime.now(UTC)
    since = now - timedelta(days=days)

    day_list = [(now.astimezone(tz).date() - timedelta(days=i)) for i in range(days, -1, -1)]

    annotations: dict[str, list[str]] | None = None
    if annotations_path is not None:
        annotations = json.loads(annotations_path.read_text())

    conn = await asyncpg.connect(dsn)
    try:
        rows = await _fetch_ph_rows(conn, since, now)
    finally:
        await conn.close()

    by_identity: dict[str, list[_PHRow]] = {}
    for row in rows:
        by_identity.setdefault(row.identity_id, []).append(row)

    all_pairs: list[_DayPairResult] = []
    for identity_id, identity_rows in by_identity.items():
        all_pairs.extend(
            _compute_day_pairs(identity_id, identity_rows, day_list, tz, min_samples_per_day)
        )

    return _render_report(
        lookback_days=days,
        since=since,
        until=now,
        identity_count=len(by_identity),
        ph_row_count=len(rows),
        pairs=all_pairs,
        threshold=threshold,
        min_samples_per_day=min_samples_per_day,
        annotations=annotations,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--min-samples-per-day", type=int, default=DEFAULT_MIN_SAMPLES_PER_DAY)
    parser.add_argument("--tz-name", default=DEFAULT_TZ_NAME)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = asyncio.run(
        run_backtest(
            days=args.days,
            threshold=args.threshold,
            min_samples_per_day=args.min_samples_per_day,
            tz_name=args.tz_name,
            annotations_path=args.annotations,
        )
    )
    if args.output is not None:
        args.output.write_text(report)
        print(f"Wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
