#!/usr/bin/env python3
"""Image-only identity scored against wearable truth, per identity and day.

For each identity and attestation grade, per UTC day and per arm (control: no wearable-taught
gallery rows; treatment: with them), it reports the attribution rate
``correct / (correct + wrong + unknown)`` and the wrong rate ``wrong / (correct + wrong)``, then
the treatment-minus-control difference of each, averaged over days, with a bootstrap 95%
interval that resamples days. Both arms score the same rounds, so each day's difference is a
paired comparison. The claim "wearable teaching helps the cameras" holds for an identity when the
attribution difference's interval lies above zero and the wrong-rate difference is not positive.

Read-only: the connection sets ``default_transaction_read_only``.

Usage:
    DATABASE_URL=postgresql://... python scripts/image_only_evaluation_report.py --since 2026-12-01
    ... --until 2026-12-14 --grade trajectory --format json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import date

import asyncpg
import numpy as np

from app.domain import ImageOnlyDailyTotal
from app.services.image_only_evaluator import DayRates, daily_rates
from app.storage.postgres.image_only_evaluation_repo import PostgresImageOnlyEvaluationRepository

_ARMS = ("control", "treatment")


def _rates(total: ImageOnlyDailyTotal | None) -> DayRates:
    return daily_rates(total)


def bootstrap_mean_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float, float] | None:
    """Mean of *values* and a percentile 95% interval from resampling them with replacement."""
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(samples, data.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(data.mean()), float(low), float(high)


def build_report(
    totals: Sequence[ImageOnlyDailyTotal], *, samples: int = 2000, seed: int = 0
) -> list[dict[str, object]]:
    """Per (identity, grade): daily rates per arm and the paired treatment-minus-control effect."""
    by_key: dict[tuple[str, str], dict[date, dict[str, ImageOnlyDailyTotal]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for total in totals:
        by_key[(total.identity_id, str(total.attestation_grade))][total.day][total.arm] = total

    report: list[dict[str, object]] = []
    for (identity_id, grade), days in sorted(by_key.items()):
        daily: list[dict[str, object]] = []
        attribution_diffs: list[float] = []
        wrong_diffs: list[float] = []
        for day in sorted(days):
            rates = {arm: _rates(days[day].get(arm)) for arm in _ARMS}
            control, treatment = rates["control"], rates["treatment"]
            if control.attribution is not None and treatment.attribution is not None:
                attribution_diffs.append(treatment.attribution - control.attribution)
            if control.wrong is not None and treatment.wrong is not None:
                wrong_diffs.append(treatment.wrong - control.wrong)
            daily.append(
                {
                    "day": day.isoformat(),
                    **{
                        f"{arm}_{name}": getattr(rates[arm], name)
                        for arm in _ARMS
                        for name in ("attribution", "wrong", "seconds")
                    },
                }
            )
        report.append(
            {
                "identity_id": identity_id,
                "attestation_grade": grade,
                "days": daily,
                "attribution_difference": bootstrap_mean_interval(
                    attribution_diffs, samples=samples, seed=seed
                ),
                "wrong_rate_difference": bootstrap_mean_interval(
                    wrong_diffs, samples=samples, seed=seed
                ),
            }
        )
    return report


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, tuple):
        mean, low, high = value
        return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"
    return str(value)


def render_table(report: Sequence[dict[str, object]]) -> str:
    lines: list[str] = []
    for entry in report:
        lines.append(f"identity {entry['identity_id']} ({entry['attestation_grade']} grade)")
        lines.append("  day         control attr  treatment attr  control wrong  treatment wrong")
        days = entry["days"]
        assert isinstance(days, list)
        for day in days:
            lines.append(
                f"  {day['day']}  {_fmt(day['control_attribution']):>12}  "
                f"{_fmt(day['treatment_attribution']):>14}  {_fmt(day['control_wrong']):>13}  "
                f"{_fmt(day['treatment_wrong']):>15}"
            )
        lines.append(f"  attribution difference: {_fmt(entry['attribution_difference'])}")
        lines.append(f"  wrong-rate difference:  {_fmt(entry['wrong_rate_difference'])}")
    return "\n".join(lines) if lines else "no image-only evaluations recorded"


def _dsn_from_env() -> str:
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise SystemExit("DATABASE_URL environment variable is not set")
    return raw.replace("postgresql+asyncpg://", "postgresql://")


async def _main(args: argparse.Namespace) -> int:
    pool = await asyncpg.create_pool(
        _dsn_from_env(),
        min_size=1,
        max_size=1,
        server_settings={"default_transaction_read_only": "on"},
    )
    try:
        totals = await PostgresImageOnlyEvaluationRepository(pool).list_totals(
            since=date.fromisoformat(args.since) if args.since else None,
            until=date.fromisoformat(args.until) if args.until else None,
        )
    finally:
        await pool.close()
    if args.grade:
        totals = [t for t in totals if str(t.attestation_grade) == args.grade]
    report = build_report(totals, samples=args.bootstrap_samples, seed=args.seed)
    output = json.dumps(report, indent=2) if args.format == "json" else render_table(report)
    sys.stdout.write(output + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--since", default=None, help="First UTC day (YYYY-MM-DD)")
    parser.add_argument("--until", default=None, help="Last UTC day (YYYY-MM-DD)")
    parser.add_argument("--grade", choices=["room", "trajectory"], default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
