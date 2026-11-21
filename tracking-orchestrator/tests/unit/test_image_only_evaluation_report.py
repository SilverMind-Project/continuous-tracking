"""scripts/image_only_evaluation_report.py: paired rates per day and the bootstrap effect."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from app.domain import ImageOnlyDailyTotal, WearableAttestationGrade

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import image_only_evaluation_report as report_script  # noqa: E402

_DAY = date(2026, 12, 1)


def _total(
    arm: str, day: date, correct: float, wrong: float, unknown: float
) -> ImageOnlyDailyTotal:
    return ImageOnlyDailyTotal(
        day=day,
        identity_id="alice",
        arm=arm,
        attestation_grade=WearableAttestationGrade.TRAJECTORY,
        correct_s=correct,
        wrong_s=wrong,
        unknown_s=unknown,
    )


def test_rates_and_a_positive_paired_effect() -> None:
    totals = []
    for i in range(5):
        day = _DAY + timedelta(days=i)
        totals.append(_total("control", day, correct=20, wrong=0, unknown=80))
        totals.append(_total("treatment", day, correct=60 + i, wrong=0, unknown=40 - i))
    (entry,) = report_script.build_report(totals, samples=500, seed=1)
    first = entry["days"][0]
    assert first["control_attribution"] == pytest.approx(0.2)
    assert first["treatment_attribution"] == pytest.approx(0.6)
    assert first["control_wrong"] == pytest.approx(0.0)
    mean, low, high = entry["attribution_difference"]
    assert mean == pytest.approx(0.42)
    assert 0.0 < low <= mean <= high
    assert entry["wrong_rate_difference"] == (0.0, 0.0, 0.0)


def test_rates_are_none_without_time_and_days_without_both_arms_are_skipped() -> None:
    totals = [
        _total("control", _DAY, correct=0, wrong=0, unknown=10),
        _total("treatment", _DAY + timedelta(days=1), correct=5, wrong=5, unknown=0),
    ]
    (entry,) = report_script.build_report(totals, samples=100, seed=0)
    day0, day1 = entry["days"]
    assert day0["control_wrong"] is None  # nothing named, so no wrong rate
    assert day0["treatment_attribution"] is None
    assert day1["treatment_wrong"] == pytest.approx(0.5)
    assert entry["attribution_difference"] is None


def test_table_renders() -> None:
    totals = [_total(arm, _DAY, 1, 1, 1) for arm in ("control", "treatment")]
    text = report_script.render_table(report_script.build_report(totals, samples=50))
    assert "identity alice (trajectory grade)" in text
    assert report_script.render_table([]) == "no image-only evaluations recorded"
