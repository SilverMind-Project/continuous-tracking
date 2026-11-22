"""Unit tests for offline UWB bench evaluation, battery projection, and necessity gate."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cts_contracts.wearables import AdapterKind, FixKind

from rf_ingress.bench import (
    BenchAccuracyStats,
    BenchGroundTruthPoint,
    compare_adapter_bench_parity,
    evaluate_bench_fixes,
    evaluate_necessity_gate,
    main,
    project_battery_runtime,
)
from rf_ingress.domain import EmittedFix

_T0 = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)


def _pos_fix(x_m: float, y_m: float, offset_s: float, seq: int) -> EmittedFix:
    return EmittedFix(
        tag_id="tag-1",
        measured_at=_T0 + timedelta(seconds=offset_s),
        kind=FixKind.POSITION,
        epoch_seq=seq,
        adapter_id="uwb-home",
        adapter_kind=AdapterKind.ANCHOR_RANGING,
        adapter_version="0.1.0",
        floor_plan_id="fp-1",
        x_mm=round(x_m * 1000.0),
        y_mm=round(y_m * 1000.0),
        cov_xx_m2=0.04,
        cov_xy_m2=0.0,
        cov_yy_m2=0.04,
        residual_rms_m=0.09,
    )


def test_evaluate_bench_fixes_passes_and_fails_gates() -> None:
    gt = [
        BenchGroundTruthPoint(
            point_id="P01",
            x_m=2.0,
            y_m=3.0,
            z_m=1.0,
            start_at=_T0,
            end_at=_T0 + timedelta(seconds=10),
        )
    ]
    good_fixes = [_pos_fix(2.0 + 0.02 * i, 3.0, float(i), i + 1) for i in range(9)]
    stats = evaluate_bench_fixes(good_fixes, gt, expected_epochs=10)
    assert stats.passed is True
    assert stats.availability_fraction == pytest.approx(0.9)
    assert stats.median_error_m is not None and stats.median_error_m <= 0.30
    assert stats.p95_error_m is not None and stats.p95_error_m <= 1.00

    poor_fixes = [_pos_fix(2.8, 3.0, float(i), i + 1) for i in range(5)]
    poor = evaluate_bench_fixes(poor_fixes, gt, expected_epochs=10)
    assert poor.passed is False
    assert poor.passed_median is False
    assert poor.passed_availability is False

    empty = evaluate_bench_fixes([], gt, expected_epochs=10)
    assert empty.passed is False
    assert empty.median_error_m is None


def test_compare_adapter_bench_parity_enforces_10pct_tolerance() -> None:
    direct = BenchAccuracyStats(
        expected_epochs=100,
        solved_position_fixes=95,
        evaluated_position_fixes=95,
        availability_fraction=0.95,
        median_error_m=0.18,
        p95_error_m=0.42,
        mean_error_m=0.20,
        residual_rms_median_m=0.08,
        passed_median=True,
        passed_p95=True,
        passed_availability=True,
    )
    close = BenchAccuracyStats(
        expected_epochs=100,
        solved_position_fixes=94,
        evaluated_position_fixes=94,
        availability_fraction=0.94,
        median_error_m=0.19,
        p95_error_m=0.44,
        mean_error_m=0.21,
        residual_rms_median_m=0.08,
        passed_median=True,
        passed_p95=True,
        passed_availability=True,
    )
    drifted = BenchAccuracyStats(
        expected_epochs=100,
        solved_position_fixes=94,
        evaluated_position_fixes=94,
        availability_fraction=0.94,
        median_error_m=0.25,
        p95_error_m=0.44,
        mean_error_m=0.26,
        residual_rms_median_m=0.08,
        passed_median=True,
        passed_p95=True,
        passed_availability=True,
    )
    assert compare_adapter_bench_parity(direct, close).within_tolerance is True
    assert compare_adapter_bench_parity(direct, drifted).within_tolerance is False


def test_project_battery_runtime_36h_and_72h_gates() -> None:
    # Stock always-on ESP32-S3 (~90 mA on 500 mAh cell) fails decisively.
    stock = project_battery_runtime(cell_capacity_mah=500.0, avg_current_ma=90.0)
    assert stock.passes_wearable_gate_36h is False
    assert stock.passes_senior_target_72h is False
    assert stock.projected_hours < 6.0

    # Duty-cycled 10 mA on 500 mAh * 0.85 = 42.5 h -> passes 36 h, below 72 h senior target.
    general = project_battery_runtime(cell_capacity_mah=500.0, avg_current_ma=10.0)
    assert general.projected_hours == pytest.approx(42.5)
    assert general.passes_wearable_gate_36h is True
    assert general.passes_senior_target_72h is False

    # Deep-sleep 4.5 mA on 500 mAh * 0.85 = 94.44 h -> passes both 36 h and 72 h.
    senior = project_battery_runtime(cell_capacity_mah=500.0, avg_current_ma=4.5)
    assert senior.passes_wearable_gate_36h is True
    assert senior.passes_senior_target_72h is True

    with pytest.raises(ValueError, match="avg_current_ma"):
        project_battery_runtime(cell_capacity_mah=500.0, avg_current_ma=0.0)


def test_evaluate_necessity_gate_all_outcomes() -> None:
    empty = evaluate_necessity_gate(
        worn_in_coverage_s=0.0,
        accepted_s=0.0,
        room_only_shared_room_s_by_room={},
        calibrated_rooms=[],
    )
    assert empty.gate_met is False
    assert empty.reason == "no_worn_coverage_data"

    below = evaluate_necessity_gate(
        worn_in_coverage_s=1000.0,
        accepted_s=750.0,
        room_only_shared_room_s_by_room={"living-room": 180.0},
        calibrated_rooms=["living-room"],
    )
    assert below.gate_met is False
    assert below.reason == "shared_room_share_below_threshold"
    assert below.share_b == pytest.approx(0.18)

    uncal = evaluate_necessity_gate(
        worn_in_coverage_s=1000.0,
        accepted_s=550.0,
        room_only_shared_room_s_by_room={"living-room": 300.0, "kitchen": 100.0},
        calibrated_rooms=["living-room"],
    )
    assert uncal.gate_met is False
    assert uncal.reason == "responsible_rooms_uncalibrated"
    assert uncal.uncalibrated_responsible_rooms == ("kitchen",)

    passed = evaluate_necessity_gate(
        worn_in_coverage_s=1000.0,
        accepted_s=550.0,
        room_only_shared_room_s_by_room={"living-room": 300.0, "kitchen": 100.0},
        calibrated_rooms=["living-room", "kitchen"],
    )
    assert passed.gate_met is True
    assert passed.reason == "necessity_gate_met"
    assert passed.share_a == pytest.approx(0.55)
    assert passed.share_b == pytest.approx(0.40)


def test_bench_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["battery", "--capacity-mah", "500", "--avg-current-ma", "5.0"]) == 0
    out_pass = json.loads(capsys.readouterr().out)
    assert out_pass["passes_wearable_gate_36h"] is True
    assert out_pass["passes_senior_target_72h"] is True

    assert main(["battery", "--capacity-mah", "500", "--avg-current-ma", "80.0"]) == 1
    out_fail = json.loads(capsys.readouterr().out)
    assert out_fail["passes_wearable_gate_36h"] is False

    gate_file = tmp_path / "gate.json"
    gate_file.write_text(
        json.dumps(
            {
                "worn_in_coverage_s": 1000.0,
                "accepted_s": 600.0,
                "room_only_shared_room_s_by_room": {"living-room": 320.0},
                "calibrated_rooms": ["living-room"],
            }
        ),
        encoding="utf-8",
    )
    assert main(["necessity-gate", "--input", str(gate_file)]) == 0
    gate_out = json.loads(capsys.readouterr().out)
    assert gate_out["gate_met"] is True
