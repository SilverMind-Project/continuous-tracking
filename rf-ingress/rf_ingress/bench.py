"""UWB bench accuracy replay, battery projection, and necessity-gate evaluation tooling.

Provides deterministic offline analysis helpers and a CLI (``python -m rf_ingress.bench``)
for evaluating:
- Ranging bench accuracy (horizontal median error, 95th percentile error, fix availability)
  against tape-measured ground-truth floor points, plus direct-solver vs. MQTT adapter
  replay parity within a configurable relative tolerance (default 10%).
- Powered-tag battery runtime projection from measured scripted-hour average current against
  the 36-hour general wearable gate and 72-hour senior wearable target.
- Room-level wearable necessity gate evaluation from 30-day live association statistics and
  room camera calibration status.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from cts_contracts.wearables import AdapterKind, FixKind

from rf_ingress.adapters._mqtt_transport import InMemoryMqttMessageStream
from rf_ingress.adapters.anchor_ranging_mqtt import (
    AnchorNodeCalibrationConfig,
    AnchorRangingMqttAdapter,
)
from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.domain import (
    EmittedFix,
    RangeMeasurement,
    SiteConfigSnapshot,
    ensure_utc,
)
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher
from rf_ingress.site_config import InMemorySiteConfigProvider
from rf_ingress.solver.epoch import EpochBuilder
from rf_ingress.solver.multilateration import MultilaterationConfig, MultilaterationSolver

UWB_BENCH_MAX_MEDIAN_ERROR_M = 0.30
UWB_BENCH_MAX_P95_ERROR_M = 1.00
UWB_BENCH_MIN_AVAILABILITY = 0.80
UWB_ADAPTER_PARITY_MAX_REL_DIFF = 0.10

WEARABLE_MIN_RUNTIME_HOURS = 36.0
SENIOR_TARGET_RUNTIME_HOURS = 72.0
NECESSITY_MIN_SHARED_ROOM_SHARE = 0.25


@dataclass(frozen=True)
class BenchGroundTruthPoint:
    """One tape-measured floor ground-truth interval for a ranging bench run.

    ``x_m``, ``y_m``, and ``z_m`` are metres in the shared household floor frame.
    Fixes whose ``measured_at`` falls inside ``[start_at, end_at)`` are evaluated
    against this ground-truth position.
    """

    point_id: str
    x_m: float
    y_m: float
    z_m: float
    start_at: datetime
    end_at: datetime
    facing_dir: str = ""

    def __post_init__(self) -> None:
        start = ensure_utc(self.start_at)
        end = ensure_utc(self.end_at)
        if end <= start:
            raise ValueError("Ground-truth interval end_at must be strictly after start_at")
        if not (math.isfinite(self.x_m) and math.isfinite(self.y_m) and math.isfinite(self.z_m)):
            raise ValueError("Ground-truth coordinates must be finite")

    def covers(self, ts: datetime) -> bool:
        """Return ``True`` when *ts* lies inside ``[start_at, end_at)``."""
        t = ensure_utc(ts)
        return ensure_utc(self.start_at) <= t < ensure_utc(self.end_at)


@dataclass(frozen=True)
class BenchAccuracyStats:
    """Summary accuracy and availability metrics for a ranging bench session."""

    expected_epochs: int
    solved_position_fixes: int
    evaluated_position_fixes: int
    availability_fraction: float
    median_error_m: float | None
    p95_error_m: float | None
    mean_error_m: float | None
    residual_rms_median_m: float | None
    passed_median: bool
    passed_p95: bool
    passed_availability: bool

    @property
    def passed(self) -> bool:
        """Return ``True`` only when median, p95, and availability gates all pass."""
        return self.passed_median and self.passed_p95 and self.passed_availability


@dataclass(frozen=True)
class AdapterBenchParityResult:
    """Comparison of direct solver replay vs. MQTT adapter replay on identical bench data."""

    direct_stats: BenchAccuracyStats
    adapter_stats: BenchAccuracyStats
    median_rel_diff: float | None
    p95_rel_diff: float | None
    availability_rel_diff: float
    within_tolerance: bool


@dataclass(frozen=True)
class BatteryRuntimeProjection:
    """Projected powered-tag battery runtime from bench current profiler measurements."""

    cell_capacity_mah: float
    usable_fraction: float
    avg_current_ma: float
    projected_hours: float
    passes_wearable_gate_36h: bool
    passes_senior_target_72h: bool


@dataclass(frozen=True)
class NecessityGateReport:
    """Outcome of the 30-day room-level wearable necessity gate evaluation."""

    worn_in_coverage_s: float
    accepted_s: float
    room_only_shared_room_s: float
    share_a: float
    share_b: float
    shared_room_shares: dict[str, float]
    responsible_rooms: tuple[str, ...]
    uncalibrated_responsible_rooms: tuple[str, ...]
    gate_met: bool
    reason: str


def _ground_truth_for(
    ts: datetime,
    points: Sequence[BenchGroundTruthPoint],
) -> BenchGroundTruthPoint | None:
    for point in points:
        if point.covers(ts):
            return point
    return None


def evaluate_bench_fixes(
    fixes: Sequence[EmittedFix],
    ground_truth_points: Sequence[BenchGroundTruthPoint],
    *,
    expected_epochs: int,
    max_median_error_m: float = UWB_BENCH_MAX_MEDIAN_ERROR_M,
    max_p95_error_m: float = UWB_BENCH_MAX_P95_ERROR_M,
    min_availability: float = UWB_BENCH_MIN_AVAILABILITY,
) -> BenchAccuracyStats:
    """Compute horizontal error statistics and availability for solved bench fixes."""
    if expected_epochs <= 0:
        raise ValueError(f"expected_epochs must be positive, got {expected_epochs}")

    pos_fixes = [
        fix
        for fix in fixes
        if fix.kind == FixKind.POSITION and fix.x_mm is not None and fix.y_mm is not None
    ]
    errors_m: list[float] = []
    residuals_m: list[float] = []
    for fix in pos_fixes:
        gt = _ground_truth_for(fix.measured_at, ground_truth_points)
        if gt is None:
            continue
        assert fix.x_mm is not None and fix.y_mm is not None
        dx_m = (fix.x_mm / 1000.0) - gt.x_m
        dy_m = (fix.y_mm / 1000.0) - gt.y_m
        errors_m.append(math.hypot(dx_m, dy_m))
        if fix.residual_rms_m is not None:
            residuals_m.append(fix.residual_rms_m)

    availability = min(1.0, len(errors_m) / float(expected_epochs))
    if not errors_m:
        return BenchAccuracyStats(
            expected_epochs=expected_epochs,
            solved_position_fixes=len(pos_fixes),
            evaluated_position_fixes=0,
            availability_fraction=availability,
            median_error_m=None,
            p95_error_m=None,
            mean_error_m=None,
            residual_rms_median_m=None,
            passed_median=False,
            passed_p95=False,
            passed_availability=False,
        )

    arr = np.asarray(errors_m, dtype=np.float64)
    median_err = float(np.median(arr))
    p95_err = float(np.percentile(arr, 95))
    mean_err = float(np.mean(arr))
    res_med = float(np.median(np.asarray(residuals_m, dtype=np.float64))) if residuals_m else None

    return BenchAccuracyStats(
        expected_epochs=expected_epochs,
        solved_position_fixes=len(pos_fixes),
        evaluated_position_fixes=len(errors_m),
        availability_fraction=availability,
        median_error_m=median_err,
        p95_error_m=p95_err,
        mean_error_m=mean_err,
        residual_rms_median_m=res_med,
        passed_median=median_err <= max_median_error_m,
        passed_p95=p95_err <= max_p95_error_m,
        passed_availability=availability >= min_availability,
    )


def _relative_diff(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    denom = max(abs(reference), 1e-6)
    return abs(candidate - reference) / denom


def compare_adapter_bench_parity(
    direct_stats: BenchAccuracyStats,
    adapter_stats: BenchAccuracyStats,
    *,
    max_rel_diff: float = UWB_ADAPTER_PARITY_MAX_REL_DIFF,
) -> AdapterBenchParityResult:
    """Compare direct solver bench stats against full MQTT adapter replay stats."""
    if not math.isfinite(max_rel_diff) or max_rel_diff < 0.0:
        raise ValueError(f"max_rel_diff must be finite and >= 0, got {max_rel_diff!r}")
    med_diff = _relative_diff(direct_stats.median_error_m, adapter_stats.median_error_m)
    p95_diff = _relative_diff(direct_stats.p95_error_m, adapter_stats.p95_error_m)
    avail_diff = abs(adapter_stats.availability_fraction - direct_stats.availability_fraction) / (
        max(abs(direct_stats.availability_fraction), 1e-6)
    )
    within = (
        med_diff is not None
        and p95_diff is not None
        and med_diff <= max_rel_diff
        and p95_diff <= max_rel_diff
        and avail_diff <= max_rel_diff
    )
    return AdapterBenchParityResult(
        direct_stats=direct_stats,
        adapter_stats=adapter_stats,
        median_rel_diff=med_diff,
        p95_rel_diff=p95_diff,
        availability_rel_diff=avail_diff,
        within_tolerance=within,
    )


def solve_ranges_directly(
    measurements: Sequence[RangeMeasurement],
    site_config: SiteConfigSnapshot,
    *,
    epoch_window_s: float = 0.25,
    solver_config: MultilaterationConfig | None = None,
) -> list[EmittedFix]:
    """Group *measurements* with :class:`EpochBuilder` and solve via :class:`MultilaterationSolver`.

    Provides the reference baseline for adapter-vs-solver parity verification.
    """
    builder = EpochBuilder(epoch_window_s=epoch_window_s)
    nodes_by_adapter: dict[str, set[str]] = {}
    for node in site_config.nodes:
        nodes_by_adapter.setdefault(node.adapter_id, set()).add(node.anchor_id)
    builder.on_site_config_version(
        site_config.version or "bench:v1",
        nodes_by_adapter={aid: frozenset(ids) for aid, ids in nodes_by_adapter.items()},
    )
    solver = MultilaterationSolver(solver_config)
    nodes = site_config.node_map()

    epochs = []
    for meas in measurements:
        epochs.extend(builder.add(meas))
    epochs.extend(builder.flush_all())

    fixes: list[EmittedFix] = []
    for epoch in epochs:
        result = solver.solve(epoch, nodes)
        if result is None:
            continue
        fixes.append(
            EmittedFix(
                tag_id=epoch.tag_id,
                measured_at=epoch.measured_at,
                kind=FixKind.POSITION,
                epoch_seq=epoch.epoch_seq,
                adapter_id=epoch.adapter_id,
                adapter_kind=AdapterKind.ANCHOR_RANGING,
                adapter_version="direct-bench",
                floor_plan_id=site_config.floor_plan_id or "",
                x_mm=round(result.x_m * 1000.0),
                y_mm=round(result.y_m * 1000.0),
                z_mm=round(result.z_m * 1000.0) if result.z_m is not None else None,
                cov_xx_m2=result.cov_xx_m2,
                cov_xy_m2=result.cov_xy_m2,
                cov_yy_m2=result.cov_yy_m2,
                wear_state=epoch.wear_state,
                motion_state=epoch.motion_state,
                quality=result.quality,
                anchors_used=result.anchors_used,
                residual_rms_m=result.residual_rms_m,
                battery_fraction=epoch.battery_fraction,
            )
        )
    return fixes


async def replay_fira_anchor_bench_session(
    mqtt_messages: Sequence[tuple[str, bytes]],
    *,
    site_config: SiteConfigSnapshot,
    ground_truth_points: Sequence[BenchGroundTruthPoint],
    expected_epochs: int,
    adapter_id: str = "uwb-anchors",
    uwb_default_sigma_m: float = 0.25,
    sigma_floor_m: float = 0.10,
    node_calibrations: Mapping[str, AnchorNodeCalibrationConfig] | None = None,
    solver_config: MultilaterationConfig | None = None,
    epoch_window_s: float = 0.25,
    max_rel_diff: float = UWB_ADAPTER_PARITY_MAX_REL_DIFF,
) -> AdapterBenchParityResult:
    """Replay recorded ``fira_anchor.v1`` MQTT messages through both adapter and direct solver."""
    provider = InMemorySiteConfigProvider(site_config)

    # First pass: collect raw RangeMeasurements from the adapter for the direct-solver baseline.
    extract_adapter = AnchorRangingMqttAdapter(
        adapter_id=adapter_id,
        deps=AdapterDeps(site_config=provider),
        stream=InMemoryMqttMessageStream(mqtt_messages),
        uwb_default_sigma_m=uwb_default_sigma_m,
        sigma_floor_m=sigma_floor_m,
        node_calibrations=node_calibrations,
    )
    raw_measurements: list[RangeMeasurement] = []
    async for meas in extract_adapter.ranges():
        raw_measurements.append(meas)
    await extract_adapter.close()

    direct_fixes = solve_ranges_directly(
        raw_measurements,
        site_config,
        epoch_window_s=epoch_window_s,
        solver_config=solver_config,
    )
    direct_stats = evaluate_bench_fixes(
        direct_fixes,
        ground_truth_points,
        expected_epochs=expected_epochs,
    )

    # Second pass: replay identical messages end-to-end through RfIngressRuntime
    # using each report's recorded timestamp as arrival_at so historical bench
    # sessions replay deterministically without wall-clock offset drift.
    runtime_adapter = AnchorRangingMqttAdapter(
        adapter_id=adapter_id,
        deps=AdapterDeps(site_config=provider),
        stream=InMemoryMqttMessageStream(mqtt_messages),
        uwb_default_sigma_m=uwb_default_sigma_m,
        sigma_floor_m=sigma_floor_m,
        node_calibrations=node_calibrations,
    )
    publisher = InMemoryTagFixPublisher()
    runtime = RfIngressRuntime(
        adapters=[runtime_adapter],
        site_config=provider,
        publisher=publisher,
        solver_config=solver_config,
        epoch_window_s=epoch_window_s,
    )
    async for meas in runtime_adapter.ranges():
        replay_arrival = meas.sent_at if meas.sent_at is not None else meas.measured_at
        await runtime.handle_range_measurement(
            runtime_adapter,
            meas,
            arrival_at=replay_arrival,
        )
    # Flush any final open epoch via the public runtime lifecycle API.
    await runtime.run_once()
    await runtime_adapter.close()

    adapter_stats = evaluate_bench_fixes(
        [rec.fix for rec in publisher.fixes],
        ground_truth_points,
        expected_epochs=expected_epochs,
    )
    return compare_adapter_bench_parity(
        direct_stats,
        adapter_stats,
        max_rel_diff=max_rel_diff,
    )


def project_battery_runtime(
    *,
    cell_capacity_mah: float,
    avg_current_ma: float,
    usable_fraction: float = 0.85,
) -> BatteryRuntimeProjection:
    """Project wearable battery runtime from measured average current and LiPo capacity.

    ``usable_fraction`` defaults to 0.85 to account for LiPo cutoff voltage and
    temperature derating; pass ``1.0`` for raw nominal capacity division.
    """
    if not math.isfinite(cell_capacity_mah) or cell_capacity_mah <= 0.0:
        raise ValueError(f"cell_capacity_mah must be finite and > 0, got {cell_capacity_mah!r}")
    if not math.isfinite(avg_current_ma) or avg_current_ma <= 0.0:
        raise ValueError(f"avg_current_ma must be finite and > 0, got {avg_current_ma!r}")
    if not math.isfinite(usable_fraction) or not (0.0 < usable_fraction <= 1.0):
        raise ValueError(f"usable_fraction must be in (0, 1], got {usable_fraction!r}")

    projected_hours = (cell_capacity_mah * usable_fraction) / avg_current_ma
    return BatteryRuntimeProjection(
        cell_capacity_mah=cell_capacity_mah,
        usable_fraction=usable_fraction,
        avg_current_ma=avg_current_ma,
        projected_hours=projected_hours,
        passes_wearable_gate_36h=projected_hours >= WEARABLE_MIN_RUNTIME_HOURS,
        passes_senior_target_72h=projected_hours >= SENIOR_TARGET_RUNTIME_HOURS,
    )


def evaluate_necessity_gate(
    *,
    worn_in_coverage_s: float,
    accepted_s: float,
    room_only_shared_room_s_by_room: Mapping[str, float],
    calibrated_rooms: Sequence[str],
    min_shared_room_share: float = NECESSITY_MIN_SHARED_ROOM_SHARE,
) -> NecessityGateReport:
    """Evaluate the 30-day wearable necessity gate for precision UWB deployment.

    The gate passes only when:
    1. ``worn_in_coverage_s > 0``;
    2. share ``B = room_only_shared_room_s / worn_in_coverage_s >= min_shared_room_share``
       (default 0.25); and
    3. every responsible room contributing to ``B`` has calibrated cameras.
    """
    if not math.isfinite(worn_in_coverage_s) or worn_in_coverage_s < 0.0:
        raise ValueError("worn_in_coverage_s must be finite and >= 0")
    if not math.isfinite(accepted_s) or accepted_s < 0.0:
        raise ValueError("accepted_s must be finite and >= 0")
    if not math.isfinite(min_shared_room_share) or not (0.0 < min_shared_room_share <= 1.0):
        raise ValueError("min_shared_room_share must be in (0, 1]")

    for room_id, seconds in room_only_shared_room_s_by_room.items():
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"Invalid shared-room duration for {room_id!r}: {seconds!r}")

    total_shared_s = sum(room_only_shared_room_s_by_room.values())
    if worn_in_coverage_s == 0.0:
        return NecessityGateReport(
            worn_in_coverage_s=0.0,
            accepted_s=accepted_s,
            room_only_shared_room_s=total_shared_s,
            share_a=0.0,
            share_b=0.0,
            shared_room_shares={},
            responsible_rooms=(),
            uncalibrated_responsible_rooms=(),
            gate_met=False,
            reason="no_worn_coverage_data",
        )

    share_a = min(1.0, accepted_s / worn_in_coverage_s)
    share_b = min(1.0, total_shared_s / worn_in_coverage_s)
    per_room_share = {
        room: seconds / worn_in_coverage_s
        for room, seconds in sorted(
            room_only_shared_room_s_by_room.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if seconds > 0.0
    }
    responsible = tuple(per_room_share.keys())
    calibrated_set = set(calibrated_rooms)
    uncalibrated = tuple(room for room in responsible if room not in calibrated_set)

    if share_b < min_shared_room_share:
        return NecessityGateReport(
            worn_in_coverage_s=worn_in_coverage_s,
            accepted_s=accepted_s,
            room_only_shared_room_s=total_shared_s,
            share_a=share_a,
            share_b=share_b,
            shared_room_shares=per_room_share,
            responsible_rooms=responsible,
            uncalibrated_responsible_rooms=uncalibrated,
            gate_met=False,
            reason="shared_room_share_below_threshold",
        )
    if uncalibrated:
        return NecessityGateReport(
            worn_in_coverage_s=worn_in_coverage_s,
            accepted_s=accepted_s,
            room_only_shared_room_s=total_shared_s,
            share_a=share_a,
            share_b=share_b,
            shared_room_shares=per_room_share,
            responsible_rooms=responsible,
            uncalibrated_responsible_rooms=uncalibrated,
            gate_met=False,
            reason="responsible_rooms_uncalibrated",
        )
    return NecessityGateReport(
        worn_in_coverage_s=worn_in_coverage_s,
        accepted_s=accepted_s,
        room_only_shared_room_s=total_shared_s,
        share_a=share_a,
        share_b=share_b,
        shared_room_shares=per_room_share,
        responsible_rooms=responsible,
        uncalibrated_responsible_rooms=(),
        gate_met=True,
        reason="necessity_gate_met",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rf_ingress.bench",
        description="Offline UWB bench battery projection and necessity-gate calculator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bat = sub.add_parser("battery", help="Project tag battery runtime from bench profiler current")
    bat.add_argument("--capacity-mah", type=float, required=True, help="LiPo cell capacity in mAh")
    bat.add_argument(
        "--avg-current-ma",
        type=float,
        required=True,
        help="Measured scripted-hour average current in mA",
    )
    bat.add_argument(
        "--usable-fraction",
        type=float,
        default=0.85,
        help="Usable cell capacity fraction (default 0.85; use 1.0 for nominal)",
    )

    gate = sub.add_parser(
        "necessity-gate",
        help="Evaluate the 30-day shared-room necessity gate from JSON summary inputs",
    )
    gate.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON file with worn_in_coverage_s, accepted_s, "
        "room_only_shared_room_s_by_room, and calibrated_rooms",
    )
    return parser


def _run_cli(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "battery":
        proj = project_battery_runtime(
            cell_capacity_mah=args.capacity_mah,
            avg_current_ma=args.avg_current_ma,
            usable_fraction=args.usable_fraction,
        )
        bat_out: dict[str, object] = {
            "cell_capacity_mah": proj.cell_capacity_mah,
            "usable_fraction": proj.usable_fraction,
            "avg_current_ma": proj.avg_current_ma,
            "projected_hours": round(proj.projected_hours, 2),
            "passes_wearable_gate_36h": proj.passes_wearable_gate_36h,
            "passes_senior_target_72h": proj.passes_senior_target_72h,
        }
        sys.stdout.write(json.dumps(bat_out, indent=2, sort_keys=True) + "\n")
        return 0 if proj.passes_wearable_gate_36h else 1

    if args.command == "necessity-gate":
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        report = evaluate_necessity_gate(
            worn_in_coverage_s=float(raw["worn_in_coverage_s"]),
            accepted_s=float(raw["accepted_s"]),
            room_only_shared_room_s_by_room={
                str(k): float(v) for k, v in raw["room_only_shared_room_s_by_room"].items()
            },
            calibrated_rooms=[str(r) for r in raw["calibrated_rooms"]],
        )
        gate_out: dict[str, object] = {
            "worn_in_coverage_s": report.worn_in_coverage_s,
            "accepted_s": report.accepted_s,
            "room_only_shared_room_s": report.room_only_shared_room_s,
            "share_a": round(report.share_a, 4),
            "share_b": round(report.share_b, 4),
            "shared_room_shares": {k: round(v, 4) for k, v in report.shared_room_shares.items()},
            "responsible_rooms": list(report.responsible_rooms),
            "uncalibrated_responsible_rooms": list(report.uncalibrated_responsible_rooms),
            "gate_met": report.gate_met,
            "reason": report.reason,
        }
        sys.stdout.write(json.dumps(gate_out, indent=2, sort_keys=True) + "\n")
        return 0 if report.gate_met else 1

    return 2


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m rf_ingress.bench``."""
    return _run_cli(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
