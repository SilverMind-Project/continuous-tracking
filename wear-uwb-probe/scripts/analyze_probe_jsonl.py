"""Offline telemetry analyzer for standalone Wear OS UWB bench probe JSONL logs.

Evaluates pulled JSONL files across all four probe stages:
- Stage P1: backend capability audit and controller-session readiness gate.
- Stage P2: OOB handshake latency (Pre-Shared STS broadcast vs. BLE GATT) and
  multicast session stability gate.
- Stage P3: distance-only multilateration accuracy via ``rf_ingress.solver.multilateration``,
  wrist-rotation azimuth coupling (raw vs. quaternion-compensated), and cardinal-facing
  body-shadowing distance bias.
- Stage P4: coulomb-counter battery drain, added daily drain over BLE control, thermal
  throttling, seated false burst rate, and projected full-charge watch runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from cts_contracts.wearables import MotionState, RangingTechnology, WearState
from rf_ingress.domain import RangeEpoch, RangeMeasurement, RfNodeGeometry
from rf_ingress.solver.multilateration import MultilaterationConfig, MultilaterationSolver

P2_MAX_FIRST_3_ANCHOR_MS = 2000.0
P2_PRESHARED_TARGET_MS = 500.0
P2_MIN_SUSTAINED_RATE_HZ = 1.0
P2_MIN_EPOCH_SHARE = 0.90

P3_MAX_MEDIAN_ERROR_M = 0.30
P3_MAX_P95_ERROR_M = 1.00
P3_MIN_AVAILABILITY = 0.80
P3_DEFAULT_UWB_SIGMA_M = 0.25

P4_MIN_PROJECTED_HOURS = 30.0
P4_WEARABLE_36H_HOURS = 36.0
P4_MAX_ADDED_DRAIN_PP_PER_24H = 15.0
P4_MAX_SEATED_FALSE_TRIGGERS_PER_HOUR = 1.0
DEFAULT_WATCH_CAPACITY_MAH = 390.0

_THERMAL_ORDER: dict[str, int] = {
    "NONE": 0,
    "LIGHT": 1,
    "MODERATE": 2,
    "SEVERE": 3,
    "CRITICAL": 4,
}


@dataclass(frozen=True)
class AnchorDefinition:
    """Known floor-frame anchor coordinates (metres) keyed by FiRa short address."""

    anchor_id: str
    uwb_address_hex: str
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class GroundTruthPoint:
    """Tape-measured floor point coordinates (metres in the household floor frame)."""

    point_id: str
    x_m: float
    y_m: float
    z_m: float = 1.0


@dataclass(frozen=True)
class P1BackendRow:
    """Capability audit summary row for one UWB backend."""

    backend_name: str
    feature_flag_present: bool
    is_available: bool
    supports_distance: bool
    supports_azimuth: bool
    supports_elevation: bool
    supports_multicast_ds_twr: bool
    local_address_hex: str | None
    assigned_channel: int | None
    assigned_preamble_index: int | None
    raw_status_code: int | None
    error_detail: str | None
    screen_off_foreground_service: bool
    passes_controller_gate: bool


@dataclass(frozen=True)
class P1Report:
    """Stage P1 capability audit report."""

    rows: tuple[P1BackendRow, ...]
    winning_backends: tuple[str, ...]
    gate_passed: bool


@dataclass(frozen=True)
class P2StrategyStats:
    """Stage P2 OOB and multicast summary for one negotiation strategy."""

    strategy: str
    trials: int
    oob_success_rate: float
    oob_latency_median_ms: float | None
    oob_latency_p95_ms: float | None
    first_range_median_ms: float | None
    first_3_anchor_median_ms: float | None
    first_3_anchor_p95_ms: float | None
    median_peers_acquired: float | None
    min_peer_rate_hz: float | None
    three_anchor_epoch_share: float
    passes_gate: bool


@dataclass(frozen=True)
class P2Report:
    """Stage P2 OOB handshake and multicast stability report."""

    strategies: tuple[P2StrategyStats, ...]
    gate_passed: bool


@dataclass(frozen=True)
class FacingDirectionStats:
    """Distance bias and spread for one cardinal body-facing direction."""

    facing_dir: str
    sample_count: int
    mean_range_bias_m: float
    std_range_error_m: float


@dataclass(frozen=True)
class P3Report:
    """Stage P3 multilateration accuracy, wrist azimuth, and shadowing report."""

    expected_epochs: int
    solved_epochs: int
    availability_fraction: float
    median_horizontal_error_m: float | None
    p95_horizontal_error_m: float | None
    mean_horizontal_error_m: float | None
    raw_azimuth_std_deg: float | None
    compensated_world_azimuth_std_deg: float | None
    facing_direction_stats: tuple[FacingDirectionStats, ...]
    passed_median: bool
    passed_p95: bool
    passed_availability: bool
    gate_passed: bool


@dataclass(frozen=True)
class P4RegimeStats:
    """Stage P4 coulomb-counter drain, thermal, and burst stats for one regime."""

    regime: str
    duration_hours: float
    delta_uah: int | None
    drain_uah_per_hour: float | None
    drain_percent_per_hour: float | None
    peak_current_ma: float | None
    max_thermal_status: str
    thermal_throttle_events: int
    bursts_per_hour: float | None
    seated_false_triggers_per_hour: float | None
    projected_runtime_hours: float | None


@dataclass(frozen=True)
class P4Report:
    """Stage P4 battery and thermal benchmark report."""

    cell_capacity_mah: float
    regimes: tuple[P4RegimeStats, ...]
    regime_c_added_drain_pp_per_24h: float | None
    regime_c_passes_30h_or_added_drain: bool
    regime_c_passes_36h_wearable_rule: bool
    regime_c_zero_thermal_throttling: bool
    regime_c_passes_false_triggers: bool
    gate_passed: bool


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from a UTF-8 JSONL file."""
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        records.append(parsed)
    return records


def load_anchors(path: Path) -> dict[str, AnchorDefinition]:
    """Load anchor floor-frame coordinates keyed by uppercase ``XX:XX`` UWB address."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["anchors"] if isinstance(raw, dict) and "anchors" in raw else raw
    result: dict[str, AnchorDefinition] = {}
    for entry in items:
        addr = str(entry["uwb_address_hex"]).strip().upper()
        result[addr] = AnchorDefinition(
            anchor_id=str(entry["anchor_id"]),
            uwb_address_hex=addr,
            x_m=float(entry["x_m"]),
            y_m=float(entry["y_m"]),
            z_m=float(entry["z_m"]),
        )
    return result


def load_ground_truth_points(path: Path) -> dict[str, GroundTruthPoint]:
    """Load tape-measured ground-truth floor points keyed by ``point_id``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["points"] if isinstance(raw, dict) and "points" in raw else raw
    result: dict[str, GroundTruthPoint] = {}
    for entry in items:
        pid = str(entry["point_id"]).strip()
        result[pid] = GroundTruthPoint(
            point_id=pid,
            x_m=float(entry["x_m"]),
            y_m=float(entry["y_m"]),
            z_m=float(entry.get("z_m", 1.0)),
        )
    return result


def analyze_p1(records: Sequence[Mapping[str, Any]]) -> P1Report:
    """Analyze Stage P1 capability records and evaluate Gate P1."""
    rows: list[P1BackendRow] = []
    for rec in records:
        if rec.get("record_type") != "p1_capability":
            continue
        caps = rec.get("capabilities", {})
        backend = str(caps.get("backendName", "unknown"))
        is_avail = bool(caps.get("isAvailable", False))
        local_hex = caps.get("localAddressHex")
        channel = caps.get("assignedChannel")
        preamble = caps.get("assignedPreambleIndex")
        status_code = caps.get("rawStatusCode")
        passes = (
            is_avail
            and bool(local_hex)
            and channel is not None
            and preamble is not None
            and status_code is None
        )
        rows.append(
            P1BackendRow(
                backend_name=backend,
                feature_flag_present=bool(caps.get("featureFlagPresent", False)),
                is_available=is_avail,
                supports_distance=bool(caps.get("supportsDistance", False)),
                supports_azimuth=bool(caps.get("supportsAzimuth", False)),
                supports_elevation=bool(caps.get("supportsElevation", False)),
                supports_multicast_ds_twr=bool(caps.get("supportsMulticastDsTwr", False)),
                local_address_hex=str(local_hex) if local_hex is not None else None,
                assigned_channel=int(channel) if channel is not None else None,
                assigned_preamble_index=int(preamble) if preamble is not None else None,
                raw_status_code=int(status_code) if status_code is not None else None,
                error_detail=str(caps["errorDetail"]) if caps.get("errorDetail") else None,
                screen_off_foreground_service=bool(rec.get("screenOffForegroundService", False)),
                passes_controller_gate=passes,
            )
        )

    winners = tuple(
        row.backend_name
        for row in rows
        if row.passes_controller_gate and row.backend_name in {"gms_nearby_uwb", "jetpack_uwb"}
    )
    return P1Report(rows=tuple(rows), winning_backends=winners, gate_passed=len(winners) > 0)


def analyze_p2(
    records: Sequence[Mapping[str, Any]],
    *,
    epoch_bucket_s: float = 1.0,
) -> P2Report:
    """Analyze Stage P2 OOB trials and multicast sessions and evaluate Gate P2."""
    if epoch_bucket_s <= 0.0:
        raise ValueError("epoch_bucket_s must be positive")

    trial_strategy: dict[int, str] = {}
    oob_by_strategy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    summaries_by_strategy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    samples_by_strategy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for rec in records:
        rtype = rec.get("record_type")
        if rtype == "p2_oob_trial":
            trial_idx = int(rec["trialIndex"])
            result = rec["result"]
            strat = str(result["strategy"])
            trial_strategy[trial_idx] = strat
            oob_by_strategy[strat].append(result)

    for rec in records:
        rtype = rec.get("record_type")
        if rtype == "p2_session_summary":
            trial_idx = int(rec["trialIndex"])
            strat = trial_strategy.get(trial_idx, "UNKNOWN")
            summaries_by_strategy[strat].append(rec["summary"])
        elif rtype == "p2_ranging_sample":
            trial_idx = int(rec["trialIndex"])
            strat = trial_strategy.get(trial_idx, "UNKNOWN")
            samples_by_strategy[strat].append(rec["sample"])

    bucket_ns = int(epoch_bucket_s * 1_000_000_000.0)
    stats_list: list[P2StrategyStats] = []
    for strat in sorted(oob_by_strategy.keys()):
        oob_items = oob_by_strategy[strat]
        trials = len(oob_items)
        succeeded = sum(1 for item in oob_items if bool(item.get("succeeded")))
        latencies = [float(item["setupLatencyMs"]) for item in oob_items if item.get("succeeded")]

        summaries = summaries_by_strategy.get(strat, [])
        first_range = [
            float(s["timeToFirstRangeMs"])
            for s in summaries
            if s.get("timeToFirstRangeMs") is not None
        ]
        first_3 = [
            float(s["timeToFirstThreeAnchorEpochMs"])
            for s in summaries
            if s.get("timeToFirstThreeAnchorEpochMs") is not None
        ]
        peers_acq = [float(s["distinctPeersAcquired"]) for s in summaries]

        peer_rates: dict[str, list[float]] = defaultdict(list)
        for s in summaries:
            rate_map = s.get("effectiveRateHzByPeer", {})
            if isinstance(rate_map, Mapping):
                for peer, rate in rate_map.items():
                    peer_rates[str(peer)].append(float(rate))
        mean_rates_per_peer = [float(np.mean(vals)) for vals in peer_rates.values() if vals]
        top3_rates = sorted(mean_rates_per_peer, reverse=True)[:3]
        min_top3_rate = min(top3_rates) if len(top3_rates) >= 3 else 0.0

        # Compute share of time buckets containing >= 3 distinct anchors.
        buckets: dict[int, set[str]] = defaultdict(set)
        for sample in samples_by_strategy.get(strat, []):
            ts_ns = int(sample["elapsedRealtimeNanos"])
            buckets[ts_ns // bucket_ns].add(str(sample["peerAddressHex"]).upper())
        if buckets:
            min_b, max_b = min(buckets.keys()), max(buckets.keys())
            total_buckets = max(1, max_b - min_b + 1)
            good_buckets = sum(1 for peers in buckets.values() if len(peers) >= 3)
            epoch_share = good_buckets / float(total_buckets)
        else:
            epoch_share = 0.0

        first_3_med = float(np.median(first_3)) if first_3 else None
        passes = (
            first_3_med is not None
            and first_3_med <= P2_MAX_FIRST_3_ANCHOR_MS
            and min_top3_rate >= P2_MIN_SUSTAINED_RATE_HZ
            and epoch_share >= P2_MIN_EPOCH_SHARE
        )
        stats_list.append(
            P2StrategyStats(
                strategy=strat,
                trials=trials,
                oob_success_rate=succeeded / float(trials) if trials > 0 else 0.0,
                oob_latency_median_ms=float(np.median(latencies)) if latencies else None,
                oob_latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
                first_range_median_ms=float(np.median(first_range)) if first_range else None,
                first_3_anchor_median_ms=first_3_med,
                first_3_anchor_p95_ms=float(np.percentile(first_3, 95)) if first_3 else None,
                median_peers_acquired=float(np.median(peers_acq)) if peers_acq else None,
                min_peer_rate_hz=min_top3_rate if peer_rates else None,
                three_anchor_epoch_share=epoch_share,
                passes_gate=passes,
            )
        )

    return P2Report(
        strategies=tuple(stats_list),
        gate_passed=any(s.passes_gate for s in stats_list),
    )


def _circular_std_deg(angles_deg: Sequence[float]) -> float | None:
    if len(angles_deg) < 2:
        return None
    rad = np.radians(np.asarray(angles_deg, dtype=np.float64))
    mean_cos = float(np.mean(np.cos(rad)))
    mean_sin = float(np.mean(np.sin(rad)))
    r_bar = math.hypot(mean_cos, mean_sin)
    r_clamped = min(1.0, max(1e-12, r_bar))
    return math.degrees(math.sqrt(-2.0 * math.log(r_clamped)))


def _rotate_vector_by_quat(
    q: Mapping[str, Any],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    w = float(q["w"])
    x = float(q["x"])
    y = float(q["y"])
    z = float(q["z"])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return vec
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    vx, vy, vz = vec
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


def analyze_p3(
    records: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, AnchorDefinition],
    ground_truth: Mapping[str, GroundTruthPoint],
    *,
    epoch_window_s: float = 0.25,
    expected_epochs: int | None = None,
    uwb_sigma_m: float = P3_DEFAULT_UWB_SIGMA_M,
    solver_config: MultilaterationConfig | None = None,
) -> P3Report:
    """Solve Stage P3 ranges with ``MultilaterationSolver`` and characterize AoA/shadowing."""
    if epoch_window_s <= 0.0:
        raise ValueError("epoch_window_s must be positive")
    if uwb_sigma_m <= 0.0:
        raise ValueError("uwb_sigma_m must be positive")

    nodes: dict[str, RfNodeGeometry] = {
        defn.anchor_id: RfNodeGeometry(
            anchor_id=defn.anchor_id,
            adapter_id="wear-uwb-probe",
            x_m=defn.x_m,
            y_m=defn.y_m,
            z_m=defn.z_m,
            uwb_address_hex=defn.uwb_address_hex,
        )
        for defn in anchors.values()
    }
    solver = MultilaterationSolver(solver_config)
    variance_m2 = uwb_sigma_m * uwb_sigma_m
    bucket_ns = int(epoch_window_s * 1_000_000_000.0)

    # Group samples into per-point, per-facing, per-epoch-bucket collections.
    grouped_buckets: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    facing_errors: dict[str, list[float]] = defaultdict(list)
    raw_azimuth_by_peer: dict[tuple[str, str], list[float]] = defaultdict(list)
    world_azimuth_by_peer: dict[tuple[str, str], list[float]] = defaultdict(list)

    for rec in records:
        if rec.get("record_type") != "p3_ranging_orientation":
            continue
        point_id = str(rec["pointId"])
        facing_dir = str(rec.get("facingDir", "N")).upper()
        sample = rec["sample"]
        peer_hex = str(sample["peerAddressHex"]).strip().upper()
        dist = sample.get("distanceM")
        if dist is None or not math.isfinite(float(dist)) or float(dist) <= 0.0:
            continue

        ts_ns = int(sample["elapsedRealtimeNanos"])
        grouped_buckets[(point_id, facing_dir, ts_ns // bucket_ns)].append(rec)

        anchor_def = anchors.get(peer_hex)
        gt = ground_truth.get(point_id)
        if anchor_def is not None and gt is not None:
            true_dist = math.sqrt(
                (gt.x_m - anchor_def.x_m) ** 2
                + (gt.y_m - anchor_def.y_m) ** 2
                + (gt.z_m - anchor_def.z_m) ** 2
            )
            facing_errors[facing_dir].append(float(dist) - true_dist)

        if bool(rec.get("wristRotationTrial")) and sample.get("azimuthDeg") is not None:
            az_deg = float(sample["azimuthDeg"])
            raw_azimuth_by_peer[(point_id, peer_hex)].append(az_deg)
            orient = rec.get("orientation", {})
            world_q = orient.get("worldQuaternion") if isinstance(orient, Mapping) else None
            if isinstance(world_q, Mapping):
                az_rad = math.radians(az_deg)
                local_vec = (math.cos(az_rad), math.sin(az_rad), 0.0)
                wx, wy, _ = _rotate_vector_by_quat(world_q, local_vec)
                world_azimuth_by_peer[(point_id, peer_hex)].append(math.degrees(math.atan2(wy, wx)))

    horiz_errors_m: list[float] = []
    epoch_seq = 0
    for (point_id, _facing, _bucket), bucket_records in sorted(grouped_buckets.items()):
        gt = ground_truth.get(point_id)
        if gt is None:
            continue
        by_anchor: dict[str, RangeMeasurement] = {}
        epoch_ts_ms = int(bucket_records[0]["sample"]["wallClockUnixMs"])
        measured_at = datetime.fromtimestamp(epoch_ts_ms / 1000.0, tz=UTC)
        for rec in bucket_records:
            sample = rec["sample"]
            peer_hex = str(sample["peerAddressHex"]).strip().upper()
            anchor_def = anchors.get(peer_hex)
            if anchor_def is None:
                continue
            by_anchor[anchor_def.anchor_id] = RangeMeasurement(
                tag_id="probe-watch",
                anchor_id=anchor_def.anchor_id,
                adapter_id="wear-uwb-probe",
                distance_m=float(sample["distanceM"]),
                variance_m2=variance_m2,
                measured_at=measured_at,
                technology=RangingTechnology.UWB,
                wear_state=WearState.ON_BODY,
                motion_state=MotionState.STILL,
                azimuth_deg=(
                    float(sample["azimuthDeg"]) if sample.get("azimuthDeg") is not None else None
                ),
                elevation_deg=(
                    float(sample["elevationDeg"])
                    if sample.get("elevationDeg") is not None
                    else None
                ),
                azimuth_reference_frame="device_local",
            )
        if len(by_anchor) < 3:
            continue
        epoch_seq += 1
        epoch = RangeEpoch(
            tag_id="probe-watch",
            adapter_id="wear-uwb-probe",
            epoch_seq=epoch_seq,
            measured_at=measured_at,
            measurements=tuple(by_anchor.values()),
            wear_state=WearState.ON_BODY,
            motion_state=MotionState.STILL,
        )
        solved = solver.solve(epoch, nodes)
        if solved is None:
            continue
        horiz_errors_m.append(math.hypot(solved.x_m - gt.x_m, solved.y_m - gt.y_m))

    total_expected = (
        expected_epochs if expected_epochs is not None else max(1, len(grouped_buckets))
    )
    availability = min(1.0, len(horiz_errors_m) / float(total_expected))

    if horiz_errors_m:
        arr = np.asarray(horiz_errors_m, dtype=np.float64)
        median_err: float | None = float(np.median(arr))
        p95_err: float | None = float(np.percentile(arr, 95))
        mean_err: float | None = float(np.mean(arr))
    else:
        median_err = None
        p95_err = None
        mean_err = None

    raw_az_stds = [
        std for vals in raw_azimuth_by_peer.values() if (std := _circular_std_deg(vals)) is not None
    ]
    world_az_stds = [
        std
        for vals in world_azimuth_by_peer.values()
        if (std := _circular_std_deg(vals)) is not None
    ]

    facing_stats = tuple(
        FacingDirectionStats(
            facing_dir=direction,
            sample_count=len(errs),
            mean_range_bias_m=float(np.mean(errs)),
            std_range_error_m=float(np.std(errs)),
        )
        for direction, errs in sorted(facing_errors.items())
        if errs
    )

    passed_med = median_err is not None and median_err <= P3_MAX_MEDIAN_ERROR_M
    passed_p95 = p95_err is not None and p95_err <= P3_MAX_P95_ERROR_M
    passed_avail = availability >= P3_MIN_AVAILABILITY

    return P3Report(
        expected_epochs=total_expected,
        solved_epochs=len(horiz_errors_m),
        availability_fraction=availability,
        median_horizontal_error_m=median_err,
        p95_horizontal_error_m=p95_err,
        mean_horizontal_error_m=mean_err,
        raw_azimuth_std_deg=float(np.mean(raw_az_stds)) if raw_az_stds else None,
        compensated_world_azimuth_std_deg=(
            float(np.mean(world_az_stds)) if world_az_stds else None
        ),
        facing_direction_stats=facing_stats,
        passed_median=passed_med,
        passed_p95=passed_p95,
        passed_availability=passed_avail,
        gate_passed=passed_med and passed_p95 and passed_avail,
    )


def analyze_p4(
    records: Sequence[Mapping[str, Any]],
    *,
    cell_capacity_mah: float = DEFAULT_WATCH_CAPACITY_MAH,
) -> P4Report:
    """Analyze Stage P4 coulomb-counter battery logs and evaluate Gate P4."""
    if not math.isfinite(cell_capacity_mah) or cell_capacity_mah <= 0.0:
        raise ValueError(f"cell_capacity_mah must be finite and > 0, got {cell_capacity_mah!r}")

    by_regime: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seated_bursts_by_regime: dict[str, int] = defaultdict(int)
    total_bursts_by_regime: dict[str, int] = defaultdict(int)

    for rec in records:
        rtype = rec.get("record_type")
        if rtype == "p4_battery_sample":
            bat = rec["battery"]
            by_regime[str(bat["regime"])].append(bat)
        elif rtype == "p4_fsm_transition" and rec.get("action") == "START_UWB_BURST":
            regime = str(rec["regime"])
            total_bursts_by_regime[regime] += 1
            if str(rec.get("activityLabel", "")).lower() == "seated":
                seated_bursts_by_regime[regime] += 1

    regime_stats: list[P4RegimeStats] = []
    stats_map: dict[str, P4RegimeStats] = {}
    for regime in sorted(by_regime.keys()):
        samples = sorted(by_regime[regime], key=lambda item: int(item["elapsedRealtimeNanos"]))
        if len(samples) < 2:
            duration_h = 0.0
            delta_uah = None
            drain_uah_h = None
            drain_pct_h = None
        else:
            first, last = samples[0], samples[-1]
            duration_h = (
                int(last["elapsedRealtimeNanos"]) - int(first["elapsedRealtimeNanos"])
            ) / 3_600_000_000_000.0
            uah_vals = [
                int(s["chargeCounterUah"]) for s in samples if s.get("chargeCounterUah") is not None
            ]
            delta_uah = max(0, uah_vals[0] - uah_vals[-1]) if len(uah_vals) >= 2 else None
            drain_uah_h = (
                delta_uah / duration_h if delta_uah is not None and duration_h > 0.0 else None
            )
            delta_pct = max(0, int(first["capacityPercent"]) - int(last["capacityPercent"]))
            drain_pct_h = delta_pct / duration_h if duration_h > 0.0 else None

        currents_ma = [
            abs(int(s["currentNowUa"])) / 1000.0
            for s in samples
            if s.get("currentNowUa") is not None
        ]
        peak_ma = max(currents_ma) if currents_ma else None

        thermals = [str(s.get("thermalStatus", "NONE")) for s in samples]
        max_thermal = max(thermals, key=lambda t: _THERMAL_ORDER.get(t, 0)) if thermals else "NONE"
        throttle_count = sum(1 for t in thermals if _THERMAL_ORDER.get(t, 0) >= 2)

        counter_bursts = max(
            (int(s.get("burstsTriggeredTotal", 0)) for s in samples),
            default=0,
        )
        bursts_count = max(counter_bursts, total_bursts_by_regime.get(regime, 0))
        bursts_per_h = bursts_count / duration_h if duration_h > 0.0 else None
        seated_false_per_h = (
            seated_bursts_by_regime.get(regime, 0) / duration_h if duration_h > 0.0 else None
        )

        if drain_uah_h is not None and drain_uah_h > 0.0:
            projected_h: float | None = (cell_capacity_mah * 1000.0) / drain_uah_h
        elif drain_pct_h is not None and drain_pct_h > 0.0:
            projected_h = 100.0 / drain_pct_h
        else:
            projected_h = None

        stat = P4RegimeStats(
            regime=regime,
            duration_hours=duration_h,
            delta_uah=delta_uah,
            drain_uah_per_hour=drain_uah_h,
            drain_percent_per_hour=drain_pct_h,
            peak_current_ma=peak_ma,
            max_thermal_status=max_thermal,
            thermal_throttle_events=throttle_count,
            bursts_per_hour=bursts_per_h,
            seated_false_triggers_per_hour=seated_false_per_h,
            projected_runtime_hours=projected_h,
        )
        regime_stats.append(stat)
        stats_map[regime] = stat

    reg_a = stats_map.get("A_BLE_CONTROL")
    reg_c = stats_map.get("C_MOTION_DUTY_CYCLED")

    added_pp_24h: float | None = None
    if reg_a is not None and reg_c is not None:
        if reg_a.drain_uah_per_hour is not None and reg_c.drain_uah_per_hour is not None:
            delta_uah_h = reg_c.drain_uah_per_hour - reg_a.drain_uah_per_hour
            added_pp_24h = (delta_uah_h * 24.0 / (cell_capacity_mah * 1000.0)) * 100.0
        elif reg_a.drain_percent_per_hour is not None and reg_c.drain_percent_per_hour is not None:
            added_pp_24h = (reg_c.drain_percent_per_hour - reg_a.drain_percent_per_hour) * 24.0

    passes_30h_or_added = False
    passes_36h = False
    zero_thermal = False
    passes_false_triggers = False
    if reg_c is not None:
        proj_ok = (
            reg_c.projected_runtime_hours is not None
            and reg_c.projected_runtime_hours >= P4_MIN_PROJECTED_HOURS
        )
        added_ok = added_pp_24h is not None and added_pp_24h <= P4_MAX_ADDED_DRAIN_PP_PER_24H
        passes_30h_or_added = proj_ok or added_ok
        passes_36h = (
            reg_c.projected_runtime_hours is not None
            and reg_c.projected_runtime_hours >= P4_WEARABLE_36H_HOURS
        )
        zero_thermal = reg_c.thermal_throttle_events == 0
        passes_false_triggers = (
            reg_c.seated_false_triggers_per_hour is not None
            and reg_c.seated_false_triggers_per_hour <= P4_MAX_SEATED_FALSE_TRIGGERS_PER_HOUR
        )

    return P4Report(
        cell_capacity_mah=cell_capacity_mah,
        regimes=tuple(regime_stats),
        regime_c_added_drain_pp_per_24h=added_pp_24h,
        regime_c_passes_30h_or_added_drain=passes_30h_or_added,
        regime_c_passes_36h_wearable_rule=passes_36h,
        regime_c_zero_thermal_throttling=zero_thermal,
        regime_c_passes_false_triggers=passes_false_triggers,
        gate_passed=passes_30h_or_added and zero_thermal and passes_false_triggers,
    )


def _fmt_opt(val: float | None, digits: int = 2) -> str:
    return f"{val:.{digits}f}" if val is not None else "n/a"


def format_markdown(stage: str, report: P1Report | P2Report | P3Report | P4Report) -> str:
    """Render a stage report as a GitHub-flavored Markdown summary table."""
    lines: list[str] = [f"### Stage {stage.upper()} Summary (gate_passed={report.gate_passed})", ""]
    if isinstance(report, P1Report):
        lines.append(
            "| Backend | Feature | Available | Dist | Az | El | Multicast | "
            "LocalAddr | Ch/Pre | Status | Gate |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in report.rows:
            ch_pre = (
                f"{r.assigned_channel}/{r.assigned_preamble_index}"
                if r.assigned_channel is not None
                else "n/a"
            )
            lines.append(
                f"| {r.backend_name} | {r.feature_flag_present} | {r.is_available} | "
                f"{r.supports_distance} | {r.supports_azimuth} | {r.supports_elevation} | "
                f"{r.supports_multicast_ds_twr} | {r.local_address_hex or 'n/a'} | {ch_pre} | "
                f"{r.raw_status_code if r.raw_status_code is not None else 'OK'} | "
                f"{r.passes_controller_gate} |"
            )
    elif isinstance(report, P2Report):
        lines.append(
            "| Strategy | Trials | OOB OK | OOB Med (ms) | First 3-Anchor Med (ms) | "
            "Min Top-3 Rate (Hz) | >=3 Anchor Epochs | Gate |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for s in report.strategies:
            lines.append(
                f"| {s.strategy} | {s.trials} | {s.oob_success_rate * 100:.1f}% | "
                f"{_fmt_opt(s.oob_latency_median_ms, 1)} | "
                f"{_fmt_opt(s.first_3_anchor_median_ms, 1)} | "
                f"{_fmt_opt(s.min_peer_rate_hz, 2)} | {s.three_anchor_epoch_share * 100:.1f}% | "
                f"{s.passes_gate} |"
            )
    elif isinstance(report, P3Report):
        lines.append("| Metric | Value | Gate |")
        lines.append("| --- | --- | --- |")
        lines.append(
            f"| Median horizontal error (m) | {_fmt_opt(report.median_horizontal_error_m, 3)} | "
            f"{report.passed_median} (<= {P3_MAX_MEDIAN_ERROR_M:.2f} m) |"
        )
        lines.append(
            f"| p95 horizontal error (m) | {_fmt_opt(report.p95_horizontal_error_m, 3)} | "
            f"{report.passed_p95} (<= {P3_MAX_P95_ERROR_M:.2f} m) |"
        )
        lines.append(
            f"| Fix availability | {report.availability_fraction * 100:.1f}% "
            f"({report.solved_epochs}/{report.expected_epochs}) | "
            f"{report.passed_availability} (>= {P3_MIN_AVAILABILITY * 100:.0f}%) |"
        )
        lines.append(
            f"| Wrist rotation raw azimuth std (deg) | {_fmt_opt(report.raw_azimuth_std_deg, 1)} "
            "| observability |"
        )
        lines.append(
            "| Quaternion-compensated world azimuth std (deg) | "
            f"{_fmt_opt(report.compensated_world_azimuth_std_deg, 1)} | observability |"
        )
    elif isinstance(report, P4Report):
        lines.append(
            "| Regime | Duration (h) | Drain (uAh/h) | Drain (%/h) | Peak (mA) | "
            "Thermal Throttles | Bursts/h | Seated False/h | Projected (h) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for reg in report.regimes:
            lines.append(
                f"| {reg.regime} | {reg.duration_hours:.2f} | "
                f"{_fmt_opt(reg.drain_uah_per_hour, 0)} | "
                f"{_fmt_opt(reg.drain_percent_per_hour, 2)} | "
                f"{_fmt_opt(reg.peak_current_ma, 1)} | {reg.thermal_throttle_events} | "
                f"{_fmt_opt(reg.bursts_per_hour, 2)} | "
                f"{_fmt_opt(reg.seated_false_triggers_per_hour, 2)} | "
                f"{_fmt_opt(reg.projected_runtime_hours, 1)} |"
            )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_probe_jsonl.py",
        description="Analyze pulled Wear OS UWB bench probe JSONL telemetry files.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown)",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    p1 = sub.add_parser("p1", help="Analyze Stage P1 capability audit JSONL")
    p1.add_argument("--input", type=Path, required=True)

    p2 = sub.add_parser("p2", help="Analyze Stage P2 OOB & multicast JSONL")
    p2.add_argument("--input", type=Path, required=True)
    p2.add_argument("--epoch-bucket-s", type=float, default=1.0)

    p3 = sub.add_parser("p3", help="Analyze Stage P3 accuracy & wrist azimuth JSONL")
    p3.add_argument("--input", type=Path, required=True)
    p3.add_argument("--anchors", type=Path, required=True)
    p3.add_argument("--ground-truth", type=Path, required=True)
    p3.add_argument("--epoch-window-s", type=float, default=0.25)
    p3.add_argument("--expected-epochs", type=int, default=None)
    p3.add_argument("--uwb-sigma-m", type=float, default=P3_DEFAULT_UWB_SIGMA_M)

    p4 = sub.add_parser("p4", help="Analyze Stage P4 battery & thermal JSONL")
    p4.add_argument("--input", type=Path, required=True)
    p4.add_argument("--cell-capacity-mah", type=float, default=DEFAULT_WATCH_CAPACITY_MAH)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning exit code 0 when the stage gate passes, 1 otherwise."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    records = load_jsonl(args.input)

    report: P1Report | P2Report | P3Report | P4Report
    if args.stage == "p1":
        report = analyze_p1(records)
    elif args.stage == "p2":
        report = analyze_p2(records, epoch_bucket_s=args.epoch_bucket_s)
    elif args.stage == "p3":
        anchors = load_anchors(args.anchors)
        gt = load_ground_truth_points(args.ground_truth)
        report = analyze_p3(
            records,
            anchors,
            gt,
            epoch_window_s=args.epoch_window_s,
            expected_epochs=args.expected_epochs,
            uwb_sigma_m=args.uwb_sigma_m,
        )
    elif args.stage == "p4":
        report = analyze_p4(records, cell_capacity_mah=args.cell_capacity_mah)
    else:
        return 2

    if args.format == "json":
        sys.stdout.write(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(format_markdown(args.stage, report))
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
