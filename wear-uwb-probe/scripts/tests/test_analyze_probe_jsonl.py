"""Unit tests for ``analyze_probe_jsonl.py`` on synthetic Stage P1-P4 JSONL fixtures."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_probe_jsonl import (  # noqa: E402
    AnchorDefinition,
    GroundTruthPoint,
    analyze_p1,
    analyze_p2,
    analyze_p3,
    analyze_p4,
    main,
)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _yaw_quat(deg: float) -> dict[str, float]:
    half = math.radians(deg) / 2.0
    return {"w": math.cos(half), "x": 0.0, "y": 0.0, "z": math.sin(half)}


def test_stage_p1_passes_when_gms_opens_controller_even_if_jetpack_blocked() -> None:
    records = [
        {
            "record_type": "p1_capability",
            "wallClockUnixMs": 1_700_000_000_000,
            "elapsedRealtimeNanos": 1_000_000_000,
            "screenOffForegroundService": True,
            "capabilities": {
                "backendName": "gms_nearby_uwb",
                "featureFlagPresent": True,
                "isAvailable": True,
                "supportsDistance": True,
                "supportsAzimuth": True,
                "supportsElevation": False,
                "supportsMulticastDsTwr": True,
                "localAddressHex": "AA:BB",
                "assignedChannel": 9,
                "assignedPreambleIndex": 10,
                "rawStatusCode": None,
                "errorDetail": None,
            },
        },
        {
            "record_type": "p1_capability",
            "wallClockUnixMs": 1_700_000_001_000,
            "elapsedRealtimeNanos": 2_000_000_000,
            "screenOffForegroundService": True,
            "capabilities": {
                "backendName": "jetpack_uwb",
                "featureFlagPresent": True,
                "isAvailable": False,
                "supportsDistance": False,
                "supportsAzimuth": False,
                "supportsElevation": False,
                "supportsMulticastDsTwr": False,
                "rawStatusCode": 42000,
                "errorDetail": "SERVICE_NOT_ENABLED",
            },
        },
    ]

    report = analyze_p1(records)
    assert report.gate_passed is True
    assert report.winning_backends == ("gms_nearby_uwb",)

    # If both GMS and Jetpack fail with 42000, Gate P1 fails cleanly.
    blocked = [records[1]]
    blocked_report = analyze_p1(blocked)
    assert blocked_report.gate_passed is False
    assert blocked_report.winning_backends == ()


def test_stage_p2_compares_preshared_sts_vs_ble_gatt_and_checks_gate() -> None:
    records: list[dict[str, Any]] = []
    # Trial 1: Pre-Shared STS broadcast (fast OOB 210 ms, 10 epochs at 1 Hz across 4 anchors).
    records.append(
        {
            "record_type": "p2_oob_trial",
            "wallClockUnixMs": 1_700_000_000_000,
            "elapsedRealtimeNanos": 0,
            "trialIndex": 1,
            "result": {
                "strategy": "PRESHARED_STS_BROADCAST",
                "succeeded": True,
                "setupLatencyMs": 210,
                "peersNegotiated": ["00:01", "00:02", "00:03", "00:04"],
                "peersFailed": [],
            },
        }
    )
    for sec in range(10):
        for idx, addr in enumerate(["00:01", "00:02", "00:03", "00:04"]):
            records.append(
                {
                    "record_type": "p2_ranging_sample",
                    "wallClockUnixMs": 1_700_000_000_000 + sec * 1000 + idx * 10,
                    "elapsedRealtimeNanos": sec * 1_000_000_000 + idx * 10_000_000,
                    "trialIndex": 1,
                    "sample": {
                        "backendName": "gms_nearby_uwb",
                        "elapsedRealtimeNanos": sec * 1_000_000_000 + idx * 10_000_000,
                        "wallClockUnixMs": 1_700_000_000_000 + sec * 1000 + idx * 10,
                        "peerAddressHex": addr,
                        "distanceM": 2.4,
                    },
                }
            )
    records.append(
        {
            "record_type": "p2_session_summary",
            "wallClockUnixMs": 1_700_000_010_000,
            "elapsedRealtimeNanos": 10_000_000_000,
            "trialIndex": 1,
            "summary": {
                "backendName": "gms_nearby_uwb",
                "sessionId": 1,
                "startedAtUnixMs": 1_700_000_000_000,
                "endedAtUnixMs": 1_700_000_010_000,
                "samplesCollected": 40,
                "distinctPeersAcquired": 4,
                "timeToFirstRangeMs": 240,
                "timeToFirstThreeAnchorEpochMs": 320,
                "effectiveRateHzByPeer": {"00:01": 1.0, "00:02": 1.0, "00:03": 1.0, "00:04": 1.0},
            },
        }
    )

    # Trial 2: BLE GATT sequential baseline (slow 2800 ms -> fails 2000 ms gate).
    records.append(
        {
            "record_type": "p2_oob_trial",
            "wallClockUnixMs": 1_700_000_020_000,
            "elapsedRealtimeNanos": 20_000_000_000,
            "trialIndex": 2,
            "result": {
                "strategy": "BLE_GATT_SEQUENTIAL",
                "succeeded": True,
                "setupLatencyMs": 2800,
                "peersNegotiated": ["00:01", "00:02", "00:03", "00:04"],
                "peersFailed": [],
            },
        }
    )
    records.append(
        {
            "record_type": "p2_session_summary",
            "wallClockUnixMs": 1_700_000_030_000,
            "elapsedRealtimeNanos": 30_000_000_000,
            "trialIndex": 2,
            "summary": {
                "backendName": "gms_nearby_uwb",
                "sessionId": 2,
                "startedAtUnixMs": 1_700_000_020_000,
                "endedAtUnixMs": 1_700_000_030_000,
                "samplesCollected": 20,
                "distinctPeersAcquired": 4,
                "timeToFirstRangeMs": 2950,
                "timeToFirstThreeAnchorEpochMs": 3100,
                "effectiveRateHzByPeer": {"00:01": 0.5, "00:02": 0.5, "00:03": 0.5, "00:04": 0.5},
            },
        }
    )

    report = analyze_p2(records)
    assert report.gate_passed is True
    by_name = {s.strategy: s for s in report.strategies}
    assert by_name["PRESHARED_STS_BROADCAST"].passes_gate is True
    assert by_name["PRESHARED_STS_BROADCAST"].oob_latency_median_ms == pytest.approx(210.0)
    assert by_name["BLE_GATT_SEQUENTIAL"].passes_gate is False


def test_stage_p3_solves_12_points_via_multilateration_and_decouples_wrist_azimuth() -> None:
    rng = np.random.default_rng(42)
    anchors = {
        "00:01": AnchorDefinition("anchor-1", "00:01", 0.0, 0.0, 1.2),
        "00:02": AnchorDefinition("anchor-2", "00:02", 6.0, 0.0, 2.3),
        "00:03": AnchorDefinition("anchor-3", "00:03", 6.0, 5.0, 1.4),
        "00:04": AnchorDefinition("anchor-4", "00:04", 0.0, 5.0, 2.1),
    }
    ground_truth = {
        f"P{i:02d}": GroundTruthPoint(
            point_id=f"P{i:02d}",
            x_m=1.2 + (i % 4) * 1.1,
            y_m=1.0 + (i // 4) * 1.4,
            z_m=1.05,
        )
        for i in range(12)
    }

    records: list[dict[str, Any]] = []
    t_ns = 10_000_000_000
    base_unix_ms = 1_700_000_000_000

    for idx_pt, (pid, gt) in enumerate(ground_truth.items()):
        for facing in ("N", "E", "S", "W"):
            # Simulate mild torso NLOS bias (+0.08 m) when facing South.
            shadow_bias_m = 0.08 if facing == "S" else 0.0
            for epoch_i in range(4):
                wrist_yaw_deg = float(epoch_i * 30.0) if (idx_pt == 0 and facing == "N") else 0.0
                world_bearing_deg = 40.0
                # Watch local azimuth shifts oppositely as the wrist rotates in world frame.
                local_az_deg = world_bearing_deg - wrist_yaw_deg
                world_q = _yaw_quat(wrist_yaw_deg)

                for addr, anchor in anchors.items():
                    true_d = math.sqrt(
                        (gt.x_m - anchor.x_m) ** 2
                        + (gt.y_m - anchor.y_m) ** 2
                        + (gt.z_m - anchor.z_m) ** 2
                    )
                    noisy_d = max(0.2, true_d + shadow_bias_m + float(rng.normal(0.0, 0.06)))
                    records.append(
                        {
                            "record_type": "p3_ranging_orientation",
                            "wallClockUnixMs": base_unix_ms + t_ns // 1_000_000,
                            "elapsedRealtimeNanos": t_ns,
                            "pointId": pid,
                            "facingDir": facing,
                            "wristRotationTrial": idx_pt == 0 and facing == "N",
                            "sample": {
                                "backendName": "gms_nearby_uwb",
                                "elapsedRealtimeNanos": t_ns,
                                "wallClockUnixMs": base_unix_ms + t_ns // 1_000_000,
                                "peerAddressHex": addr,
                                "distanceM": noisy_d,
                                "azimuthDeg": local_az_deg,
                                "elevationDeg": 5.0,
                            },
                            "orientation": {
                                "targetElapsedRealtimeNanos": t_ns,
                                "worldQuaternion": world_q,
                                "extrapolated": False,
                            },
                        }
                    )
                t_ns += 1_000_000_000

    report = analyze_p3(records, anchors, ground_truth, expected_epochs=12 * 4 * 4)
    assert report.gate_passed is True
    assert report.median_horizontal_error_m is not None
    assert report.median_horizontal_error_m <= 0.30
    assert report.p95_horizontal_error_m is not None
    assert report.p95_horizontal_error_m <= 1.00
    assert report.availability_fraction == pytest.approx(1.0)
    # Wrist rotation creates large raw azimuth spread (> 25 deg), while quaternion-compensated
    # world bearing collapses near 0 deg.
    assert report.raw_azimuth_std_deg is not None and report.raw_azimuth_std_deg > 25.0
    assert (
        report.compensated_world_azimuth_std_deg is not None
        and report.compensated_world_azimuth_std_deg < 1.0
    )
    facing_map = {f.facing_dir: f for f in report.facing_direction_stats}
    assert facing_map["S"].mean_range_bias_m > facing_map["N"].mean_range_bias_m + 0.04


def test_stage_p4_evaluates_coulomb_counter_regimes_and_cli(tmp_path: Path) -> None:
    hour_ns = 3_600_000_000_000
    records = [
        # Regime A (2 h BLE control): 20,000 uAh drain -> 10,000 uAh/h (39 h projected).
        {
            "record_type": "p4_battery_sample",
            "wallClockUnixMs": 1_000,
            "elapsedRealtimeNanos": 0,
            "battery": {
                "elapsedRealtimeNanos": 0,
                "wallClockUnixMs": 1_000,
                "regime": "A_BLE_CONTROL",
                "eventTag": "regime_start",
                "capacityPercent": 100,
                "chargeCounterUah": 390_000,
                "currentNowUa": -10_000,
                "thermalStatus": "NONE",
            },
        },
        {
            "record_type": "p4_battery_sample",
            "wallClockUnixMs": 7_201_000,
            "elapsedRealtimeNanos": 2 * hour_ns,
            "battery": {
                "elapsedRealtimeNanos": 2 * hour_ns,
                "wallClockUnixMs": 7_201_000,
                "regime": "A_BLE_CONTROL",
                "eventTag": "regime_end",
                "capacityPercent": 95,
                "chargeCounterUah": 370_000,
                "currentNowUa": -11_000,
                "thermalStatus": "NONE",
            },
        },
        # Regime C (4 h motion-duty-cycled UWB): 48,000 uAh -> 12,000 uAh/h -> 32.5 h projected
        # and added drain over A = (2000 * 24 / 390000)*100 = 12.31 pp / 24 h (<= 15 pp).
        {
            "record_type": "p4_battery_sample",
            "wallClockUnixMs": 10_000_000,
            "elapsedRealtimeNanos": 3 * hour_ns,
            "battery": {
                "elapsedRealtimeNanos": 3 * hour_ns,
                "wallClockUnixMs": 10_000_000,
                "regime": "C_MOTION_DUTY_CYCLED",
                "eventTag": "regime_start",
                "capacityPercent": 100,
                "chargeCounterUah": 390_000,
                "currentNowUa": -12_000,
                "thermalStatus": "NONE",
                "burstsTriggeredTotal": 0,
            },
        },
        {
            "record_type": "p4_fsm_transition",
            "wallClockUnixMs": 12_000_000,
            "elapsedRealtimeNanos": 4 * hour_ns,
            "regime": "C_MOTION_DUTY_CYCLED",
            "fromState": "LOCOMOTION_CANDIDATE",
            "toState": "UWB_RANGING_BURST",
            "action": "START_UWB_BURST",
            "activityLabel": "seated",
        },
        {
            "record_type": "p4_battery_sample",
            "wallClockUnixMs": 24_400_000,
            "elapsedRealtimeNanos": 7 * hour_ns,
            "battery": {
                "elapsedRealtimeNanos": 7 * hour_ns,
                "wallClockUnixMs": 24_400_000,
                "regime": "C_MOTION_DUTY_CYCLED",
                "eventTag": "regime_end",
                "capacityPercent": 88,
                "chargeCounterUah": 342_000,
                "currentNowUa": -92_000,
                "thermalStatus": "LIGHT",
                "burstsTriggeredTotal": 16,
            },
        },
    ]

    report = analyze_p4(records, cell_capacity_mah=390.0)
    assert report.gate_passed is True
    assert report.regime_c_passes_30h_or_added_drain is True
    assert report.regime_c_passes_36h_wearable_rule is False
    assert report.regime_c_zero_thermal_throttling is True
    assert report.regime_c_passes_false_triggers is True
    assert report.regime_c_added_drain_pp_per_24h == pytest.approx(12.3077, abs=1e-2)

    jsonl_path = _write_jsonl(tmp_path / "p4_battery.jsonl", records)
    assert main(["--format", "json", "p4", "--input", str(jsonl_path)]) == 0
    assert main(["--format", "markdown", "p4", "--input", str(jsonl_path)]) == 0
