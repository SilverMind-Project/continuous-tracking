"""Unit tests for pure BLE RSSI variability and telemetry motion inference."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from cts_contracts.wearables import MotionState

from rf_ingress.adapters.ble_motion import BleMotionConfig, BleMotionInferer

_T0 = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)


def test_constant_rssi_with_1db_noise_yields_still_after_120s() -> None:
    rng = random.Random(20261121)
    inferer = BleMotionInferer(
        BleMotionConfig(
            moving_window_s=30.0,
            still_duration_s=120.0,
            motion_rssi_std_db=3.0,
            max_gap_s=15.0,
        )
    )
    last_state = MotionState.UNSPECIFIED
    states_before_120: list[MotionState] = []
    states_at_or_after_120: list[MotionState] = []

    for sec in range(0, 130, 1):
        ts = _T0 + timedelta(seconds=sec)
        for node_id, base_rssi in (("node-lr", -65.0), ("node-kit", -72.0), ("node-hall", -78.0)):
            noisy = base_rssi + rng.gauss(0.0, 0.8)
            last_state = inferer.observe(
                tag_id="tag-senior",
                node_id=node_id,
                rssi_dbm=noisy,
                observed_at=ts,
            )
        if sec < 120:
            states_before_120.append(last_state)
        else:
            states_at_or_after_120.append(last_state)

    assert all(s == MotionState.UNSPECIFIED for s in states_before_120)
    assert states_at_or_after_120
    assert all(s == MotionState.STILL for s in states_at_or_after_120)


def test_6db_swings_on_two_nodes_yield_moving() -> None:
    inferer = BleMotionInferer(BleMotionConfig(motion_rssi_std_db=3.0))
    state = MotionState.UNSPECIFIED

    for sec in range(10):
        ts = _T0 + timedelta(seconds=sec)
        swing = 6.0 if sec % 2 == 0 else -6.0
        inferer.observe(
            tag_id="tag-senior",
            node_id="node-lr",
            rssi_dbm=-66.0 + swing,
            observed_at=ts,
        )
        state = inferer.observe(
            tag_id="tag-senior",
            node_id="node-kit",
            rssi_dbm=-71.0 - swing,
            observed_at=ts,
        )

    assert state == MotionState.MOVING


def test_single_node_swing_does_not_declare_moving_or_still() -> None:
    inferer = BleMotionInferer(BleMotionConfig(motion_rssi_std_db=3.0, min_nodes_moving=2))
    state = MotionState.UNSPECIFIED

    for sec in range(130):
        ts = _T0 + timedelta(seconds=sec)
        swing = 6.0 if sec % 2 == 0 else -6.0
        inferer.observe(
            tag_id="tag-senior",
            node_id="node-lr",
            rssi_dbm=-66.0 + swing,
            observed_at=ts,
        )
        state = inferer.observe(
            tag_id="tag-senior",
            node_id="node-kit",
            rssi_dbm=-72.0,
            observed_at=ts,
        )

    assert state == MotionState.UNSPECIFIED


def test_observation_gap_yields_unspecified_never_still() -> None:
    inferer = BleMotionInferer(
        BleMotionConfig(
            moving_window_s=30.0,
            still_duration_s=120.0,
            motion_rssi_std_db=3.0,
            max_gap_s=15.0,
        )
    )
    # Reach STILL first over 125 seconds.
    state = MotionState.UNSPECIFIED
    for sec in range(0, 126, 2):
        ts = _T0 + timedelta(seconds=sec)
        for node_id in ("node-lr", "node-kit"):
            state = inferer.observe(
                tag_id="tag-senior",
                node_id=node_id,
                rssi_dbm=-68.0,
                observed_at=ts,
            )
    assert state == MotionState.STILL

    # Evaluating after a gap > max_gap_s must return UNSPECIFIED.
    assert inferer.evaluate("tag-senior", at=_T0 + timedelta(seconds=150)) == (
        MotionState.UNSPECIFIED
    )

    # New observations after the gap must restart the 120 s calm timer.
    resumed = inferer.observe(
        tag_id="tag-senior",
        node_id="node-lr",
        rssi_dbm=-68.0,
        observed_at=_T0 + timedelta(seconds=160),
    )
    assert resumed == MotionState.UNSPECIFIED


def test_accelerometer_telemetry_motion_overrides_calm_rssi() -> None:
    inferer = BleMotionInferer()
    for sec in range(0, 125, 2):
        ts = _T0 + timedelta(seconds=sec)
        inferer.observe(tag_id="tag-senior", node_id="node-lr", rssi_dbm=-64.0, observed_at=ts)

    assert inferer.evaluate("tag-senior", at=_T0 + timedelta(seconds=125)) == MotionState.STILL

    moving = inferer.observe(
        tag_id="tag-senior",
        node_id="node-lr",
        rssi_dbm=-64.0,
        observed_at=_T0 + timedelta(seconds=126),
        telemetry_moving=True,
    )
    assert moving == MotionState.MOVING
