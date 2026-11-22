"""Pure motion-state inference for BLE beacons from RSSI variability and telemetry.

Classifies a beacon's radio signal into :class:`cts_contracts.wearables.MotionState`:

- ``MOVING`` when per-node RSSI standard deviation over the trailing window
  exceeds ``motion_rssi_std_db`` on at least ``min_nodes_moving`` receivers,
  or when beacon accelerometer telemetry reports motion.
- ``STILL`` only after continuous observations (no gap exceeding ``max_gap_s``)
  for at least ``still_duration_s`` with every evaluated receiver at or below
  ``motion_rssi_std_db`` and no active motion telemetry.
- ``UNSPECIFIED`` otherwise, including after any observation gap (a gap never
  produces ``STILL``).

This module evaluates radio-signal variability only and performs no smoothing
of solved positions.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cts_contracts.wearables import MotionState

from rf_ingress.domain import ensure_utc


@dataclass(frozen=True)
class BleMotionConfig:
    """Thresholds for BLE RSSI variability and telemetry motion inference."""

    moving_window_s: float = 30.0
    still_duration_s: float = 120.0
    motion_rssi_std_db: float = 3.0
    min_nodes_moving: int = 2
    min_samples_per_node: int = 3
    max_gap_s: float = 15.0

    def __post_init__(self) -> None:
        if self.moving_window_s <= 0.0:
            raise ValueError("moving_window_s must be > 0")
        if self.still_duration_s <= 0.0:
            raise ValueError("still_duration_s must be > 0")
        if self.motion_rssi_std_db <= 0.0:
            raise ValueError("motion_rssi_std_db must be > 0")
        if self.min_nodes_moving < 1:
            raise ValueError("min_nodes_moving must be >= 1")
        if self.min_samples_per_node < 2:
            raise ValueError("min_samples_per_node must be >= 2")
        if self.max_gap_s <= 0.0:
            raise ValueError("max_gap_s must be > 0")


@dataclass
class _TagMotionState:
    by_node: dict[str, deque[tuple[datetime, float]]] = field(default_factory=dict)
    last_observed_at: datetime | None = None
    calm_since_at: datetime | None = None
    last_telemetry_moving_at: datetime | None = None

    def reset_on_gap(self) -> None:
        self.by_node.clear()
        self.calm_since_at = None
        self.last_telemetry_moving_at = None


class BleMotionInferer:
    """Stateful per-tag BLE motion classifier over raw RSSI hearings."""

    def __init__(self, config: BleMotionConfig | None = None) -> None:
        self._cfg: BleMotionConfig = config or BleMotionConfig()
        self._tags: dict[str, _TagMotionState] = {}

    def observe(
        self,
        *,
        tag_id: str,
        node_id: str,
        rssi_dbm: float,
        observed_at: datetime,
        telemetry_moving: bool | None = None,
    ) -> MotionState:
        """Record one receiver hearing for *tag_id* and return its current motion state."""
        ts = ensure_utc(observed_at)
        if not math.isfinite(rssi_dbm):
            raise ValueError(f"rssi_dbm must be finite, got {rssi_dbm!r}")

        state = self._tags.setdefault(tag_id, _TagMotionState())

        if state.last_observed_at is not None:
            gap_s = (ts - state.last_observed_at).total_seconds()
            if gap_s > self._cfg.max_gap_s:
                state.reset_on_gap()
            elif gap_s < -self._cfg.moving_window_s:
                return self.evaluate(tag_id, at=state.last_observed_at)

        if state.last_observed_at is None or ts > state.last_observed_at:
            state.last_observed_at = ts

        samples = state.by_node.setdefault(node_id, deque())
        samples.append((ts, rssi_dbm))

        if telemetry_moving is True:
            state.last_telemetry_moving_at = ts
            state.calm_since_at = None
        elif telemetry_moving is False:
            state.last_telemetry_moving_at = None

        return self._evaluate_state(state, at=ts)

    def evaluate(self, tag_id: str, *, at: datetime) -> MotionState:
        """Evaluate the current motion state for *tag_id* at timestamp *at*."""
        ts = ensure_utc(at)
        state = self._tags.get(tag_id)
        if state is None or state.last_observed_at is None:
            return MotionState.UNSPECIFIED
        if (ts - state.last_observed_at).total_seconds() > self._cfg.max_gap_s:
            state.reset_on_gap()
            return MotionState.UNSPECIFIED
        return self._evaluate_state(state, at=ts)

    def _evaluate_state(self, state: _TagMotionState, *, at: datetime) -> MotionState:
        cutoff = at - timedelta(seconds=self._cfg.moving_window_s)

        if state.last_telemetry_moving_at is not None:
            if state.last_telemetry_moving_at >= cutoff:
                state.calm_since_at = None
                return MotionState.MOVING
            state.last_telemetry_moving_at = None

        moving_nodes = 0
        evaluated_nodes = 0
        for node_id in list(state.by_node.keys()):
            samples = state.by_node[node_id]
            while samples and samples[0][0] < cutoff:
                samples.popleft()
            if not samples:
                del state.by_node[node_id]
                continue
            if len(samples) < self._cfg.min_samples_per_node:
                continue
            evaluated_nodes += 1
            std_db = statistics.stdev(rssi for _, rssi in samples)
            if std_db > self._cfg.motion_rssi_std_db:
                moving_nodes += 1

        if moving_nodes >= self._cfg.min_nodes_moving:
            state.calm_since_at = None
            return MotionState.MOVING

        if moving_nodes > 0:
            # At least one receiver shows elevated RSSI swing: not calm, but below
            # the multi-node moving quorum.
            state.calm_since_at = None
            return MotionState.UNSPECIFIED

        if state.calm_since_at is None:
            state.calm_since_at = at

        if evaluated_nodes == 0:
            return MotionState.UNSPECIFIED

        calm_duration_s = (at - state.calm_since_at).total_seconds()
        if calm_duration_s >= self._cfg.still_duration_s:
            return MotionState.STILL

        return MotionState.UNSPECIFIED
