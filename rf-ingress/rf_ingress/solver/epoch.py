"""Per-tag epoch grouping and core room-level fallback for range reports.

Epochs are grouped strictly per ``(adapter_id, tag_id)`` over either an
adapter-supplied ``epoch_seq`` or a short sliding collection window
(``epoch_window_s``). No temporal smoothing is performed across epochs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cts_contracts.wearables import MotionState, WearState

from rf_ingress.domain import (
    RangeEpoch,
    RangeMeasurement,
    RfNodeGeometry,
    RoomReport,
    ensure_utc,
)


@dataclass
class _OpenEpoch:
    tag_id: str
    adapter_id: str
    epoch_seq: int
    window_start: datetime
    explicit_seq: int | None
    by_anchor: dict[str, RangeMeasurement] = field(default_factory=dict)

    def add(self, measurement: RangeMeasurement) -> None:
        existing = self.by_anchor.get(measurement.anchor_id)
        if existing is None or measurement.measured_at >= existing.measured_at:
            self.by_anchor[measurement.anchor_id] = measurement

    def freeze(self) -> RangeEpoch:
        ordered = tuple(sorted(self.by_anchor.values(), key=lambda m: (m.anchor_id, m.measured_at)))
        newest = max(ordered, key=lambda m: m.measured_at)
        wear_state = WearState.UNSPECIFIED
        motion_state = MotionState.UNSPECIFIED
        battery: float | None = None
        sent_at: datetime | None = None
        for m in sorted(ordered, key=lambda item: item.measured_at):
            if m.wear_state != WearState.UNSPECIFIED:
                wear_state = m.wear_state
            if m.motion_state != MotionState.UNSPECIFIED:
                motion_state = m.motion_state
            if m.battery_fraction is not None:
                battery = m.battery_fraction
            if m.sent_at is not None:
                sent_at = m.sent_at
        return RangeEpoch(
            tag_id=self.tag_id,
            adapter_id=self.adapter_id,
            epoch_seq=self.epoch_seq,
            measured_at=newest.measured_at,
            measurements=ordered,
            wear_state=wear_state,
            motion_state=motion_state,
            battery_fraction=battery,
            sent_at=sent_at,
        )


class EpochBuilder:
    """Groups range reports per tag into solver epochs.

    - When measurements carry an explicit ``epoch_seq``, all reports sharing
      ``(adapter_id, tag_id, epoch_seq)`` form one epoch; starting a new
      ``epoch_seq`` flushes the previous open epoch for that tag.
    - Otherwise reports within ``epoch_window_s`` of the bucket's earliest
      ``measured_at`` form one epoch; a report beyond the window closes and
      emits the prior epoch.
    - When all expected active nodes for the adapter have reported in the
      current epoch, the epoch closes immediately without waiting for the
      window to expire.
    - Duplicate reports from the same anchor within one epoch keep the latest.
    - Calling :meth:`on_site_config_version` with a changed version drops all
      in-flight epochs so no solve mixes nodes from two site-config revisions.
    """

    def __init__(self, *, epoch_window_s: float = 0.25) -> None:
        if epoch_window_s <= 0.0:
            raise ValueError("epoch_window_s must be > 0")
        self._epoch_window_s: float = epoch_window_s
        self._open_by_key: dict[tuple[str, str], _OpenEpoch] = {}
        self._next_seq_by_key: dict[tuple[str, str], int] = {}
        self._expected_nodes_by_adapter: dict[str, frozenset[str]] = {}
        self._site_config_version: str = ""

    def set_expected_nodes(self, adapter_id: str, anchor_ids: frozenset[str]) -> None:
        """Register the configured active node IDs for *adapter_id*."""
        self._expected_nodes_by_adapter[adapter_id] = anchor_ids

    def on_site_config_version(
        self,
        version: str,
        *,
        nodes_by_adapter: Mapping[str, frozenset[str]] | None = None,
    ) -> int:
        """Update expected nodes and drop open epochs if *version* changed.

        Returns the number of dropped in-flight epochs.
        """
        if nodes_by_adapter is not None:
            self._expected_nodes_by_adapter = dict(nodes_by_adapter)
        if not self._site_config_version:
            self._site_config_version = version
            return 0
        if version == self._site_config_version:
            return 0
        dropped = len(self._open_by_key)
        self._open_by_key.clear()
        self._site_config_version = version
        return dropped

    def _allocate_seq(self, key: tuple[str, str], explicit_seq: int | None) -> int:
        if explicit_seq is not None:
            self._next_seq_by_key[key] = max(self._next_seq_by_key.get(key, 0), explicit_seq + 1)
            return explicit_seq
        seq = self._next_seq_by_key.get(key, 1)
        self._next_seq_by_key[key] = seq + 1
        return seq

    def add(self, measurement: RangeMeasurement) -> list[RangeEpoch]:
        """Add *measurement* and return zero, one, or two completed epochs."""
        key = (measurement.adapter_id, measurement.tag_id)
        emitted: list[RangeEpoch] = []
        current = self._open_by_key.get(key)

        if current is not None:
            should_flush = False
            if measurement.epoch_seq is not None or current.explicit_seq is not None:
                if measurement.epoch_seq != current.explicit_seq:
                    should_flush = True
            else:
                elapsed_s = (measurement.measured_at - current.window_start).total_seconds()
                if elapsed_s >= self._epoch_window_s or elapsed_s < -self._epoch_window_s:
                    should_flush = True
            if should_flush:
                emitted.append(current.freeze())
                del self._open_by_key[key]
                current = None

        if current is None:
            seq = self._allocate_seq(key, measurement.epoch_seq)
            current = _OpenEpoch(
                tag_id=measurement.tag_id,
                adapter_id=measurement.adapter_id,
                epoch_seq=seq,
                window_start=measurement.measured_at,
                explicit_seq=measurement.epoch_seq,
            )
            self._open_by_key[key] = current

        current.add(measurement)

        expected = self._expected_nodes_by_adapter.get(measurement.adapter_id)
        if expected and len(expected) >= 1 and expected.issubset(current.by_anchor.keys()):
            emitted.append(current.freeze())
            del self._open_by_key[key]

        return emitted

    def flush_expired(self, now: datetime) -> list[RangeEpoch]:
        """Emit all open epochs whose collection window has elapsed by *now*."""
        now_utc = ensure_utc(now)
        cutoff = now_utc - timedelta(seconds=self._epoch_window_s)
        emitted: list[RangeEpoch] = []
        expired_keys = [
            key for key, bucket in self._open_by_key.items() if bucket.window_start <= cutoff
        ]
        for key in expired_keys:
            emitted.append(self._open_by_key.pop(key).freeze())
        return emitted

    def flush_all(self) -> list[RangeEpoch]:
        """Emit all open epochs regardless of window expiration (e.g. on shutdown)."""
        emitted = [bucket.freeze() for bucket in self._open_by_key.values()]
        self._open_by_key.clear()
        return emitted

    def open_epoch_count(self) -> int:
        """Return the number of currently open in-flight epochs."""
        return len(self._open_by_key)


def room_fallback_from_epoch(
    epoch: RangeEpoch,
    nodes: Mapping[str, RfNodeGeometry],
) -> RoomReport | None:
    """Build a room-level fallback report when an epoch has too few nodes to solve.

    Selects the known RF node with the smallest measured distance whose site
    config declares a ``room_id``. Computes ``probability`` from the
    standardized distance margin to the nearest competing node in another room.
    """
    candidates: list[tuple[RangeMeasurement, RfNodeGeometry]] = []
    for measurement in epoch.measurements:
        node = nodes.get(measurement.anchor_id)
        if node is not None and node.room_id:
            candidates.append((measurement, node))
    if not candidates:
        return None

    candidates.sort(key=lambda pair: (pair[0].distance_m, pair[0].variance_m2, pair[0].anchor_id))
    best_meas, best_node = candidates[0]
    assert best_node.room_id is not None

    other_room = [
        (m, n)
        for m, n in candidates[1:]
        if n.room_id is not None and n.room_id != best_node.room_id
    ]
    if other_room:
        runner_up_meas, _ = other_room[0]
        combined_sigma = math.sqrt(best_meas.variance_m2 + runner_up_meas.variance_m2)
        margin_sigma = max(0.0, runner_up_meas.distance_m - best_meas.distance_m) / max(
            combined_sigma, 1e-6
        )
        probability = 0.5 + 0.45 * math.tanh(0.75 * margin_sigma)
    elif len(candidates) >= 2:
        # Multiple nodes in the same room heard the tag with no competing room.
        probability = 0.85
    else:
        # Single-node observation: moderate confidence bounded away from certainty.
        probability = 0.55 + 0.25 / (1.0 + best_meas.distance_m / 3.0)

    clamped_prob = max(0.05, min(0.99, float(probability)))
    return RoomReport(
        tag_id=epoch.tag_id,
        measured_at=epoch.measured_at,
        room_id=best_node.room_id,
        probability=clamped_prob,
        wear_state=epoch.wear_state,
        motion_state=epoch.motion_state,
        battery_fraction=epoch.battery_fraction,
        adapter_id=epoch.adapter_id,
        epoch_seq=epoch.epoch_seq,
        sent_at=epoch.sent_at,
        anchor_id=best_node.anchor_id,
        distance_m=best_meas.distance_m,
        anchors_used=len(candidates),
    )
