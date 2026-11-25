"""Per-tag epoch grouping and core room-level fallback for range reports.

Epochs are grouped strictly per ``(adapter_id, tag_id)`` over either an
adapter-supplied ``epoch_seq`` or a short sliding collection window
(``epoch_window_s``). No temporal smoothing is performed across epochs.

Grouping uses device-derived ``measured_at`` timestamps, but expiry runs on
the server clock: a burst-batched sender delivers epochs seconds after they
were captured, so an open epoch is closed only after it has been idle (no new
measurement arrived) for ``epoch_window_s`` of server time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
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
    opened_at: datetime
    last_arrival_at: datetime
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
    - An epoch closes immediately when every expected node of the adapter has
      reported, or, for unbatched senders (no ``sent_at``), when every node
      registered in one room (at least 3) has reported and no node outside that
      room has been heard.
    - :meth:`flush_expired` closes epochs idle for ``epoch_window_s`` (or open
      longer than ``max_open_s``) on the server clock passed as ``now``.
    - A late measurement repeating the explicit ``epoch_seq`` of an epoch that
      closed for the same tag within ``straggler_horizon_s`` is dropped instead
      of reopening a duplicate epoch. Sequence counters may restart (device or
      adapter restart), so only an exact repeat of the last closed sequence is
      treated as a straggler, never ``seq <= last_closed``.
    - Duplicate reports from the same anchor within one epoch keep the latest.
    - Calling :meth:`on_site_config_version` with a changed version drops all
      in-flight epochs so no solve mixes nodes from two site-config revisions.
    """

    def __init__(
        self,
        *,
        epoch_window_s: float = 0.25,
        max_open_s: float | None = None,
        straggler_horizon_s: float | None = None,
        on_late_measurement: Callable[[], None] | None = None,
    ) -> None:
        if not math.isfinite(epoch_window_s) or epoch_window_s <= 0.0:
            raise ValueError(f"epoch_window_s must be finite and > 0, got {epoch_window_s!r}")
        self._epoch_window_s: float = epoch_window_s
        self._max_open_s: float = (
            max_open_s if max_open_s is not None else max(2.0, 8.0 * epoch_window_s)
        )
        if not math.isfinite(self._max_open_s) or self._max_open_s < epoch_window_s:
            raise ValueError("max_open_s must be finite and >= epoch_window_s")
        self._straggler_horizon_s: float = (
            straggler_horizon_s
            if straggler_horizon_s is not None
            else max(1.0, 4.0 * epoch_window_s)
        )
        if not math.isfinite(self._straggler_horizon_s) or self._straggler_horizon_s < 0.0:
            raise ValueError("straggler_horizon_s must be finite and >= 0")
        self._on_late_measurement: Callable[[], None] | None = on_late_measurement
        self._open_by_key: dict[tuple[str, str], _OpenEpoch] = {}
        self._next_seq_by_key: dict[tuple[str, str], int] = {}
        self._last_closed_by_key: dict[tuple[str, str], tuple[int, datetime]] = {}
        self._expected_nodes_by_adapter: dict[str, frozenset[str]] = {}
        self._room_nodes_by_adapter: dict[str, dict[str, frozenset[str]]] = {}
        self._anchor_room_by_adapter: dict[str, dict[str, str]] = {}
        self._site_config_version: str = ""

    def set_expected_nodes(self, adapter_id: str, anchor_ids: frozenset[str]) -> None:
        """Register the configured active node IDs for *adapter_id* (no room grouping)."""
        self._expected_nodes_by_adapter[adapter_id] = anchor_ids
        self._room_nodes_by_adapter.pop(adapter_id, None)
        self._anchor_room_by_adapter.pop(adapter_id, None)

    def set_expected_nodes_with_rooms(
        self,
        adapter_id: str,
        nodes_to_room: Mapping[str, str | None],
    ) -> None:
        """Register expected node IDs for *adapter_id* and their optional room grouping."""
        self._expected_nodes_by_adapter[adapter_id] = frozenset(nodes_to_room.keys())
        by_room: dict[str, set[str]] = {}
        anchor_room: dict[str, str] = {}
        for anchor_id, room_id in nodes_to_room.items():
            if room_id:
                by_room.setdefault(room_id, set()).add(anchor_id)
                anchor_room[anchor_id] = room_id
        self._room_nodes_by_adapter[adapter_id] = {
            rid: frozenset(ids) for rid, ids in by_room.items()
        }
        self._anchor_room_by_adapter[adapter_id] = anchor_room

    def on_site_config_version(
        self,
        version: str,
        *,
        nodes_by_adapter: Mapping[str, frozenset[str]] | None = None,
        node_rooms_by_adapter: Mapping[str, Mapping[str, str | None]] | None = None,
    ) -> int:
        """Update expected nodes and drop open epochs if *version* changed.

        *node_rooms_by_adapter* (``adapter_id -> {anchor_id: room_id}``) replaces
        both the expected-node map and the per-room grouping in one step, so
        room-scoped completion never uses a stale floor-plan revision. When only
        *nodes_by_adapter* is given, room grouping is cleared.

        Returns the number of dropped in-flight epochs.
        """
        if node_rooms_by_adapter is not None:
            self._expected_nodes_by_adapter = {}
            self._room_nodes_by_adapter = {}
            self._anchor_room_by_adapter = {}
            for adapter_id, nodes_to_room in node_rooms_by_adapter.items():
                self.set_expected_nodes_with_rooms(adapter_id, nodes_to_room)
        elif nodes_by_adapter is not None:
            self._expected_nodes_by_adapter = dict(nodes_by_adapter)
            self._room_nodes_by_adapter = {}
            self._anchor_room_by_adapter = {}
        if not self._site_config_version:
            self._site_config_version = version
            return 0
        if version == self._site_config_version:
            return 0
        dropped = len(self._open_by_key)
        self._open_by_key.clear()
        self._last_closed_by_key.clear()
        self._site_config_version = version
        return dropped

    def _allocate_seq(self, key: tuple[str, str], explicit_seq: int | None) -> int:
        if explicit_seq is not None:
            self._next_seq_by_key[key] = max(self._next_seq_by_key.get(key, 0), explicit_seq + 1)
            return explicit_seq
        seq = self._next_seq_by_key.get(key, 1)
        self._next_seq_by_key[key] = seq + 1
        return seq

    def _close(self, key: tuple[str, str], bucket: _OpenEpoch) -> RangeEpoch:
        del self._open_by_key[key]
        if bucket.explicit_seq is not None:
            self._last_closed_by_key[key] = (bucket.explicit_seq, bucket.last_arrival_at)
        return bucket.freeze()

    def _is_straggler(
        self,
        key: tuple[str, str],
        measurement: RangeMeasurement,
        current: _OpenEpoch | None,
        arrival_at: datetime,
    ) -> bool:
        if measurement.epoch_seq is None:
            return False
        if current is not None and current.explicit_seq == measurement.epoch_seq:
            return False
        last_closed = self._last_closed_by_key.get(key)
        if last_closed is None or last_closed[0] != measurement.epoch_seq:
            return False
        since_close_s = (arrival_at - last_closed[1]).total_seconds()
        return 0.0 <= since_close_s <= self._straggler_horizon_s

    def _room_complete(self, adapter_id: str, heard: frozenset[str]) -> bool:
        if len(heard) < 3:
            return False
        anchor_rooms = self._anchor_room_by_adapter.get(adapter_id)
        if not anchor_rooms:
            return False
        heard_rooms = {anchor_rooms.get(anchor_id) for anchor_id in heard}
        if len(heard_rooms) != 1:
            return False
        sole_room = next(iter(heard_rooms))
        if sole_room is None:
            return False
        room_nodes = self._room_nodes_by_adapter.get(adapter_id, {}).get(sole_room)
        return room_nodes is not None and len(room_nodes) >= 3 and room_nodes.issubset(heard)

    def add(
        self,
        measurement: RangeMeasurement,
        *,
        now: datetime | None = None,
    ) -> list[RangeEpoch]:
        """Add *measurement* and return zero, one, or two completed epochs.

        *now* is the server arrival time used for idle expiry and straggler
        detection. When omitted (offline replay and bench evaluation), the
        measurement's own ``measured_at`` stands in for it.
        """
        key = (measurement.adapter_id, measurement.tag_id)
        arrival_at = ensure_utc(now) if now is not None else measurement.measured_at
        emitted: list[RangeEpoch] = []
        current = self._open_by_key.get(key)

        if self._is_straggler(key, measurement, current, arrival_at):
            if self._on_late_measurement is not None:
                self._on_late_measurement()
            return emitted

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
                emitted.append(self._close(key, current))
                current = None

        if current is None:
            seq = self._allocate_seq(key, measurement.epoch_seq)
            current = _OpenEpoch(
                tag_id=measurement.tag_id,
                adapter_id=measurement.adapter_id,
                epoch_seq=seq,
                window_start=measurement.measured_at,
                explicit_seq=measurement.epoch_seq,
                opened_at=arrival_at,
                last_arrival_at=arrival_at,
            )
            self._open_by_key[key] = current

        current.add(measurement)
        current.last_arrival_at = max(current.last_arrival_at, arrival_at)

        heard = frozenset(current.by_anchor.keys())
        expected = self._expected_nodes_by_adapter.get(measurement.adapter_id)
        # Room-scoped completion only for unbatched senders: a batch (sent_at set)
        # delivers the whole round at once, so closing on the first room would
        # turn the rest of the same message into dropped stragglers.
        room_complete = measurement.sent_at is None and self._room_complete(
            measurement.adapter_id, heard
        )
        if (expected and expected.issubset(heard)) or room_complete:
            emitted.append(self._close(key, current))

        return emitted

    def flush_expired(self, now: datetime) -> list[RangeEpoch]:
        """Emit open epochs idle for ``epoch_window_s`` (or open past ``max_open_s``) at *now*."""
        now_utc = ensure_utc(now)
        idle_cutoff = now_utc - timedelta(seconds=self._epoch_window_s)
        open_cutoff = now_utc - timedelta(seconds=self._max_open_s)
        expired_keys = [
            key
            for key, bucket in self._open_by_key.items()
            if bucket.last_arrival_at <= idle_cutoff or bucket.opened_at <= open_cutoff
        ]
        emitted = [self._close(key, self._open_by_key[key]) for key in expired_keys]
        self._prune_closed(now_utc)
        return emitted

    def _prune_closed(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._straggler_horizon_s)
        stale = [key for key, (_, at) in self._last_closed_by_key.items() if at < cutoff]
        for key in stale:
            del self._last_closed_by_key[key]

    def flush_all(self) -> list[RangeEpoch]:
        """Emit all open epochs regardless of window expiration (e.g. on shutdown)."""
        emitted = [self._close(key, bucket) for key, bucket in list(self._open_by_key.items())]
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


def infer_epoch_room(
    epoch: RangeEpoch,
    nodes: Mapping[str, RfNodeGeometry],
    *,
    exclude_anchor_id: str | None = None,
) -> tuple[str, float] | None:
    """Attribute a solved position epoch to a room as ``(room_id, probability)``.

    Uses the same nearest-node and margin-probability rule as
    :func:`room_fallback_from_epoch`, restricted to inlier measurements (the
    solver's dropped outlier anchor, if any, is excluded). The probability is
    never certainty: a corner anchor can sit on a wall shared with the next
    room, and downstream consumers read a room without a probability as 1.0.
    Returns ``None`` when no contributing node declares a ``room_id``.
    """
    measurements = tuple(m for m in epoch.measurements if m.anchor_id != exclude_anchor_id)
    if not measurements:
        return None
    report = room_fallback_from_epoch(replace(epoch, measurements=measurements), nodes)
    if report is None or report.probability is None:
        return None
    return report.room_id, report.probability
