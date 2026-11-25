"""Per-device burst clock-offset decoupling and per-adapter clock-skew tracking.

Batched wireless senders (such as a watch flushing 3-second micro-batches during
a 12-second UWB burst) carry both a packet-level ``sent_at`` timestamp and
per-epoch ``measured_at`` timestamps. Estimating offset from
``arrival - measured_at`` would mistake the batch flush delay for clock skew.
:class:`BurstClockOffsetTracker` decouples the two by tracking the minimum
transport delta ``arrival - sent_at`` over a sliding window (matching NTP's
minimum-delay filter because network queueing delay is strictly non-negative).
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median

import structlog

from rf_ingress.domain import ensure_utc
from rf_ingress.observability import CLOCK_SKEW_SECONDS, DEVICE_CLOCK_OFFSET_MS

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _DelaySample:
    """One batch transport delta ``arrival - sent_at`` stamped with server arrival time."""

    arrival_utc: datetime
    delta_s: float


@dataclass
class _TagClockState:
    samples: deque[_DelaySample]
    pending_step: list[_DelaySample] = field(default_factory=list)
    last_sent_at: datetime | None = None
    offset_s: float | None = None


class BurstClockOffsetTracker:
    """Minimum-delay transport clock-offset estimator per registered ``tag_id``.

    For each incoming batch carrying ``sent_at`` and server ``arrival_at``,
    ``delta_transport_s = (arrival_at - sent_at).total_seconds()``. Because
    ``arrival_at = true_send + network_delay`` and
    ``sent_at = true_send + clock_ahead``, ``delta_transport_s = network_delay -
    clock_ahead`` is minimized when ``network_delay`` is smallest. Taking the
    running minimum over the sliding window estimates
    ``delta_clock_s ~ -clock_ahead + min_network_delay``, and reconstructing
    each epoch timestamp as ``epoch_measured_at + delta_clock_s`` removes the
    device clock offset without being biased by multi-second batch buffering.

    Duty-cycled senders publish only a few bursts per hour, so the window is
    bounded in time as well as in count:

    - Samples older than ``max_sample_age_s`` (server arrival time) expire.
    - A downward step (device clock jumped ahead) needs no detection: the new,
      smaller delta becomes the minimum immediately.
    - An upward step (device clock stepped back, or its boot anchor moved) is
      confirmed only by ``step_confirm_batches`` consecutive batches that all
      exceed the window minimum by at least ``step_reset_threshold_s`` and
      agree with each other within ``step_consistency_s``. The window is then
      reset to those batches. One late batch alone never resets the window:
      a backlog delivered late looks exactly like an upward step for a single
      sample, and resetting on it would make stale data look fresh. A backlog
      flush also fails the agreement test because its batches were sent
      seconds apart and arrive together.
    - At most ``max_tracked_tags`` tags are tracked; the least recently seen
      tag is evicted first.
    """

    def __init__(
        self,
        *,
        window_size: int = 20,
        max_sample_age_s: float = 1800.0,
        step_reset_threshold_s: float = 2.0,
        step_confirm_batches: int = 2,
        step_consistency_s: float = 0.5,
        max_tracked_tags: int = 256,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not math.isfinite(max_sample_age_s) or max_sample_age_s <= 0.0:
            raise ValueError("max_sample_age_s must be finite and > 0")
        if not math.isfinite(step_reset_threshold_s) or step_reset_threshold_s <= 0.0:
            raise ValueError("step_reset_threshold_s must be finite and > 0")
        if step_confirm_batches < 2:
            raise ValueError("step_confirm_batches must be >= 2")
        if not math.isfinite(step_consistency_s) or step_consistency_s <= 0.0:
            raise ValueError("step_consistency_s must be finite and > 0")
        if max_tracked_tags < 1:
            raise ValueError("max_tracked_tags must be >= 1")
        self._window_size: int = window_size
        self._max_sample_age: timedelta = timedelta(seconds=max_sample_age_s)
        self._step_reset_threshold_s: float = step_reset_threshold_s
        self._step_confirm_batches: int = step_confirm_batches
        self._step_consistency_s: float = step_consistency_s
        self._max_tracked_tags: int = max_tracked_tags
        self._state_by_tag: OrderedDict[str, _TagClockState] = OrderedDict()

    def _state_for(self, tag_id: str) -> _TagClockState:
        state = self._state_by_tag.get(tag_id)
        if state is None:
            state = _TagClockState(samples=deque(maxlen=self._window_size))
            self._state_by_tag[tag_id] = state
            while len(self._state_by_tag) > self._max_tracked_tags:
                self._state_by_tag.popitem(last=False)
        else:
            self._state_by_tag.move_to_end(tag_id)
        return state

    def observe_batch(
        self,
        tag_id: str,
        *,
        sent_at: datetime,
        arrival_at: datetime,
        record_metric: bool = True,
    ) -> float:
        """Record one batch arrival and return the updated clock offset in seconds.

        Multiple range measurements unpacked from the same batch share one
        ``sent_at`` timestamp; consecutive reports with the same ``sent_at``
        for a tag count as a single batch sample so the sliding window always
        spans the last ``window_size`` batch messages.
        """
        sent_utc = ensure_utc(sent_at)
        arrival_utc = ensure_utc(arrival_at)
        state = self._state_for(tag_id)
        if state.last_sent_at == sent_utc and state.offset_s is not None:
            return state.offset_s
        state.last_sent_at = sent_utc
        sample = _DelaySample(
            arrival_utc=arrival_utc,
            delta_s=(arrival_utc - sent_utc).total_seconds(),
        )

        cutoff = arrival_utc - self._max_sample_age
        while state.samples and state.samples[0].arrival_utc < cutoff:
            state.samples.popleft()
        state.pending_step = [s for s in state.pending_step if s.arrival_utc >= cutoff]

        if state.samples:
            window_min = min(s.delta_s for s in state.samples)
            if sample.delta_s - window_min >= self._step_reset_threshold_s:
                state.pending_step.append(sample)
                del state.pending_step[: -self._step_confirm_batches]
            else:
                state.pending_step.clear()
        state.samples.append(sample)

        pending = state.pending_step
        if len(pending) >= self._step_confirm_batches:
            recent = list(pending)
            spread_s = max(s.delta_s for s in recent) - min(s.delta_s for s in recent)
            if spread_s <= self._step_consistency_s:
                state.samples.clear()
                state.samples.extend(recent)
                state.pending_step.clear()
                logger.info(
                    "rf_device_clock_step_reset",
                    confirm_batches=len(recent),
                    new_offset_s=round(min(s.delta_s for s in recent), 4),
                )

        offset_s = min(s.delta_s for s in state.samples)
        state.offset_s = offset_s
        if record_metric:
            DEVICE_CLOCK_OFFSET_MS.labels(tag_id=tag_id).set(offset_s * 1000.0)
        return offset_s

    def offset_seconds(self, tag_id: str) -> float | None:
        """Return the latest estimated clock offset in seconds for *tag_id*."""
        state = self._state_by_tag.get(tag_id)
        return state.offset_s if state is not None else None

    def correct_timestamp(self, tag_id: str, measured_at: datetime) -> datetime:
        """Apply the estimated clock offset for *tag_id* to *measured_at*."""
        measured_utc = ensure_utc(measured_at)
        offset_s = self.offset_seconds(tag_id)
        if offset_s is None:
            return measured_utc
        return measured_utc + timedelta(seconds=offset_s)


class AdapterClockSkewEstimator:
    """Sliding-window median ``arrival - device`` skew monitor per adapter."""

    def __init__(
        self,
        *,
        window_s: float = 60.0,
        warn_threshold_s: float = 0.5,
    ) -> None:
        if window_s <= 0.0:
            raise ValueError("window_s must be > 0")
        if warn_threshold_s <= 0.0:
            raise ValueError("warn_threshold_s must be > 0")
        self._window_s: float = window_s
        self._warn_threshold_s: float = warn_threshold_s
        self._samples_by_adapter: dict[str, deque[tuple[datetime, float]]] = {}
        self._warned_adapters: set[str] = set()

    def observe(
        self,
        adapter_id: str,
        *,
        device_time: datetime,
        arrival_time: datetime,
    ) -> float:
        """Record one observation and return the current median skew in seconds."""
        device_utc = ensure_utc(device_time)
        arrival_utc = ensure_utc(arrival_time)
        delta_s = (arrival_utc - device_utc).total_seconds()
        samples = self._samples_by_adapter.get(adapter_id)
        if samples is None:
            samples = deque()
            self._samples_by_adapter[adapter_id] = samples
        samples.append((arrival_utc, delta_s))
        cutoff = arrival_utc - timedelta(seconds=self._window_s)
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        skew_s = float(median(s for _, s in samples))
        CLOCK_SKEW_SECONDS.labels(adapter_id=adapter_id).set(skew_s)
        if abs(skew_s) > self._warn_threshold_s:
            if adapter_id not in self._warned_adapters:
                self._warned_adapters.add(adapter_id)
                logger.warning(
                    "rf_adapter_clock_skew_high",
                    adapter_id=adapter_id,
                    skew_s=round(skew_s, 4),
                    threshold_s=self._warn_threshold_s,
                )
        else:
            self._warned_adapters.discard(adapter_id)
        return skew_s

    def current_skew_seconds(self, adapter_id: str) -> float | None:
        """Return the latest median skew in seconds for *adapter_id*, if any."""
        samples = self._samples_by_adapter.get(adapter_id)
        if not samples:
            return None
        return float(median(s for _, s in samples))


def resolve_report_timestamp(
    *,
    measured_at: datetime,
    arrival_at: datetime,
    ntp_synced: bool,
    latency_s: float,
    tag_id: str,
    sent_at: datetime | None,
    burst_tracker: BurstClockOffsetTracker,
    record_tag_metric: bool = True,
) -> datetime:
    """Determine the UTC capture timestamp for an incoming report.

    - If ``sent_at`` is provided (batched wireless delivery), updates
      *burst_tracker* and applies the minimum-delay clock offset to
      ``measured_at``.
    - Else if ``ntp_synced`` is ``True``, preserves ``measured_at`` directly.
    - Otherwise uses ``arrival_at - latency_s``.
    """
    measured_utc = ensure_utc(measured_at)
    arrival_utc = ensure_utc(arrival_at)
    if sent_at is not None:
        burst_tracker.observe_batch(
            tag_id,
            sent_at=sent_at,
            arrival_at=arrival_utc,
            record_metric=record_tag_metric,
        )
        return burst_tracker.correct_timestamp(tag_id, measured_utc)
    if ntp_synced:
        return measured_utc
    return arrival_utc - timedelta(seconds=latency_s)
