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

from collections import deque
from datetime import datetime, timedelta
from statistics import median

import structlog

from rf_ingress.domain import ensure_utc
from rf_ingress.observability import CLOCK_SKEW_SECONDS, DEVICE_CLOCK_OFFSET_MS

logger = structlog.get_logger(__name__)


class BurstClockOffsetTracker:
    """Minimum-delay transport clock-offset estimator per registered ``tag_id``.

    For each incoming batch carrying ``sent_at`` and server ``arrival_at``,
    ``delta_transport_s = (arrival_at - sent_at).total_seconds()``. Because
    ``arrival_at = true_send + network_delay`` and
    ``sent_at = true_send + clock_ahead``, ``delta_transport_s = network_delay -
    clock_ahead`` is minimized when ``network_delay`` is smallest. Taking the
    running minimum over the sliding window estimates
    ``delta_clock_s ≈ -clock_ahead + min_network_delay``, and reconstructing
    each epoch timestamp as ``epoch_measured_at + delta_clock_s`` removes the
    device clock offset without being biased by multi-second batch buffering.
    """

    def __init__(self, *, window_size: int = 20) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size: int = window_size
        self._samples_by_tag: dict[str, deque[float]] = {}
        self._last_sent_at_by_tag: dict[str, datetime] = {}
        self._offset_by_tag_s: dict[str, float] = {}

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
        if self._last_sent_at_by_tag.get(tag_id) == sent_utc and tag_id in self._offset_by_tag_s:
            return self._offset_by_tag_s[tag_id]
        self._last_sent_at_by_tag[tag_id] = sent_utc
        delta_s = (arrival_utc - sent_utc).total_seconds()
        samples = self._samples_by_tag.get(tag_id)
        if samples is None:
            samples = deque(maxlen=self._window_size)
            self._samples_by_tag[tag_id] = samples
        samples.append(delta_s)
        offset_s = min(samples)
        self._offset_by_tag_s[tag_id] = offset_s
        if record_metric:
            DEVICE_CLOCK_OFFSET_MS.labels(tag_id=tag_id).set(offset_s * 1000.0)
        return offset_s

    def offset_seconds(self, tag_id: str) -> float | None:
        """Return the latest estimated clock offset in seconds for *tag_id*."""
        return self._offset_by_tag_s.get(tag_id)

    def correct_timestamp(self, tag_id: str, measured_at: datetime) -> datetime:
        """Apply the estimated clock offset for *tag_id* to *measured_at*."""
        measured_utc = ensure_utc(measured_at)
        offset_s = self._offset_by_tag_s.get(tag_id)
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
