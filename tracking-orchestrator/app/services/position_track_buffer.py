"""In-memory buffer of recent anonymous position tracks for device-free tracker rounds.

Epoch: rolling window of ``max_age_s`` per sensor track, bounded by ``max_tracks``.
Samples are kept per ``(sensor_id, sensor_track_id)`` in ascending ``measured_at`` order.

A round takes each confirmed track's newest sample that no earlier round took. Handing a
sample to two rounds would apply one measurement to the Kalman filter twice and make the PH
falsely certain, so each track carries a consumed-through watermark. A track is confirmed
once it has existed for ``min_track_age_s`` without a gap of ``track_gap_s``: radar multipath
ghosts are mostly short-lived, and sensors reuse track ids after a target is lost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.domain import AnonymousPositionObservation

# Allow small forward clock skew between sensor gateways and the orchestrator.
_MAX_FUTURE_SKEW_S: float = 2.0


@dataclass
class _TrackState:
    samples: list[AnonymousPositionObservation] = field(default_factory=list)
    # First sample of the current uninterrupted run of this track id.
    started_at: datetime | None = None
    consumed_through: datetime | None = None


class PositionTrackBuffer:
    """Bounded per-track buffer implementing :class:`PositionTrackSource`."""

    def __init__(
        self,
        *,
        max_age_s: float = 10.0,
        track_gap_s: float = 2.0,
        max_tracks: int = 64,
        max_samples_per_track: int = 64,
    ) -> None:
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be positive")
        if track_gap_s <= 0.0:
            raise ValueError("track_gap_s must be positive")
        if max_tracks < 1 or max_samples_per_track < 1:
            raise ValueError("max_tracks and max_samples_per_track must be >= 1")
        self._max_age_s = max_age_s
        self._track_gap_s = track_gap_s
        self._max_tracks = max_tracks
        self._max_samples = max_samples_per_track
        self._tracks: dict[tuple[str, str], _TrackState] = {}

    def __len__(self) -> int:
        return len(self._tracks)

    def add(self, obs: AnonymousPositionObservation) -> None:
        """Insert *obs* into its track in ``measured_at`` order (exact repeats are dropped)."""
        state = self._tracks.get(obs.track_key)
        if state is None:
            if len(self._tracks) >= self._max_tracks:
                self._evict_least_recent()
            state = _TrackState()
            self._tracks[obs.track_key] = state
        samples = state.samples
        if any(s.measured_at == obs.measured_at for s in samples):
            return
        if samples and (obs.measured_at - samples[-1].measured_at).total_seconds() > (
            self._track_gap_s
        ):
            # The sensor reused this track id for a new target, or lost and found it again.
            samples.clear()
            state.started_at = None
        idx = len(samples)
        while idx > 0 and samples[idx - 1].measured_at > obs.measured_at:
            idx -= 1
        samples.insert(idx, obs)
        if state.started_at is None or obs.measured_at < state.started_at:
            state.started_at = obs.measured_at
        if len(samples) > self._max_samples:
            del samples[: len(samples) - self._max_samples]
        cutoff = samples[-1].measured_at - timedelta(seconds=self._max_age_s)
        while samples and samples[0].measured_at < cutoff:
            samples.pop(0)

    def take_round_tracks(
        self, *, now: datetime, max_age_s: float, min_track_age_s: float
    ) -> Sequence[AnonymousPositionObservation]:
        """Return each confirmed track's newest unconsumed sample and mark it consumed."""
        selected = self._select(now=now, max_age_s=max_age_s, min_track_age_s=min_track_age_s)
        for obs in selected:
            self._tracks[obs.track_key].consumed_through = obs.measured_at
        return selected

    def peek_round_tracks(
        self, *, now: datetime, max_age_s: float, min_track_age_s: float
    ) -> Sequence[AnonymousPositionObservation]:
        """Return what :meth:`take_round_tracks` would return, without consuming it."""
        return self._select(now=now, max_age_s=max_age_s, min_track_age_s=min_track_age_s)

    def clear(self) -> None:
        """Drop every buffered track."""
        self._tracks.clear()

    def _select(
        self, *, now: datetime, max_age_s: float, min_track_age_s: float
    ) -> tuple[AnonymousPositionObservation, ...]:
        self._prune(now)
        oldest = now - timedelta(seconds=max_age_s)
        newest_allowed = now + timedelta(seconds=_MAX_FUTURE_SKEW_S)
        selected: list[AnonymousPositionObservation] = []
        for state in self._tracks.values():
            candidates = [
                s
                for s in state.samples
                if oldest <= s.measured_at <= newest_allowed
                and (state.consumed_through is None or s.measured_at > state.consumed_through)
            ]
            if not candidates or state.started_at is None:
                continue
            latest = candidates[-1]
            if (latest.measured_at - state.started_at).total_seconds() < min_track_age_s:
                continue
            selected.append(latest)
        selected.sort(key=lambda o: (o.measured_at, o.sensor_id, o.sensor_track_id))
        return tuple(selected)

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._max_age_s)
        stale = [
            key
            for key, state in self._tracks.items()
            if not state.samples or state.samples[-1].measured_at < cutoff
        ]
        for key in stale:
            del self._tracks[key]

    def _evict_least_recent(self) -> None:
        oldest_key = min(
            self._tracks.items(),
            key=lambda item: (
                item[1].samples[-1].measured_at.timestamp() if item[1].samples else float("-inf")
            ),
        )[0]
        del self._tracks[oldest_key]
