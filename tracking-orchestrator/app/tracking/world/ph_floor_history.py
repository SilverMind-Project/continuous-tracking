"""Bounded calibrated floor-plane position history per Person Hypothesis.

Used by wearable-to-PH trajectory association to interpolate a PH's floor-plane
position and covariance at each wearable fix's physical ``measured_at`` timestamp.

Invariants:
- Only rounds updated by a calibrated observation append a sample; uncalibrated
  virtual-cell coordinates and coasting predictions are never recorded.
- A sample is the camera posterior, or, while wearable spatial fusion is enabled, the
  raw camera measurement: a posterior that absorbed a fused wearable fix must never be scored
  against that wearable. The tracker decides which; this class stores either.
- History is retained across PH close-and-revive windows
  (``max_span_s = ph_close_grace_s + revive_max_age_s``) and cleared on PH split
  or merge.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class PHFloorSample:
    """One calibrated floor-plane state sample for a Person Hypothesis."""

    captured_at: datetime
    x_m: float
    y_m: float
    cov_m2: tuple[float, float, float, float]
    speed_m_s: float


class PHFloorHistory:
    """Per-PH bounded deque of calibrated floor-plane samples with interpolation."""

    def __init__(
        self,
        *,
        max_span_s: float = 45.0,
        max_samples_per_ph: int = 512,
        max_interp_gap_s: float = 5.0,
    ) -> None:
        if max_span_s <= 0.0:
            raise ValueError("max_span_s must be positive")
        if max_samples_per_ph < 2:
            raise ValueError("max_samples_per_ph must be >= 2")
        self._max_span_s = max_span_s
        self._max_samples_per_ph = max_samples_per_ph
        self._max_interp_gap_s = max_interp_gap_s
        self._samples: dict[str, deque[PHFloorSample]] = {}

    def record(
        self,
        ph_id: str,
        *,
        captured_at: datetime,
        x_m: float,
        y_m: float,
        cov_m2: tuple[float, float, float, float],
        speed_m_s: float,
    ) -> None:
        """Append one calibrated sample for *ph_id* and trim older samples."""
        sample = PHFloorSample(
            captured_at=captured_at,
            x_m=x_m,
            y_m=y_m,
            cov_m2=cov_m2,
            speed_m_s=speed_m_s,
        )
        dq = self._samples.get(ph_id)
        if dq is None:
            dq = deque(maxlen=self._max_samples_per_ph)
            self._samples[ph_id] = dq

        if not dq or captured_at >= dq[-1].captured_at:
            if dq and dq[-1].captured_at == captured_at:
                dq[-1] = sample
            else:
                dq.append(sample)
        else:
            # Maintain ascending timestamp order if an out-of-order calibrated
            # sample arrives inside a multi-camera round.
            ordered = list(dq)
            idx = len(ordered)
            while idx > 0 and ordered[idx - 1].captured_at > captured_at:
                idx -= 1
            if idx > 0 and ordered[idx - 1].captured_at == captured_at:
                ordered[idx - 1] = sample
            else:
                ordered.insert(idx, sample)
            dq = deque(ordered[-self._max_samples_per_ph :], maxlen=self._max_samples_per_ph)
            self._samples[ph_id] = dq

        cutoff = dq[-1].captured_at - timedelta(seconds=self._max_span_s)
        while dq and dq[0].captured_at < cutoff:
            dq.popleft()

    def has_calibrated_history(self, ph_id: str) -> bool:
        """Return True if *ph_id* has at least one calibrated floor sample."""
        dq = self._samples.get(ph_id)
        return bool(dq)

    def samples_for(self, ph_id: str) -> tuple[PHFloorSample, ...]:
        """Return immutable snapshot of stored calibrated samples for *ph_id*."""
        dq = self._samples.get(ph_id)
        return tuple(dq) if dq else ()

    def reset_ph(self, ph_id: str) -> None:
        """Clear stored history for *ph_id* (called on PH split or merge)."""
        self._samples.pop(ph_id, None)

    def prune(self, now: datetime) -> None:
        """Evict samples older than ``now - max_span_s`` and drop empty PH entries."""
        cutoff = now - timedelta(seconds=self._max_span_s)
        empty: list[str] = []
        for ph_id, dq in self._samples.items():
            while dq and dq[0].captured_at < cutoff:
                dq.popleft()
            if not dq:
                empty.append(ph_id)
        for ph_id in empty:
            del self._samples[ph_id]

    def interpolate(
        self,
        ph_id: str,
        at: datetime,
        *,
        max_extrapolation_s: float = 0.5,
    ) -> PHFloorSample | None:
        """Interpolate calibrated ``(x_m, y_m, cov_m2, speed_m_s)`` at timestamp *at*.

        Linearly interpolates between bracketing calibrated samples when their
        separation does not exceed ``max_interp_gap_s``. Outside the sample
        span, returns the boundary sample only when within
        ``max_extrapolation_s``; returns ``None`` otherwise.
        """
        dq = self._samples.get(ph_id)
        if not dq:
            return None

        first = dq[0]
        last = dq[-1]

        if at <= first.captured_at:
            if (first.captured_at - at).total_seconds() <= max_extrapolation_s:
                return first
            return None

        if at >= last.captured_at:
            if (at - last.captured_at).total_seconds() <= max_extrapolation_s:
                return last
            return None

        # Binary search for the bracketing pair s0.t <= at <= s1.t.
        lo = 0
        hi = len(dq) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if dq[mid].captured_at <= at:
                lo = mid
            else:
                hi = mid

        s0 = dq[lo]
        s1 = dq[hi]
        dt = (s1.captured_at - s0.captured_at).total_seconds()
        if dt <= 0.0:
            return s0
        if dt > self._max_interp_gap_s:
            # Refuse to interpolate across long unobserved gaps (for example a revival
            # occlusion); allow boundary match only within max_extrapolation_s.
            if (at - s0.captured_at).total_seconds() <= max_extrapolation_s:
                return s0
            if (s1.captured_at - at).total_seconds() <= max_extrapolation_s:
                return s1
            return None

        alpha = (at - s0.captured_at).total_seconds() / dt
        one_minus = 1.0 - alpha
        cov = (
            one_minus * s0.cov_m2[0] + alpha * s1.cov_m2[0],
            one_minus * s0.cov_m2[1] + alpha * s1.cov_m2[1],
            one_minus * s0.cov_m2[2] + alpha * s1.cov_m2[2],
            one_minus * s0.cov_m2[3] + alpha * s1.cov_m2[3],
        )
        return PHFloorSample(
            captured_at=at,
            x_m=one_minus * s0.x_m + alpha * s1.x_m,
            y_m=one_minus * s0.y_m + alpha * s1.y_m,
            cov_m2=cov,
            speed_m_s=one_minus * s0.speed_m_s + alpha * s1.speed_m_s,
        )
