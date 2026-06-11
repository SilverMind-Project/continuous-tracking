"""Per-PersonHypothesis fall-feature extractor from pose, posture, and Kalman state.

Consumes what the frame pipeline already computes every frame per detection
(pose keypoints, bbox, posture scores, motion energy, Kalman floor speed) and
maintains a short time-based ring buffer (default 3.0 s) per ``ph_id``,
exposing a :class:`FallFeatures` snapshot on demand. Pure arithmetic: no I/O,
no Triton, no DB.

Vertical conventions
--------------------
All vertical quantities are expressed in *normalized person-height units* so
they are camera-distance invariant: a pixel measurement is divided by ``H_est``,
the person's standing-height estimate (rolling p90 of bbox height over the
buffer). p90 adapts slowly, so it stays near the upright height for ~3 s after a
collapse, which is what lets the descent of the body toward the floor stand out.

Two distinct measures are derived from the body's vertical position, because a
single quantity cannot serve both jobs:

* ``max_descent_rate_hps`` is built from the body part's **absolute image y**
  (``bbox.y_min + kp.y * bbox.height``, down-positive). A fall translates the
  body down the frame, so absolute-y increases sharply. A person walking away
  from the camera shrinks their bbox *uniformly about its centre*, so the
  hip (near bbox centre) holds its absolute-y and produces no descent. This is
  the "box shrank because the person dropped" vs. "walked away" discriminator.
* ``height_ratio_now`` is built from the body part's **height above the bbox
  bottom** (``bbox.y_max - y_abs``, up-positive) normalized by ``H_est``. Upright
  it is ~1.0; on the floor the bbox is short so it falls to ~0.3. Walking away
  shrinks the bbox but the p90 anchor lags, so the ratio stays high (> 0.8).

Calibration of the thresholds that read these features lives in the detector
(task 2) and is documented there; this module only produces the features.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

import numpy as np

from ..domain import BoundingBox
from ..inference.schemas import Keypoint
from .posture import PostureScores

# COCO-17 keypoint index groups used for the vertical body proxy.
_HEAD_INDICES = (0, 1, 2, 3, 4)  # nose, eyes, ears
_SHOULDER_INDICES = (5, 6)  # left/right shoulder (head fallback)
_HIP_INDICES = (11, 12)  # left/right hip (primary vertical proxy)

_DEFAULT_BUFFER_S = 3.0
_DEFAULT_DESCENT_WINDOW_S = 1.0
_DEFAULT_POST_WINDOW_S = 2.0
_DEFAULT_SCORE_FLOOR = 0.3
_DEFAULT_ENTER_GRACE_S = 1.0
_EVICT_AFTER_S = 300.0  # seconds since last update before per-PH state is freed


@dataclass(frozen=True)
class FallFrameInput:
    """One frame's worth of already-computed per-detection signals for one PH."""

    captured_at: datetime
    bbox: BoundingBox  # frame pixels
    keypoints: tuple[Keypoint, ...] | None  # COCO-17, normalized crop coords, may be None
    posture_scores: PostureScores | None  # from a PostureStrategy.score
    floor_speed_m_s: float | None  # Kalman floor speed; None when uncalibrated
    motion_energy_nu_s: float | None  # from MotionEnergyTracker (nu/s)


@dataclass(frozen=True)
class FallFeatures:
    """Fall-relevant features over a PH's recent buffer.

    See the module docstring for the vertical conventions. ``*_at_event`` fields
    are sampled at the frame ending the maximum-descent adjacent pair.
    """

    max_descent_rate_hps: float
    """Maximum downward velocity of the body in heights-per-second over any
    adjacent vy-present pair within ``descent_window_s``. 0.0 when no valid pair
    exists. The canonical fall signature is > ~0.8 hps; controlled sitting is
    2-4x slower."""

    height_ratio_now: float
    """Current body height above the bbox bottom relative to the buffer p90.
    ~1.0 upright, ~0.3 on the floor. 1.0 by definition until ``enter_grace_s``
    of data exists (no anchor yet)."""

    lying_score_now: float
    """Latest ``posture_scores.lying`` (0.0 when None)."""

    post_event_motion_nu_s: float | None
    """Mean motion energy over ``post_window_s`` after the max-descent event.
    None when there is no event or the event is younger than the window."""

    floor_speed_at_event_m_s: float | None
    """Kalman floor speed at the max-descent event. None when there is no event
    or the camera was uncalibrated at that frame."""

    samples: int
    """Frame count in the buffer, for the detector's sufficiency gate."""

    pose_available_now: bool
    """True when the latest frame had at least one keypoint above the score
    floor. False means pose was unavailable (person occluded or flat on
    floor). Used by detector rule 4: absence of pose after a descent spike is
    itself evidence of a fall."""


@dataclass(frozen=True)
class _Sample:
    """Pre-derived per-frame values retained in the ring buffer."""

    captured_at: datetime
    bbox_height: float
    vertical_y: float | None  # absolute image y of the body proxy (down-positive)
    height_above_floor_px: float | None  # bbox.y_max - vertical_y (up-positive)
    lying_score: float
    floor_speed_m_s: float | None
    motion_energy_nu_s: float | None


def _mean_image_y(
    keypoints: tuple[Keypoint, ...],
    indices: tuple[int, ...],
    bbox: BoundingBox,
    score_floor: float,
) -> float | None:
    """Mean absolute image y of the named keypoints above the score floor.

    Crop-normalized ``kp.y`` is converted to absolute frame pixels via
    ``bbox.y_min + kp.y * bbox.height``. Returns None when no listed keypoint
    clears the score floor (never fabricate a position).
    """
    ys = [
        bbox.y_min + keypoints[i].y * bbox.height
        for i in indices
        if keypoints[i].score >= score_floor
    ]
    if not ys:
        return None
    return float(np.mean(ys))


def _body_vertical_y(
    keypoints: tuple[Keypoint, ...],
    bbox: BoundingBox,
    score_floor: float,
) -> float | None:
    """Absolute image y of the most reliable vertical body proxy.

    Prefers the hip midpoint (near the bbox centre, so robust to uniform bbox
    shrink). Falls back to the head proxy (nose/eyes/ears), then the shoulder
    midpoint, mirroring the posture classifier's fallback chain. Returns None
    when no usable keypoints exist.
    """
    hip_y = _mean_image_y(keypoints, _HIP_INDICES, bbox, score_floor)
    if hip_y is not None:
        return hip_y
    head_y = _mean_image_y(keypoints, _HEAD_INDICES, bbox, score_floor)
    if head_y is not None:
        return head_y
    return _mean_image_y(keypoints, _SHOULDER_INDICES, bbox, score_floor)


class FallFeatureExtractor:
    """Maintains a per-PH time-based ring buffer and emits :class:`FallFeatures`.

    The buffer length is time-based (not count-based) because camera cadence
    varies (5 fps polling, motion-gated gaps). State is freed per PH on
    :meth:`evict` and automatically after ``evict_after_s`` of idleness.

    Usage::

        extractor = FallFeatureExtractor()
        features = extractor.update("ph-001", frame_input)
    """

    def __init__(
        self,
        *,
        buffer_s: float = _DEFAULT_BUFFER_S,
        descent_window_s: float = _DEFAULT_DESCENT_WINDOW_S,
        post_window_s: float = _DEFAULT_POST_WINDOW_S,
        score_floor: float = _DEFAULT_SCORE_FLOOR,
        enter_grace_s: float = _DEFAULT_ENTER_GRACE_S,
        evict_after_s: float = _EVICT_AFTER_S,
    ) -> None:
        self._buffer_s = buffer_s
        self._descent_window_s = descent_window_s
        self._post_window_s = post_window_s
        self._score_floor = score_floor
        self._enter_grace_s = enter_grace_s
        self._evict_after_s = evict_after_s
        self._buffers: dict[str, deque[_Sample]] = {}
        self._last_update: dict[str, datetime] = {}

    def update(self, ph_id: str, frame: FallFrameInput) -> FallFeatures:
        """Ingest one frame for ``ph_id`` and return current features."""
        sample = self._to_sample(frame)

        buffer = self._buffers.setdefault(ph_id, deque())
        buffer.append(sample)
        self._last_update[ph_id] = sample.captured_at

        # Drop frames older than the buffer window, then evict idle PHs.
        cutoff = sample.captured_at.timestamp() - self._buffer_s
        while buffer and buffer[0].captured_at.timestamp() < cutoff:
            buffer.popleft()
        self._evict_idle(sample.captured_at)

        return self._compute(buffer)

    def evict(self, ph_id: str) -> None:
        """Free all state for a PH (call on PH close)."""
        self._buffers.pop(ph_id, None)
        self._last_update.pop(ph_id, None)

    def _to_sample(self, frame: FallFrameInput) -> _Sample:
        bbox_height = float(max(frame.bbox.height, 1))
        vertical_y: float | None = None
        height_above_floor_px: float | None = None
        if frame.keypoints is not None:
            vertical_y = _body_vertical_y(frame.keypoints, frame.bbox, self._score_floor)
            if vertical_y is not None:
                height_above_floor_px = float(frame.bbox.y_max) - vertical_y
        lying_score = frame.posture_scores.lying if frame.posture_scores is not None else 0.0
        return _Sample(
            captured_at=frame.captured_at,
            bbox_height=bbox_height,
            vertical_y=vertical_y,
            height_above_floor_px=height_above_floor_px,
            lying_score=lying_score,
            floor_speed_m_s=frame.floor_speed_m_s,
            motion_energy_nu_s=frame.motion_energy_nu_s,
        )

    def _evict_idle(self, now: datetime) -> None:
        stale = [
            ph_id
            for ph_id, last in self._last_update.items()
            if (now - last).total_seconds() > self._evict_after_s
        ]
        for ph_id in stale:
            self._buffers.pop(ph_id, None)
            self._last_update.pop(ph_id, None)

    def _compute(self, buffer: deque[_Sample]) -> FallFeatures:
        samples = list(buffer)
        n = len(samples)
        latest = samples[-1]
        h_est = float(np.percentile([s.bbox_height for s in samples], 90))
        h_est = max(h_est, 1.0)

        max_rate, event_idx = self._max_descent(samples, h_est)
        height_ratio = self._height_ratio(samples)
        post_motion = self._post_event_motion(samples, event_idx)
        floor_speed = samples[event_idx].floor_speed_m_s if event_idx is not None else None
        pose_available = latest.vertical_y is not None

        return FallFeatures(
            max_descent_rate_hps=round(max_rate, 6),
            height_ratio_now=round(height_ratio, 6),
            lying_score_now=round(latest.lying_score, 6),
            post_event_motion_nu_s=(None if post_motion is None else round(post_motion, 6)),
            floor_speed_at_event_m_s=floor_speed,
            samples=n,
            pose_available_now=pose_available,
        )

    def _max_descent(self, samples: list[_Sample], h_est: float) -> tuple[float, int | None]:
        """Maximum down-positive velocity (heights/s) over adjacent vy-present pairs.

        Pairs are consecutive among frames that have a vertical proxy, so a frame
        with missing keypoints is skipped rather than breaking the series. A pair
        is only a valid descent measurement when its spacing is positive and at
        most ``descent_window_s`` -- a drop measured across a long motion-gated
        gap is not a fall signature. The whole buffer is searched (not just the
        last second) so the event, and the rate it produced, persist for the
        detector and for the post-event motion window. Returns the rate and the
        buffer index of the later frame of the winning pair.
        """
        present = [(i, s) for i, s in enumerate(samples) if s.vertical_y is not None]
        if len(present) < 2:
            return 0.0, None
        max_rate = 0.0
        event_idx: int | None = None
        for (_, prev), (cur_i, cur) in pairwise(present):
            dt = (cur.captured_at - prev.captured_at).total_seconds()
            if dt <= 0 or dt > self._descent_window_s:  # duplicate, reorder, or long gap
                continue
            assert prev.vertical_y is not None and cur.vertical_y is not None
            rate = (cur.vertical_y - prev.vertical_y) / h_est / dt
            if event_idx is None or rate > max_rate:
                max_rate = rate
                event_idx = cur_i
        if event_idx is None:
            return 0.0, None
        return max(max_rate, 0.0), event_idx

    def _height_ratio(self, samples: list[_Sample]) -> float:
        """Current height-above-floor relative to the buffer p90 (1.0 upright)."""
        span = (samples[-1].captured_at - samples[0].captured_at).total_seconds()
        if span < self._enter_grace_s:
            return 1.0  # no anchor yet for a freshly entered person
        heights = [s.height_above_floor_px for s in samples if s.height_above_floor_px is not None]
        if not heights:
            return 1.0
        current = heights[-1]
        anchor = float(np.percentile(heights, 90))
        if anchor <= 0.0:
            return 1.0
        return current / anchor

    def _post_event_motion(self, samples: list[_Sample], event_idx: int | None) -> float | None:
        """Mean motion energy in ``post_window_s`` after the max-descent event.

        None when there is no event, the window is not yet complete (the newest
        frame is younger than ``event + post_window_s``), or no motion samples
        landed in the window.
        """
        if event_idx is None:
            return None
        event_ts = samples[event_idx].captured_at
        if (samples[-1].captured_at - event_ts).total_seconds() < self._post_window_s:
            return None
        window_end = event_ts.timestamp() + self._post_window_s
        energies = [
            s.motion_energy_nu_s
            for s in samples[event_idx:]
            if s.motion_energy_nu_s is not None and s.captured_at.timestamp() <= window_end
        ]
        if not energies:
            return None
        return float(np.mean(energies))
