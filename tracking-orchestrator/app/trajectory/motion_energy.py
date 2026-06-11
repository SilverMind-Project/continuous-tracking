"""Per-tracklet motion-energy tracker from frame-to-frame keypoint displacement.

Maintains a bounded deque of recent absolute keypoint positions per
global_track_id and returns per-frame MotionEnergy summaries.

Unit: mean keypoint displacement in normalized crop units per second (nu/s).
Dimensionless by construction: absolute pixel displacement / bbox_diagonal_px / dt_s.

Calibration (provisional -- recalibrate after 1 week of live data):
  A 1.7 m person at a typical bbox_diagonal of ~340 px (bbox ~150x300 px):
    Still (seated, breathing): absolute keypoint jitter ~2-3 px/frame at 5 fps
      -> 2.5 px / 340 * 5 = 0.037 nu/s typical; p95 ~0.05 nu/s.
    Walking (1 m/s): body translation ~35 px/frame (0.2 m * 300px/1.7m)
      plus limb swing ~10 px/frame relative; mean across 12 rapid joints ~40 px/frame
      -> 40 px / 340 * 5 = 0.59 nu/s; p5 ~0.3 nu/s.
  Still floor set at p95 of still segments (0.05 nu/s, rounded up from 0.037).
  Walking threshold set between still p95 and walking p5: 0.15 nu/s.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..domain import BoundingBox, GlobalTrackId
from ..inference.schemas import PoseResult

_MAX_HISTORY = 30  # frames per track
_EVICT_AFTER_S = 300  # seconds since last update
# Keypoint indices used for velocity (COCO-17: shoulders, elbows, wrists, hips, knees, ankles)
_RAPID_JOINTS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
# provisional, recalibrate after 1 week of live data using scripts/calibrate_motion_energy.py
_STILL_VELOCITY_FLOOR_NU_S = 0.05  # nu/s; below this a frame-pair is "still"


@dataclass(frozen=True)
class MotionEnergy:
    """Per-frame motion energy summary for a tracklet.

    All velocity fields are in normalized units per second (nu/s):
    absolute pixel displacement divided by bbox_diagonal_px, divided by dt.
    """

    mean_keypoint_velocity_nu_s: float
    max_joint_velocity_nu_s: float
    still_fraction: float
    sample_count: int


class MotionEnergyTracker:
    """Tracks keypoint displacement across frames for each global_track_id.

    Converts crop-normalized keypoints to absolute image coordinates using the
    bbox before differencing, so the metric is independent of bbox size and
    position (crop-tracking jitter cancels out).

    Usage::

        tracker = MotionEnergyTracker()
        energy = tracker.update("gt-001", pose_result, captured_at, bbox)
    """

    def __init__(self) -> None:
        # global_track_id -> deque of (timestamp, absolute keypoints as 17x2 array in px)
        self._history: dict[GlobalTrackId, deque[tuple[datetime, np.ndarray]]] = {}
        # Incremental: per-track deque of (frame_velocity, is_still) for each consecutive pair
        self._pair_velocities: dict[GlobalTrackId, deque[tuple[float, bool]]] = {}
        self._last_update: dict[GlobalTrackId, datetime] = {}

    def update(
        self,
        global_track_id: GlobalTrackId,
        pose: PoseResult,
        captured_at: datetime,
        bbox: BoundingBox,
    ) -> MotionEnergy:
        """Add a frame's keypoints and return the current motion energy.

        Args:
            global_track_id: track identifier.
            pose: 17 COCO keypoints from RTMPose (crop-normalized [0, 1]).
            captured_at: wall-clock time of the observation.
            bbox: bounding box in full-image pixel coordinates. Used to convert
                crop-normalized keypoints to absolute pixel positions and to
                compute the normalization scale (bbox diagonal in pixels).
        """
        # Convert crop-normalized [0,1] coords to absolute full-image pixel coords.
        bbox_w = max(bbox.width, 1)
        bbox_h = max(bbox.height, 1)
        abs_kp = np.array(
            [[bbox.x_min + kp.x * bbox_w, bbox.y_min + kp.y * bbox_h] for kp in pose.keypoints],
            dtype=np.float64,
        )
        bbox_diag = float(np.hypot(bbox_w, bbox_h))
        scale = max(bbox_diag, 1.0)

        now_ts = captured_at

        # Initialise per-track structures lazily.
        if global_track_id not in self._history:
            self._history[global_track_id] = deque(maxlen=_MAX_HISTORY)
            self._pair_velocities[global_track_id] = deque(maxlen=_MAX_HISTORY - 1)

        dq = self._history[global_track_id]
        pair_dq = self._pair_velocities[global_track_id]

        # Compute the newest pair incrementally before appending the new frame.
        if dq:
            prev_ts, prev_kp = dq[-1]
            dt = (now_ts - prev_ts).total_seconds()
            if dt > 0:
                rapid_idxs = np.array(_RAPID_JOINTS)
                disp = np.linalg.norm(abs_kp[rapid_idxs] - prev_kp[rapid_idxs], axis=1)
                frame_vel = float(np.mean(disp / scale) / dt)
                pair_dq.append((frame_vel, frame_vel < _STILL_VELOCITY_FLOOR_NU_S))

        dq.append((now_ts, abs_kp))
        self._last_update[global_track_id] = now_ts

        # Evict stale tracks.
        self._evict(now_ts)

        if not pair_dq:
            return MotionEnergy(
                mean_keypoint_velocity_nu_s=0.0,
                max_joint_velocity_nu_s=0.0,
                still_fraction=1.0,
                sample_count=len(dq),
            )

        velocities = [v for v, _ in pair_dq]
        still_count = sum(1 for _, is_still in pair_dq if is_still)
        return MotionEnergy(
            mean_keypoint_velocity_nu_s=round(float(np.mean(velocities)), 6),
            max_joint_velocity_nu_s=round(float(np.max(velocities)), 6),
            still_fraction=round(still_count / len(pair_dq), 4),
            sample_count=len(dq),
        )

    def evict_track(self, global_track_id: GlobalTrackId) -> None:
        """Remove all history for a track (call on track close)."""
        self._history.pop(global_track_id, None)
        self._pair_velocities.pop(global_track_id, None)
        self._last_update.pop(global_track_id, None)

    def _evict(self, now: datetime) -> None:
        """Remove tracks not updated for > _EVICT_AFTER_S seconds."""
        stale = [
            gt_id
            for gt_id, last in self._last_update.items()
            if (now - last).total_seconds() > _EVICT_AFTER_S
        ]
        for gt_id in stale:
            self._history.pop(gt_id, None)
            self._pair_velocities.pop(gt_id, None)
            self._last_update.pop(gt_id, None)
