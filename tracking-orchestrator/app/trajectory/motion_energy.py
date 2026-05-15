"""Per-tracklet motion-energy tracker from frame-to-frame keypoint displacement.

Maintains a bounded deque of recent keypoint positions per global_track_id
and returns per-frame ``MotionEnergy`` summaries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..domain import GlobalTrackId
from ..inference.schemas import PoseResult

_MAX_HISTORY = 30  # frames per track
_EVICT_AFTER_S = 300  # seconds since last update
# Keypoint indices used for velocity (COCO-17: shoulders, elbows, wrists, hips, knees, ankles)
_RAPID_JOINTS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
_VELOCITY_FLOOR = 0.005  # normalised px / frame; below this the frame is "still"


@dataclass(frozen=True)
class MotionEnergy:
    """Per-frame motion energy summary for a tracklet."""

    mean_keypoint_velocity_px_s: float
    max_joint_velocity_px_s: float
    still_fraction: float
    sample_count: int


class MotionEnergyTracker:
    """Tracks keypoint displacement across frames for each global_track_id.

    Usage::

        tracker = MotionEnergyTracker()
        energy = tracker.update("gt-001", pose_result, captured_at, bbox_diag_px=200.0)
    """

    def __init__(self) -> None:
        # global_track_id -> deque of (timestamp, normalized keypoints as 17x2 array)
        self._history: dict[GlobalTrackId, deque[tuple[datetime, np.ndarray]]] = {}
        self._last_update: dict[GlobalTrackId, datetime] = {}

    def update(
        self,
        global_track_id: GlobalTrackId,
        pose: PoseResult,
        captured_at: datetime,
        bbox_diag_px: float = 200.0,
    ) -> MotionEnergy:
        """Add a frame's keypoints and return the current motion energy.

        Args:
            global_track_id: track identifier.
            pose: 17 COCO keypoints from RTMPose.
            captured_at: wall-clock time of the observation.
            bbox_diag_px: bounding-box diagonal in pixels, used to normalise
                keypoint displacement so metrics are scale-invariant.
        """
        # Build a (17, 2) array of [x, y] in normalised crop coords
        kp_arr = np.array([[kp.x, kp.y] for kp in pose.keypoints], dtype=np.float64)
        now_ts = captured_at

        dq = self._history.get(global_track_id)
        if dq is None:
            dq = deque(maxlen=_MAX_HISTORY)
            self._history[global_track_id] = dq

        dq.append((now_ts, kp_arr))
        self._last_update[global_track_id] = now_ts

        # Evict stale tracks
        self._evict(now_ts)

        if len(dq) < 2:
            return MotionEnergy(
                mean_keypoint_velocity_px_s=0.0,
                max_joint_velocity_px_s=0.0,
                still_fraction=1.0,
                sample_count=len(dq),
            )

        # Compute frame-to-frame velocities for rapid joints
        velocities: list[float] = []
        still_count = 0
        n_pairs = 0

        for i in range(1, len(dq)):
            prev_ts, prev_kp = dq[i - 1]
            curr_ts, curr_kp = dq[i]
            dt = (curr_ts - prev_ts).total_seconds()
            if dt <= 0:
                continue
            n_pairs += 1

            # Per-joint displacement (only rapid joints)
            rapid_idxs = np.array(_RAPID_JOINTS)
            disp = np.linalg.norm(curr_kp[rapid_idxs] - prev_kp[rapid_idxs], axis=1)
            # Normalise by bbox diagonal so metric is scale-invariant
            scale = max(bbox_diag_px, 1.0)
            norm_disp = disp / scale
            frame_vel = float(np.mean(norm_disp) / dt)
            velocities.append(frame_vel)

            if frame_vel < _VELOCITY_FLOOR:
                still_count += 1

        if not velocities:
            return MotionEnergy(
                mean_keypoint_velocity_px_s=0.0,
                max_joint_velocity_px_s=0.0,
                still_fraction=1.0,
                sample_count=len(dq),
            )

        return MotionEnergy(
            mean_keypoint_velocity_px_s=round(np.mean(velocities).item(), 6),
            max_joint_velocity_px_s=round(np.max(velocities).item(), 6),
            still_fraction=round(still_count / n_pairs, 4),
            sample_count=len(dq),
        )

    def evict_track(self, global_track_id: GlobalTrackId) -> None:
        """Remove all history for a track (call on track close)."""
        self._history.pop(global_track_id, None)
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
            self._last_update.pop(gt_id, None)
