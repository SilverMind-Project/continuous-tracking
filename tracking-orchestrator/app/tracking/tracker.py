"""Per-camera tracker using BoT-SORT-like association (Kalman + IoU).

Each camera gets its own tracker instance. The tracker maintains a set of
active tracks, each with a Kalman-filtered state estimate and a short-lived
local track ID. Detections from the current frame are associated with
existing tracks via Hungarian algorithm on an IoU cost matrix. Unmatched
detections spawn new tracks; unmatched tracks are decremented and
terminated when their lost count exceeds a threshold.

This module is deliberately dependency-free (no Torch, no external tracking
library). It vendors only the association logic and a simple 2D Kalman filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from ..domain import BoundingBox, Detection
from ..inference.schemas import Embedding

# ---------------------------------------------------------------------------
# Kalman filter for 2D bounding-box state
# ---------------------------------------------------------------------------

# State: [x_center, y_center, aspect_ratio, height, vx, vy, v_ar, v_h]
_STATE_DIM = 8
# Observation: [x_center, y_center, aspect_ratio, height]
_OBS_DIM = 4


class SimpleKalmanFilter:
    """Minimal 2D bounding-box Kalman filter (continuous measurement, discrete time)."""

    def __init__(
        self,
        process_noise: float = 0.05,
        measurement_noise: float = 0.5,
    ) -> None:
        # State transition matrix (identity + velocity coupling)
        self._F = np.eye(_STATE_DIM, dtype=np.float64)
        self._F[0, 4] = 1.0  # x_center += vx
        self._F[1, 5] = 1.0  # y_center += vy
        self._F[2, 6] = 1.0  # aspect_ratio += v_ar
        self._F[3, 7] = 1.0  # height += v_h

        # Measurement matrix (observe position + shape, not velocity)
        self._H = np.zeros((_OBS_DIM, _STATE_DIM), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0

        # Covariance matrices
        self._P = np.eye(_STATE_DIM, dtype=np.float64) * 10.0
        self._Q = np.eye(_STATE_DIM, dtype=np.float64) * process_noise
        self._R = np.eye(_OBS_DIM, dtype=np.float64) * measurement_noise

        # Initialize state and identity
        self._x: npt.NDArray[np.float64] = np.zeros(_STATE_DIM, dtype=np.float64)
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, observation: npt.NDArray[np.float64]) -> None:
        """Initialize filter state from the first observation."""
        self._x[:_OBS_DIM] = observation.flatten()
        self._x[_OBS_DIM:] = 0.0  # velocities start at zero
        self._P = np.eye(_STATE_DIM, dtype=np.float64) * 10.0
        self._initialized = True

    def predict(self) -> npt.NDArray[np.float64]:
        """Predict the next state. Returns predicted state vector."""
        if not self._initialized:
            return np.zeros(_STATE_DIM, dtype=np.float64)
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x.copy()

    def update(self, observation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Update with a new measurement. Returns corrected state vector."""
        if not self._initialized:
            self.initialize(observation)
            return self._x.copy()

        # Innovation
        y = observation.flatten() - self._H @ self._x
        cov_s = self._H @ self._P @ self._H.T + self._R
        kalman_gain = self._P @ self._H.T @ np.linalg.inv(cov_s)
        self._x = self._x + kalman_gain @ y
        eye = np.eye(_STATE_DIM, dtype=np.float64)
        self._P = (eye - kalman_gain @ self._H) @ self._P
        return self._x.copy()

    def state(self) -> npt.NDArray[np.float64]:
        """Return current state estimate."""
        return self._x.copy()


# ---------------------------------------------------------------------------
# Internal track representation
# ---------------------------------------------------------------------------


@dataclass
class _InternalTrack:
    """Internal representation of a single track within one camera."""

    local_track_id: str
    kalman: SimpleKalmanFilter
    bbox_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    embedding_history: list[Embedding] = field(default_factory=list)
    hit_count: int = 0  # frames matched
    lost_count: int = 0  # frames unmatched
    confirmed: bool = False  # promoted to confirmed after min_hits
    age: int = 0  # frames since creation


# ---------------------------------------------------------------------------
# Public track representation returned to callers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalTrack:
    """A short-lived track within a single camera view.

    This is the public-facing type returned by the tracker. It carries the
    association result for one detection in one frame.
    """

    local_track_id: str
    detection: Detection
    bbox: BoundingBox
    confidence: float
    age: int
    hit_count: int
    lost_count: int
    confirmed: bool
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerConfig:
    """Hyperparameters for the per-camera tracker."""

    # Threshold to create a new track from an unmatched detection.
    new_track_thresh: float = 0.7

    # Threshold below which a matched detection is ignored (noise).
    track_low_thresh: float = 0.1

    # Threshold above which a match is considered confident enough to
    # extend an existing track (used as a secondary filter).
    track_high_thresh: float = 0.6

    # Minimum IoU required for a match to be accepted (higher = stricter).
    # Effective as: accept match when iou_cost <= (1 - match_thresh).
    # Lowered from 0.4 → 0.2: accept matches at IoU ≥ 0.2 so that bbox
    # shift from turning-in-place does not break the track.
    match_thresh: float = 0.2

    # Frames without a match before a track is terminated.
    max_time_lost: int = 30

    # Frames a detection must be matched before the track is "confirmed".
    min_hits: int = 3

    # IoU weight in the association cost (appearance weight = 1 - iou_weight).
    # Lowered from 0.5 → 0.15: IoU dominates Hungarian assignment so that
    # appearance changes (front/back view) don't steer pairing away from the
    # spatially correct track.  Appearance still acts as a tiebreaker.
    appearance_weight: float = 0.15

    # Post-update dedup: a newly-spawned tracklet (age==1) whose bbox overlaps
    # an existing stable tracklet (age >= dedup_min_age) by more than this IoU
    # threshold is immediately dropped. Suppresses ghost re-detections.
    dedup_iou_threshold: float = 0.6
    dedup_min_age: int = 3


# ---------------------------------------------------------------------------
# Association helpers
# ---------------------------------------------------------------------------


def _iou_matrix(
    boxes_a: npt.NDArray[np.float64],
    boxes_b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute intersection-over-union matrix between two sets of boxes.

    Args:
        boxes_a: shape (N, 4) each row is [x1, y1, x2, y2].
        boxes_b: shape (M, 4) each row is [x1, y1, x2, y2].

    Returns:
        IoU matrix of shape (N, M). Values in [0, 1].
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    # Convert to center-size for area computation
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # Intersection top-left and bottom-right
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (N, M, 2)
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # (N, M, 2)

    width = np.maximum(0, rb[:, :, 0] - lt[:, :, 0])
    height = np.maximum(0, rb[:, :, 1] - lt[:, :, 1])
    intersection = width * height

    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def _embedding_distance(
    embeddings_a: npt.NDArray[np.floating],
    embeddings_b: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Compute cosine distance matrix between two sets of embeddings.

    Args:
        embeddings_a: shape (N, D). Accepts any floating dtype.
        embeddings_b: shape (M, D). Accepts any floating dtype.

    Returns:
        Cosine distance matrix of shape (N, M). Values in [0, 1].
    """
    a = embeddings_a.astype(np.float64)
    b = embeddings_b.astype(np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    # L2-normalize rows (using the cast float64 arrays).
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    norm_a = np.where(norm_a > 0, norm_a, 1.0)
    norm_b = np.where(norm_b > 0, norm_b, 1.0)

    a_norm = a / norm_a
    b_norm = b / norm_b

    cosine_sim = a_norm @ b_norm.T
    # Clip to numerical precision
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    return (1.0 - cosine_sim) / 2.0


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------


class PerCameraTracker:
    """Per-camera BoT-SORT-like tracker.

    Usage::

        tracker = PerCameraTracker(TrackerConfig())
        # Process frames in order
        tracks = tracker.update(detections, frame_index)
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        self._tracks: dict[str, _InternalTrack] = {}
        self._next_local_id: int = 0
        self._dedup_dropped: int = 0

    @property
    def active_track_count(self) -> int:
        """Number of currently active (non-terminated) tracks."""
        return len(self._tracks)

    def update(
        self,
        detections: list[Detection],
        embeddings: list[Embedding] | None = None,
        frame_index: int = 0,
    ) -> list[LocalTrack]:
        """Associate detections to existing tracks, spawn new tracks, terminate old ones.

        Args:
            detections: detections from the current frame, in arbitrary order.
            embeddings: per-detection ReID embeddings (same order as detections).
            frame_index: monotonically increasing frame index for the camera.

        Returns:
            List of LocalTrack objects, one per confirmed detection.
        """
        if not detections:
            # Predict and age all tracks even without detections so the
            # Kalman state doesn't freeze and track ages advance.
            for track in self._tracks.values():
                track.kalman.predict()
                track.age += 1
                track.lost_count += 1
            self._advance_lost_tracks()
            return []

        # Normalise: None → list of Nones, empty list → list of Nones.
        emb_list: list[Embedding | None] = (
            list(embeddings) if embeddings else [None] * len(detections)
        )

        # ---- Step 1: predict all track states ----
        for track in self._tracks.values():
            track.kalman.predict()
            track.age += 1

        # ---- Step 2: build cost matrix and associate ----
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = self._associate(
            detections, emb_list
        )

        # ---- Step 3: update matched tracks ----
        result: list[LocalTrack] = []
        for trk_id, det_idx in zip(matched_tracks, matched_dets, strict=True):
            track = self._tracks[trk_id]
            det = detections[det_idx]
            emb = emb_list[det_idx]

            track.hit_count += 1
            track.lost_count = 0
            track.confirmed = track.hit_count >= self._config.min_hits

            # Update Kalman filter with the detection's box center + aspect ratio
            bbox = det.bbox
            obs = np.array(
                [
                    bbox.center_x,
                    bbox.center_y,
                    bbox.width / max(bbox.height, 1),
                    float(bbox.height),
                ],
                dtype=np.float64,
            )
            track.kalman.update(obs)
            track.bbox_history.append((bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max))
            # Only record real embeddings; None means no appearance evidence.
            if emb is not None:
                track.embedding_history.append(emb)

            result.append(self._make_local_track(track, det, emb))

        # ---- Step 4: handle unmatched detections (new tracks) ----
        # Collect (local_id, det, emb) for all candidate new tracks so we can
        # dedup before adding to result.
        new_candidates: list[tuple[str, Detection, Embedding | None]] = []
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            emb = emb_list[det_idx]
            if det.confidence >= self._config.new_track_thresh:
                local_id = self._create_track(det, emb)
                new_candidates.append((local_id, det, emb))

        # ---- Step 4a: dedup — drop new tracklets that heavily overlap a
        # stable existing tracklet (age >= dedup_min_age). ----
        self._dedup_dropped = 0
        if new_candidates and self._config.dedup_iou_threshold < 1.0:
            new_ids = {lid for lid, _, _ in new_candidates}
            stable_tracks = [
                t
                for t in self._tracks.values()
                if t.age >= self._config.dedup_min_age and t.local_track_id not in new_ids
            ]
            if stable_tracks:
                stable_boxes = np.array(
                    [
                        [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                        for t in stable_tracks
                        for b in [t.bbox_history[-1]]
                    ],
                    dtype=np.float64,
                )
                kept_candidates: list[tuple[str, Detection, Embedding | None]] = []
                for local_id, det, emb in new_candidates:
                    new_track = self._tracks[local_id]
                    nb = new_track.bbox_history[-1]
                    new_box = np.array(
                        [[float(nb[0]), float(nb[1]), float(nb[2]), float(nb[3])]],
                        dtype=np.float64,
                    )
                    if _iou_matrix(new_box, stable_boxes).max() > self._config.dedup_iou_threshold:
                        del self._tracks[local_id]
                        self._dedup_dropped += 1
                    else:
                        kept_candidates.append((local_id, det, emb))
                new_candidates = kept_candidates

        for local_id, det, emb in new_candidates:
            track = self._tracks[local_id]
            result.append(self._make_local_track(track, det, emb))

        # ---- Step 5: advance unmatched tracks (lost count) ----
        for trk_id in unmatched_tracks:
            if trk_id in self._tracks:
                self._tracks[trk_id].lost_count += 1

        # ---- Step 6: terminate stale tracks ----
        self._advance_lost_tracks()

        return result

    def _create_track(self, detection: Detection, embedding: Embedding | None) -> str:
        local_id = f"track-{self._next_local_id}"
        self._next_local_id += 1

        bbox = detection.bbox
        obs = np.array(
            [
                bbox.center_x,
                bbox.center_y,
                bbox.width / max(bbox.height, 1),
                float(bbox.height),
            ],
            dtype=np.float64,
        )

        kalman = SimpleKalmanFilter()
        kalman.initialize(obs)

        self._tracks[local_id] = _InternalTrack(
            local_track_id=local_id,
            kalman=kalman,
            bbox_history=[(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)],
            embedding_history=[embedding] if embedding is not None else [],
            hit_count=1,
            lost_count=0,
            confirmed=False,
            age=1,
        )
        return local_id

    def _associate(
        self,
        detections: list[Detection],
        emb_list: list[Embedding | None],
    ) -> tuple[list[str], list[int], list[str], list[int]]:
        """Run Hungarian association between existing tracks and detections.

        Returns:
            (matched_track_ids, matched_det_indices,
             unmatched_track_ids, unmatched_det_indices)
        """
        if not self._tracks:
            # No existing tracks: all detections are unmatched.
            return ([], [], [], list(range(len(detections))))

        active_tracks = [
            (track_id, track)
            for track_id, track in self._tracks.items()
            if track.lost_count < self._config.max_time_lost
        ]

        if not active_tracks:
            return ([], [], [], list(range(len(detections))))

        n_tracks = len(active_tracks)
        n_dets = len(detections)

        # ---- Build combined cost matrix ----
        # IoU cost (normalized to [0, 1]; cost = 1 - IoU)
        if n_tracks > 0 and n_dets > 0:
            track_boxes = np.array(
                [
                    [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ]
                    for _, track in active_tracks
                    for bbox in [track.bbox_history[-1]]
                ],
                dtype=np.float64,
            )

            det_boxes = np.array(
                [
                    [
                        float(d.bbox.x_min),
                        float(d.bbox.y_min),
                        float(d.bbox.x_max),
                        float(d.bbox.y_max),
                    ]
                    for d in detections
                ],
                dtype=np.float64,
            )

            iou_cost = 1.0 - _iou_matrix(track_boxes, det_boxes)

            # Embedding distance (normalized to [0, 1])
            # Only use appearance cost for tracks that have embedding history.
            # New tracks (no history yet) rely on IoU alone to avoid
            # artificial advantage from a neutral embedding.
            has_history = [bool(track.embedding_history) for _, track in active_tracks]

            # Filter out None embeddings from detections.
            det_embs_valid: list[npt.NDArray[np.float32]] = []
            det_indices_valid: list[int] = []
            for i, emb in enumerate(emb_list):
                if emb is not None:
                    det_embs_valid.append(np.asarray(emb, dtype=np.float32))
                    det_indices_valid.append(i)

            det_embs_array: npt.NDArray[np.float32] | None = (
                np.array(det_embs_valid, dtype=np.float32) if det_embs_valid else None
            )

            if all(has_history) and det_embs_array is not None:
                track_embs = np.array(
                    [track.embedding_history[-1] for _, track in active_tracks], dtype=np.float32
                )
                emb_cost = _embedding_distance(track_embs, det_embs_array)
                # Combined cost — full weighted cost for all tracks
                cost = (
                    1.0 - self._config.appearance_weight
                ) * iou_cost + self._config.appearance_weight * emb_cost
            elif any(has_history):
                # Mixed: tracks with history get embedding cost,
                # tracks without get IoU-only cost (no appearance penalty).
                if det_embs_array is not None:
                    track_embs_list: list[npt.NDArray[np.float32]] = []
                    for _, track in active_tracks:
                        if track.embedding_history:
                            track_embs_list.append(
                                np.asarray(track.embedding_history[-1], dtype=np.float32)
                            )
                        else:
                            # No appearance evidence — row will be overridden to IoU-only.
                            track_embs_list.append(np.zeros(768, dtype=np.float32))
                    track_embs = np.array(track_embs_list, dtype=np.float32)
                    emb_cost = _embedding_distance(track_embs, det_embs_array)
                    # Combined cost
                    cost = (
                        1.0 - self._config.appearance_weight
                    ) * iou_cost + self._config.appearance_weight * emb_cost
                    # For tracks without embedding history, use IoU-only cost
                    for i, has in enumerate(has_history):
                        if not has:
                            cost[i, :] = iou_cost[i, :]
                else:
                    # No valid detection embeddings: use IoU only.
                    cost = iou_cost
            else:
                # No tracks have embedding history: use IoU only
                cost = iou_cost
        else:
            cost = np.zeros((n_tracks, n_dets), dtype=np.float64)

        # ---- Hungarian assignment ----
        track_indices, det_indices = linear_sum_assignment(cost)

        # ---- Filter by threshold and map positions → track IDs ----
        # Apply threshold to IoU cost, not combined cost. The match_thresh
        # semantic is a minimum IoU — appearance similarity is additive and
        # should not invalidate a good spatial match.
        matched_tracks: list[str] = []
        matched_dets: list[int] = []
        for trk_idx, det_idx in zip(track_indices, det_indices, strict=True):
            if (
                trk_idx < n_tracks
                and det_idx < n_dets
                and iou_cost[trk_idx, det_idx] <= (1.0 - self._config.match_thresh)
            ):
                matched_tracks.append(active_tracks[trk_idx][0])
                matched_dets.append(det_idx)

        # ---- Compute unmatched sets ----
        matched_set_track_ids = set(matched_tracks)
        matched_set_dets = set(matched_dets)

        unmatched_tracks: list[str] = [
            tid for tid, _ in active_tracks if tid not in matched_set_track_ids
        ]
        unmatched_dets = [i for i in range(n_dets) if i not in matched_set_dets]

        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets

    def _advance_lost_tracks(self) -> None:
        """Terminate tracks whose lost_count exceeds the threshold."""
        to_remove = [
            local_id
            for local_id, track in self._tracks.items()
            if track.lost_count >= self._config.max_time_lost
        ]
        for local_id in to_remove:
            del self._tracks[local_id]

    def _make_local_track(
        self,
        track: _InternalTrack,
        detection: Detection,
        embedding: Embedding | None,
    ) -> LocalTrack:
        return LocalTrack(
            local_track_id=track.local_track_id,
            detection=detection,
            bbox=detection.bbox,
            confidence=detection.confidence,
            age=track.age,
            hit_count=track.hit_count,
            lost_count=track.lost_count,
            confirmed=track.confirmed,
            embedding=embedding.tolist() if embedding is not None else None,
        )


class PerCameraTrackers:
    """Registry of per-camera trackers.

    Each camera gets its own tracker instance, isolated from others.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        self._trackers: dict[str, PerCameraTracker] = {}

    def update(
        self,
        camera_id: str,
        detections: list[Detection],
        embeddings: list[Embedding] | None = None,
        frame_index: int = 0,
    ) -> list[LocalTrack]:
        """Process detections for a single camera.

        Args:
            camera_id: the camera this frame belongs to.
            detections: detections from the current frame.
            embeddings: per-detection ReID embeddings (same order as detections).
            frame_index: frame index for ordering/age tracking.

        Returns:
            List of LocalTrack objects for this frame.
        """
        if camera_id not in self._trackers:
            self._trackers[camera_id] = PerCameraTracker(self._config)
        return self._trackers[camera_id].update(detections, embeddings, frame_index)

    def get_active_count(self, camera_id: str) -> int:
        """Return the number of active tracks for a camera."""
        tracker = self._trackers.get(camera_id)
        if tracker is None:
            return 0
        return tracker.active_track_count

    def get_dedup_dropped(self, camera_id: str) -> int:
        """Return the number of tracklets dropped by dedup in the last update() call."""
        tracker = self._trackers.get(camera_id)
        return tracker._dedup_dropped if tracker is not None else 0
