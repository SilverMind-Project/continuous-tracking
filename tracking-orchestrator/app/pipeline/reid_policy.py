"""Adaptive ReID cadence policy.

Evaluated in InferenceStage before embed_batch() to decide whether
SOLIDER-ReID embeddings are needed for each frame. Ships dark behind
``pipeline.adaptive_reid.enabled`` (default false) with shadow mode
(``pipeline.adaptive_reid.shadow``, default true) for safe rollout.

Skip condition (steady-state only):
  exactly 1 detection, exactly 1 open PH within proximity_gate_m,
  that PH has a committed identity that is not approaching expiry,
  and we embedded for it less than refresh_interval_s ago.

Every other case triggers embedding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from ..domain import FaceAnchor, FloorPoint, PersonHypothesis
from ..inference.schemas import DetectionBox


@dataclass(frozen=True)
class AdaptiveReidConfig:
    """Configuration for the adaptive ReID cadence policy."""

    enabled: bool = False
    shadow: bool = True
    # Seconds between mandatory re-embeddings even in steady-state.
    refresh_interval_s: float = 10.0
    # Coarse gate radius (metres) for matching detections to nearby open PHs.
    proximity_gate_m: float = 2.0


class ReidNeedPolicy:
    """Per-frame embedding-need policy.

    Stateful (tracks last-embed-at per PH) but does no I/O.
    Thread-safety: dict mutations are GIL-protected; minor double-embeds
    on concurrent camera frames are acceptable.
    """

    def __init__(
        self,
        config: AdaptiveReidConfig,
        prior_maintenance_max_age_s: float = 120.0,
        overlap_group_cameras: frozenset[str] = frozenset(),
    ) -> None:
        self._config = config
        self._prior_maintenance_max_age_s = prior_maintenance_max_age_s
        self._overlap_group_cameras = overlap_group_cameras
        self._last_embed_at_by_ph: dict[str, datetime] = {}

    def should_embed_frame(
        self,
        detections: list[DetectionBox],
        floor_points: dict[int, FloorPoint],
        open_phs: list[PersonHypothesis],
        face_anchors: list[FaceAnchor],
        camera_id: str,
        now: datetime,
    ) -> tuple[bool, str, str | None]:
        """Decide whether ReID embedding is needed this frame.

        Returns:
            (should_embed, reason, matched_ph_id)
            - should_embed: True means call embed_batch, False means skip.
            - reason: human-readable tag for metrics/logging.
            - matched_ph_id: PH ID of the single matched PH when the skip
              path is taken, so record_embed can be called after embedding
              when should_embed is True and a single PH was matched. None
              when there is no unique PH match.
        """
        # Prune stale PH entries from closed tracks.
        active_ids = {ph.ph_id for ph in open_phs}
        for pid in list(self._last_embed_at_by_ph):
            if pid not in active_ids:
                del self._last_embed_at_by_ph[pid]

        # Camera is part of a declared overlap group → dedup needs appearance.
        if camera_id in self._overlap_group_cameras:
            return True, "overlap_group_camera", None

        # Multiple detections → ambiguous assignment.
        if len(detections) != 1:
            return True, "multi_detect", None

        # Recognized face anchor present → gallery seeding may be needed.
        if any(a.recognition_state == "recognized" for a in face_anchors):
            return True, "face_anchor", None

        fp = floor_points.get(0)
        if fp is None:
            return True, "no_floor_point", None

        det_x_m = fp.x_mm / 1000.0
        det_y_m = fp.y_mm / 1000.0

        nearby = [
            ph
            for ph in open_phs
            if _dist_m(ph.state_mean[0], ph.state_mean[1], det_x_m, det_y_m)
            <= self._config.proximity_gate_m
        ]

        if len(nearby) == 0:
            return True, "no_nearby_ph", None
        if len(nearby) > 1:
            return True, "multi_nearby_ph", None

        ph = nearby[0]

        if ph.current_identity_id is None:
            return True, "ph_unresolved", None

        committed_at = ph.current_identity_committed_at
        if committed_at is None:
            return True, "ph_unresolved", None

        identity_age_s = (now - committed_at).total_seconds()
        expiry_margin_s = self._prior_maintenance_max_age_s - self._config.refresh_interval_s
        if identity_age_s >= expiry_margin_s:
            return True, "identity_expiring", ph.ph_id

        last_embed = self._last_embed_at_by_ph.get(ph.ph_id)
        elapsed = (now - last_embed).total_seconds() if last_embed is not None else float("inf")
        if elapsed >= self._config.refresh_interval_s:
            return True, "refresh_interval_elapsed", ph.ph_id

        return False, "steady_state", ph.ph_id

    def record_embed(self, ph_id: str, now: datetime) -> None:
        """Mark that we embedded for *ph_id* at *now*."""
        self._last_embed_at_by_ph[ph_id] = now


def _dist_m(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
