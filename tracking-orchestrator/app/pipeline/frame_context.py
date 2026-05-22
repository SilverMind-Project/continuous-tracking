"""Per-frame state threaded through pipeline stages.

Each stage reads from and writes to specific fields as documented.
Fields initialized as empty/sentinel are populated by earlier stages
and consumed by later ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..domain import (
        Detection,
        FaceAnchor,
        GlobalTrack,
        IdentityDecision,
        IdentityRevision,
        PostureType,
        Tracklet,
    )
    from ..inference.schemas import DetectionBox, Embedding, PoseResult
    from ..transport.redis_streams import FrameReady


@dataclass
class FrameContext:
    """Per-frame state threaded through pipeline stages.

    Each stage reads from and writes to specific fields as documented.
    Fields initialized as empty/sentinel are populated by earlier stages
    and consumed by later ones.
    """

    # --- Input (set once by _stage_init) ---
    frame: FrameReady
    event_time: datetime
    capture_time: datetime

    # --- Stage: fetch ---
    image: npt.NDArray[np.uint8] | None = None
    effective_width: int = 0
    effective_height: int = 0

    # --- Stage: detect ---
    raw_detections: list[DetectionBox] = field(default_factory=list)

    # --- Stage: embed (build domain detections + ReID + pose) ---
    domain_detections: list[Detection] = field(default_factory=list)
    crops: list[npt.NDArray[np.uint8]] = field(default_factory=list)
    embeddings: list[Embedding] = field(default_factory=list)
    det_pose_result: dict[str, PoseResult] = field(default_factory=dict)

    # --- Stage: face_id ---
    face_anchors: list[FaceAnchor] = field(default_factory=list)

    # --- Stage: tracklet_manage (populated by tracklet_manager) ---
    active_tracklets: list[Tracklet] = field(default_factory=list)
    # Height estimates per tracklet (tracklet_id -> height_mm).
    tracklet_heights: dict[str, float] = field(default_factory=dict)

    # --- Stage: cross_camera_and_resolve ---
    active_global_tracks: list[GlobalTrack] = field(default_factory=list)
    outcome_decisions: list[IdentityDecision] = field(default_factory=list)
    new_revisions: list[IdentityRevision] = field(default_factory=list)
    committed_ids: dict[str, str | None] = field(default_factory=dict)

    # --- Stage: trajectory ---
    det_posture: dict[str, PostureType] = field(default_factory=dict)

    # --- Stage: publish ---
    identities: dict[str, tuple[str, float]] = field(default_factory=dict)
    evidence_by_gt: dict[str, tuple[float, float, bool]] = field(default_factory=dict)
    trail_by_tracklet_snapshot: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
