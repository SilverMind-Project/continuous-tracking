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
        FloorPoint,
        GlobalTrack,
        IdentityDecision,
        IdentityRevision,
        PostureType,
        WorldFrameSnapshot,
    )
    from ..inference.evidence import (
        AppearanceEvidence,
        FaceEvidence,
        PersonDetectionEvidence,
        PoseEvidence,
    )
    from ..inference.schemas import DetectionBox, Embedding, PoseResult
    from ..trajectory.posture import PostureScores
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
    # Stable detection IDs assigned per kept box (idx → detection_id).
    _detection_ids: dict[int, str] = field(default_factory=dict)

    # --- Stage: spatial_projection (pre-computed floor points, keyed by
    # index into raw_detections; consumed by inference stage) ---
    _floor_points_by_index: dict[int, FloorPoint] = field(default_factory=dict)

    # --- Stage: inference (build domain detections + ReID + pose) ---
    domain_detections: list[Detection] = field(default_factory=list)
    crops: list[npt.NDArray[np.uint8]] = field(default_factory=list)
    embeddings: list[Embedding] = field(default_factory=list)
    det_pose_result: dict[str, PoseResult] = field(default_factory=dict)
    # Typed evidence produced by inference adapters.
    _detection_evidence: dict[int, PersonDetectionEvidence] = field(default_factory=dict)
    _appearance_evidence: list[AppearanceEvidence] = field(default_factory=list)
    _pose_evidence: list[PoseEvidence] = field(default_factory=list)

    # --- Stage: face_id ---
    face_anchors: list[FaceAnchor] = field(default_factory=list)
    _face_evidence: list[FaceEvidence] = field(default_factory=list)

    # --- Stage: cross_camera_and_resolve (legacy: replaced by world_tracking) ---
    # DEPRECATED (WTR3): populated from PH ids for backward compat during
    # transition.  New stages must use ``active_ph_ids`` instead.
    active_global_tracks: list[GlobalTrack] = field(default_factory=list)
    outcome_decisions: list[IdentityDecision] = field(default_factory=list)
    new_revisions: list[IdentityRevision] = field(default_factory=list)
    committed_ids: dict[str, str | None] = field(default_factory=dict)

    # --- Stage: posture ---
    det_posture_scores: dict[str, PostureScores] = field(default_factory=dict)

    # --- Stage: trajectory ---
    det_posture: dict[str, PostureType] = field(default_factory=dict)

    # --- Stage: world_tracking (M1) ---
    # Producer: WorldTrackingStage. Consumers: ClosePHStage, TrajectoryStage,
    # KeyframeStage, PublishStage, TrailsStage.
    world_snapshots: list[WorldFrameSnapshot] = field(default_factory=list)
    # Producer: WorldTrackingStage. Consumer: DetectionBackfillStage.
    det_to_ph: dict[str, str] = field(default_factory=dict)
    # Producer: WorldTrackingStage. Consumers: ClosePHStage.
    active_ph_ids: set[str] = field(default_factory=set)

    # --- Stage: publish ---
    identities: dict[str, tuple[str, float]] = field(default_factory=dict)
    evidence_by_gt: dict[str, tuple[float, float, bool]] = field(default_factory=dict)
    trail_by_tracklet_snapshot: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    # WTR3: PH-native trail alias (TrailsStage populates both).
    trail_by_ph_snapshot: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    # --- Diagnostics (CI / observability, not business logic) ---
    stage_notes: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def require_image(self) -> npt.NDArray[np.uint8]:
        """Return ``image``, asserting it has been set by the fetch stage."""
        if self.image is None:
            raise RuntimeError("FrameContext.image is None — fetch stage must run first")
        return self.image
