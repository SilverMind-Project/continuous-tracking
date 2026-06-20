"""Domain types for the continuous tracking system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Spatial calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorPoint:
    """2D point in the shared floor-plan coordinate system, in millimetres.

    When ``calibrated=True`` the coordinates are in the shared floor-plan
    frame (a consistent coordinate system across all cameras that share
    the same ``floor_plan_id``).  When ``calibrated=False`` both ``x_mm``
    and ``y_mm`` are zero and the point must not be used for metric
    cross-camera comparisons.
    """

    x_mm: int
    y_mm: int
    calibrated: bool = False


@dataclass(frozen=True)
class CalibrationQuality:
    """Quality assessment for a stored camera homography."""

    max_residual_m: float
    mean_residual_m: float
    status: Literal["ok", "warning", "error"]
    point_count: int


@dataclass(frozen=True)
class CameraCalibration:
    """Per-camera spatial calibration with metadata."""

    camera_id: str
    floor_plan_id: str
    matrix: list[list[float]]
    image_width: int
    image_height: int
    quality: CalibrationQuality
    calibrated_at: datetime
    fov_deg: float | None = None


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

CameraId = str
DetectionId = str
TrackletId = str
GlobalTrackId = str
IdentityId = str
StreamId = str
TrackingEventId = str
RevisionId = str
GalleryEntryId = str
ActivityId = str
CorrectionId = str
PrivacyZoneId = str


# ---------------------------------------------------------------------------
# Protocol: what the IdentityResolver needs from any tracked entity
# ---------------------------------------------------------------------------


class IdentityResolvableEntity(Protocol):
    """What the IdentityResolver needs from any tracked thing (GT or PH)."""

    @property
    def entity_id(self) -> str: ...
    @property
    def observation_ids(self) -> list[str]: ...
    @property
    def camera_ids(self) -> list[str]: ...
    @property
    def current_identity_id(self) -> str | None: ...
    @property
    def current_identity_committed_at(self) -> datetime | None: ...
    @property
    def last_independent_identity_evidence_at(self) -> datetime | None: ...
    @property
    def last_seen_at(self) -> datetime: ...
    @property
    def started_at(self) -> datetime: ...
    @property
    def view_prototypes(self) -> tuple[ViewPrototype, ...]: ...


# ---------------------------------------------------------------------------
# Orientation and view prototypes
# ---------------------------------------------------------------------------


class OrientationBin(IntEnum):
    """Body orientation relative to the camera.

    IntEnum so it can be stored as a SMALLINT in the database.
    """

    FRONT = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class ViewPrototype:
    """One view-binned appearance prototype for a PersonHypothesis.

    Each prototype holds an EMA of SOLIDER-REID embeddings observed at a
    particular body orientation, plus a count of observations contributing
    to it.
    """

    orientation: OrientationBin
    embedding: tuple[float, ...]  # 768-dim L2-normalised float32 values
    count: int  # number of observations EMAd into this prototype


@dataclass(frozen=True)
class ObservationGeometry:
    """Per-observation view geometry, computed once and shared by all consumers.

    Pixel quantities are in raw image px. Residuals and distances are metres
    in the shared floor-plan frame.
    """

    footpoint_px: tuple[float, float]  # bbox bottom-centre, raw pixels
    floor_residual_m: float  # homography reprojection residual; 0.0 if unknown
    footpoint_reliable: bool  # feet visible and bbox not truncated
    detection_confidence: float  # detector confidence [0, 1]
    crop_quality: float  # CropQuality score [0, 1]
    orientation: OrientationBin  # body orientation from pose
    orientation_confidence: float  # orientation confidence [0, 1]
    distance_m: float | None = None  # camera-to-footpoint floor distance


def cov2x2_to_tuple(m: npt.NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Return a row-major 2x2 floor covariance tuple in m²."""
    return (float(m[0, 0]), float(m[0, 1]), float(m[1, 0]), float(m[1, 1]))


def tuple_to_cov2x2(t: tuple[float, float, float, float]) -> npt.NDArray[np.float64]:
    """Return a 2x2 floor covariance matrix in m² from a row-major tuple."""
    return np.array([[t[0], t[1]], [t[2], t[3]]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Person Hypothesis types
# ---------------------------------------------------------------------------

PersonHypothesisId = str
PHId = PersonHypothesisId  # shorthand used by IdentityRevision and N1 API
PH_MEAN_LEN = 4  # [x, y, vx, vy] in metres and metres/sec
PH_COV_LEN = 16  # 4x4 covariance matrix, row-major, in metres^2 and (metres/sec)^2


@dataclass(frozen=True)
class WorldObservation:
    """One camera-detection projected to floor coordinates.

    ``observation_id`` is assigned at persistence time by the repository.
    It is empty before the first save.
    """

    camera_id: str
    frame_index: int
    captured_at: datetime
    floor_point: FloorPoint
    bbox: BoundingBox
    embedding: list[float]
    detection_confidence: float
    observation_id: str = ""
    height_estimate_m: float | None = None
    face_anchor: FaceAnchor | None = None
    detection_id: str = ""
    quality: float = 0.0  # composite crop quality [0,1] from CropQuality.quality
    floor_residual_m: float | None = None
    orientation: OrientationBin = OrientationBin.UNKNOWN  # body orientation from pose keypoints
    orientation_confidence: float = 0.0  # confidence of orientation estimate [0,1]
    floor_cov_random: tuple[float, float, float, float] | None = None
    """Random part of the floor covariance J*Sigma_px*J^T, m^2, row-major 2x2.

    None for uncalibrated/synthetic points with no homography Jacobian.
    """
    footpoint_reliable: bool = True
    """False when feet are occluded or bbox truncation makes the floor contact low-trust."""
    primary_score: float = 0.0
    """Best-view score for stabilized primary-camera selection."""


@dataclass(frozen=True)
class IdentityDecisionGalleryHit:
    entry_id: str
    identity_id: str
    raw_similarity: float
    trust_multiplier: float
    recency_factor: float
    rank: int
    weighted_contribution: float
    source_episode_group: str | None = None
    orientation: str | None = None


@dataclass(frozen=True)
class IdentityEvidenceItem:
    score_type: str
    score_value: float
    source_identity_id: str | None = None
    quality: float | None = None
    camera_id: str | None = None
    timestamp: datetime | None = None
    model_version: str | None = None
    preprocessing_version: str | None = None
    calibration_version: str | None = None
    directness: str | None = None
    authoritative_eligibility: bool | None = None


@dataclass(frozen=True)
class IdentityProvenanceDecision:
    decision_id: str
    ph_id: str
    captured_at: datetime
    authority: str
    decision_source: str
    diagnostics: dict[str, object]
    observation_id: str | None = None
    inferred_identity_id: str | None = None
    effective_identity_id: str | None = None
    conflict_kind: str | None = None
    top_probability: float | None = None
    second_probability: float | None = None
    posterior_entropy: float | None = None
    last_independent_evidence_at: datetime | None = None
    config_hash: str | None = None
    resolver_version: str | None = None
    model_set_version: str | None = None
    diagnostics_schema_version: str | None = None
    evidence_items: list[IdentityEvidenceItem] = field(default_factory=list)
    gallery_hits: list[IdentityDecisionGalleryHit] = field(default_factory=list)


@dataclass(frozen=True)
class PersonHypothesis:
    """A persistent person track in world (floor-plane) coordinates.

    Replaces both ``Tracklet`` (per-camera) and ``GlobalTrack`` (cross-camera).
    There is one PH per physical person; observations from any camera update
    its Kalman state via a single Hungarian association per frame.
    """

    ph_id: PersonHypothesisId
    state_mean: tuple[float, float, float, float]  # [x, y, vx, vy] metres + m/s
    state_cov: tuple[float, ...]  # 16 floats, 4x4 row-major
    born_at: datetime
    last_seen_at: datetime
    last_seen_camera: str
    observation_count: int
    current_identity_id: str | None = None
    current_identity_committed_at: datetime | None = None
    last_independent_identity_evidence_at: datetime | None = None
    gallery_mean: list[float] | None = None  # L2-normalised SOLIDER embedding
    height_estimate_m: float | None = None
    active_cameras: frozenset[str] = frozenset()
    closed_at: datetime | None = None
    last_floor_speed_m_s: float = 0.0
    last_posture: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    mean_quality: float = 0.0  # EMA of observation quality scores
    view_prototypes: tuple[ViewPrototype, ...] = ()  # per-orientation appearance prototypes

    def __post_init__(self) -> None:
        if len(self.state_mean) != PH_MEAN_LEN:
            raise ValueError(f"state_mean must have {PH_MEAN_LEN} elements")
        if len(self.state_cov) != PH_COV_LEN:
            raise ValueError(f"state_cov must have {PH_COV_LEN} elements")

    # ---- IdentityResolvableEntity protocol (structural) ----

    @property
    def entity_id(self) -> str:
        return self.ph_id

    @property
    def observation_ids(self) -> list[str]:
        return []  # set by repository; list of recent observation UUIDs

    @property
    def camera_ids(self) -> list[str]:
        return list(self.active_cameras)

    @property
    def started_at(self) -> datetime:
        return self.born_at


@dataclass(frozen=True)
class PHContinuationCandidate:
    """Emitted when a newly-spawned PH might continue a recently-closed PH.

    Published to ``tracking.continuations`` for the inferred-presence consumer.
    """

    predecessor_ph_id: str
    successor_ph_id: str
    predecessor_closed_at: datetime
    successor_born_at: datetime
    distance_m: float
    seconds_elapsed: float
    predicted_drift_m: float
    predecessor_identity_id: str | None = None


@dataclass(frozen=True)
class WorldFrameSnapshot:
    """Per-PH view exposed to downstream stages after world tracking.

    Carries enough data for trajectory writer, posture, keyframes, and
    publish stages without coupling them to PH internals.
    """

    ph_id: str
    camera_id: str
    frame_index: int
    captured_at: datetime
    floor_x_m: float
    floor_y_m: float
    floor_vx_m_s: float
    floor_vy_m_s: float
    position_sigma_m: float
    contributing_camera_count: int = 1
    footpoint_reliable: bool = True
    identity_id: str | None = None
    identity_confidence: float = 0.0
    posterior_entropy: float = 0.0
    direct_face_evidence: bool = False
    bbox: BoundingBox | None = None
    detection_confidence: float = 0.0
    height_m: float | None = None
    room_id: str = ""
    room_name: str = ""
    mean_quality: float = 0.0  # PH-level rolling quality from quality capture (surfaces on wire)
    active_cameras: frozenset[str] = frozenset()  # all cameras on this PH (multi-camera fusion)
    floor_speed_m_s: float | None = None  # scalar Kalman speed; None for uncalibrated cameras
    evidence_json: str = "{}"
    inferred_identity_id: str = ""
    effective_identity_id: str = ""
    authority: str = ""
    decision_source: str = ""
    decision_id: str = ""
    conflict: str = ""
    last_independent_evidence_at_unix_ns: int = 0
    config_hash: str = ""
    model_set_version: str = ""


# ---------------------------------------------------------------------------
# Calibration correctness and transit zones
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HomographyValidation:
    """Result of server-side homography sanity checks."""

    ok: bool
    severity: str  # "ok" | "warning" | "error"
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraRoomBinding:
    """Immutable camera→room mapping from CC."""

    camera_id: str
    room_id: str
    room_name: str
    bound_at: datetime


@dataclass(frozen=True)
class TransitZone:
    """A door/threshold zone on the floor plan for entry/exit detection."""

    zone_id: str
    name: str
    kind: str  # "door" | "threshold"
    polygon: list[tuple[float, float]]  # normalized [0,1] floor-plan coords
    inside_room_id: str
    outside_room_id: str
    direction_vec: tuple[float, float]  # inside→outside direction


@dataclass(frozen=True)
class RoomTransitionEvent:
    """Emitted when a PH crosses a transit zone boundary."""

    ph_id: str
    transit_zone_id: str
    direction: str  # "enter" | "exit"
    inside_room_id: str
    outside_room_id: str
    floor_x_m: float
    floor_y_m: float
    event_time: datetime


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-coordinate bounding box, top-left origin."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0


@dataclass(frozen=True)
class Detection:
    """One person detected in a single frame."""

    detection_id: DetectionId
    camera_id: CameraId
    bbox: BoundingBox
    embedding: list[float]
    capture_time: datetime
    event_time: datetime
    confidence: float = 1.0
    tracklet_id: TrackletId = ""
    ph_id: PHId = ""
    floor_point: FloorPoint = field(default_factory=lambda: FloorPoint(0, 0))
    crop_quality: float = 0.0  # composite quality from CropQuality.quality
    floor_residual_m: float | None = None


# ---------------------------------------------------------------------------
# Frame / TrackingEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameRef:
    """Reference to a frame stored in object storage."""

    minio_key: str
    width: int
    height: int
    frame_index: int
    capture_time: datetime


@dataclass(frozen=True)
class TrackingEvent:
    """The domain-level output of processing one frame."""

    event_id: TrackingEventId
    camera_id: CameraId
    event_time: datetime
    frame_index: int
    frame_ref: FrameRef
    detections: list[Detection] = field(default_factory=list)
    identity_revisions: list[IdentityRevision] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracklet / GlobalTrack / Trajectory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryPoint:
    """A point in a tracked trajectory."""

    camera_id: CameraId
    observed_at: datetime
    floor_point: FloorPoint
    frame_index: int


@dataclass(frozen=True)
class Tracklet:
    """A short-lived trajectory within a single camera view."""

    tracklet_id: TrackletId
    camera_id: CameraId
    detection_ids: list[DetectionId]
    started_at: datetime
    ended_at: datetime | None = None
    state: Literal["active", "terminated"] = "active"
    # Last observed pixel bbox and its floor projection — in-memory only,
    # not persisted. Used by CrossCameraAssociator for geometric scoring.
    last_bbox: BoundingBox | None = None
    last_floor_point: FloorPoint | None = None


@dataclass(frozen=True)
class GlobalTrack:
    """A global track persists across cameras and time."""

    global_track_id: GlobalTrackId
    camera_ids: list[CameraId]
    tracklet_ids: list[TrackletId]
    started_at: datetime
    last_seen_at: datetime
    current_identity_id: IdentityId | None = None
    current_identity_committed_at: datetime | None = None
    last_independent_identity_evidence_at: datetime | None = None
    state: Literal["active", "closed"] = "active"
    last_posterior_jsonb: dict[str, Any] | None = None

    # ---- IdentityResolvableEntity protocol (structural) ----

    @property
    def entity_id(self) -> str:
        return self.global_track_id

    @property
    def observation_ids(self) -> list[str]:
        return list(self.tracklet_ids)

    @property
    def view_prototypes(self) -> tuple[ViewPrototype, ...]:
        return ()  # GlobalTrack has no view prototypes (legacy)


# ---------------------------------------------------------------------------
# Identity / Gallery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityCandidate:
    """One candidate in the Bayesian posterior over identities."""

    identity_id: IdentityId
    display_name: str
    probability: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be 0..1, got {self.probability}")


@dataclass(frozen=True)
class IdentityEvidence:
    """Snapshot of posterior distribution and evidence sources at revision time."""

    top_identity_id: str | None = None
    top_probability: float = 0.0
    second_probability: float = 0.0
    posterior_entropy: float = 0.0
    evidence_sources: list[str] = field(default_factory=list)
    observation_count: int = 0


@dataclass(frozen=True)
class IdentityRevision:
    """Record of an identity assignment change for a Person Hypothesis.

    Emitted when a PH's identity assignment is created, changed,
    or demoted.  Replaces the pre-N0 GlobalTrack-based revision
    which referenced ``global_track_id`` and ``tracklet_ids``.
    """

    revision_id: RevisionId
    ph_id: PHId
    previous_identity_id: IdentityId | None
    new_identity_id: IdentityId | None
    actor: str  # "system" | "resolver" | "user:<id>"
    reason: str
    applied_at: datetime
    rewritten_rows: int
    evidence: IdentityEvidence | None = None
    # -- M06 revision-range / projection metadata (typed proto fields 18-25) --
    revision_kind: str = ""
    range_start: datetime | None = None
    range_end: datetime | None = None
    range_authority: str = ""
    revision_range_id: str = ""
    correction_id: str = ""
    required_projections: tuple[str, ...] = ()
    revision_schema_version: str = ""


@dataclass(frozen=True)
class FaceAnchor:
    """A face-confirmed identity anchor from the face ID service.

    Face anchors are the strongest evidence source in the Bayesian
    posterior. They carry a person_id from the face recognition service
    along with confidence and quality metrics.

    In the PH-native pipeline, detection_id replaces tracklet_id
    as the primary per-detection key. tracklet_id is kept for backward
    compatibility with the legacy resolver evidence matching.
    """

    person_id: IdentityId
    confidence: float
    quality: float = 1.0
    tracklet_id: TrackletId = ""
    detection_id: str = ""
    camera_id: CameraId = ""
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # three-valued recognition state.
    recognition_state: str = "recognized"  # "recognized" | "candidate" | "unrecognized"
    # raw cosine similarity to best candidate.
    similarity: float = 0.0
    # head pose yaw in degrees (primary frontality axis).
    yaw_deg: float = 0.0
    # Calibrated ArcFace confidence from model calibration pipeline (M10).
    # None means no calibration available → authority fails closed.
    calibrated_confidence: float | None = None


@dataclass(frozen=True)
class PosteriorDist:
    """A probability distribution over identities for one GlobalTrack.

    Represents the posterior distribution after combining prior,
    face likelihood, and ReID likelihood. The sum of all probabilities
    should be close to 1.0.

    An empty distribution means "no evidence" — the caller should treat
    it as uniform over all candidates.
    """

    distribution: dict[str, float]  # identity_id -> probability

    def __post_init__(self) -> None:
        total = sum(self.distribution.values())
        if total <= 0 and self.distribution:
            raise ValueError("PosteriorDist must have positive total probability")
        # Normalize only if non-empty
        if self.distribution:
            object.__setattr__(
                self,
                "distribution",
                {k: v / total for k, v in self.distribution.items()},
            )

    def top_identity(self) -> tuple[str, float]:
        """Return (identity_id, probability) of the top candidate.

        Returns ('UNKNOWN', 0.0) for empty distributions.
        """
        if not self.distribution:
            return "UNKNOWN", 0.0
        top_id = max(self.distribution, key=self.distribution.__getitem__)
        return top_id, self.distribution[top_id]

    def top_with_margin(self) -> tuple[tuple[str, float], float]:
        """Return ((top_id, prob), margin_to_second).

        The margin is the difference between the top and second-highest
        probabilities. Used by the commit rule.

        Returns (('UNKNOWN', 0.0), 0.0) for empty distributions.
        """
        if not self.distribution:
            return ("UNKNOWN", 0.0), 0.0
        sorted_probs = sorted(self.distribution.items(), key=lambda x: x[1], reverse=True)
        top_id, top_prob = sorted_probs[0]
        margin = top_prob - sorted_probs[1][1] if len(sorted_probs) > 1 else 1.0
        return (top_id, top_prob), margin

    def entropy(self) -> float:
        """Compute Shannon entropy in bits.

        Returns 0.0 for empty distributions.
        """
        if not self.distribution:
            return 0.0
        import math

        return -sum(p * math.log2(p) for p in self.distribution.values() if p > 0)

    def has_identity(self, identity_id: IdentityId) -> bool:
        return identity_id in self.distribution


class ResolveOutcome:
    """The result of running identity resolution on a batch of GlobalTracks.

    Contains per-track decisions and any identity revisions that need
    to be emitted to downstream consumers.
    """

    def __init__(self) -> None:
        self.decisions: list[IdentityDecision] = []
        self.revisions: list[IdentityRevision] = []


@dataclass(frozen=True)
class IdentityDecision:
    """A single identity decision for one GlobalTrack."""

    ph_id: PHId
    identity_id: IdentityId | None
    posterior: PosteriorDist
    revises_previous: bool
    previous_identity_id: IdentityId | None = None
    reason: str = ""
    evidence_backed: bool = False
    evidence: dict[str, object] | None = None
    evidence_json: str = "{}"
    inferred_identity_id: str = ""
    effective_identity_id: str = ""
    authority: str = ""
    decision_source: str = ""
    decision_id: str = ""
    conflict: str = ""
    last_independent_evidence_at_unix_ns: int = 0
    config_hash: str = ""
    model_set_version: str = ""


@dataclass(frozen=True)
class Identity:
    """The opaque per-person identity record."""

    identity_id: IdentityId
    display_name: str
    enrolled_at: datetime
    height_mm: float | None = None
    height_sigma_mm: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass(frozen=True)
class GalleryEmbedding:
    """One embedding row in an identity's gallery."""

    gallery_entry_id: GalleryEntryId
    identity_id: IdentityId
    embedding: list[float]
    seen_at: datetime
    quality: float = 1.0
    origin_tracklet_id: TrackletId = ""
    face_confirmed: bool = False
    camera_id: str = ""
    orientation: int = OrientationBin.UNKNOWN  # OrientationBin stored as SMALLINT
    state: str = "pending_review"
    source_episode_id: str | None = None

@dataclass(frozen=True)
class IdentityCorrection:
    """Legacy GlobalTrack-era manual identity correction or merge decision.

    Retained for the vestigial ``misc.save_correction`` store. New operator
    corrections use :class:`IdentitySegmentCorrection` and the M06 revision-range
    model below.
    """

    correction_id: CorrectionId
    global_track_id: GlobalTrackId
    from_identity_id: IdentityId
    to_identity_id: IdentityId
    corrected_at: datetime
    corrected_by: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# M06: Segment correction, revision ranges, jobs, and effective projections
# ---------------------------------------------------------------------------

# Structured reason codes captured on every operator correction.
CorrectionReasonCode = Literal[
    "wrong_person",
    "identity_uncertain",
    "track_handoff",
    "duplicate_hypothesis",
    "bad_bbox",
    "other",
]

# Kind of correction operation. ``geometry`` shares the audit envelope but is a
# separate revision type; ``compensation`` is an undo.
CorrectionKind = Literal[
    "label",
    "frame_only",
    "handoff_split",
    "geometry",
    "compensation",
]

# Whether an effective range was set by an operator or by inference. Operator
# ranges are authoritative inside their bounds.
RevisionAuthority = Literal["operator", "inferred"]

RevisionJobStatus = Literal["pending", "applying", "completed", "failed"]

ProjectionAckStatus = Literal["acked", "failed"]


@dataclass(frozen=True)
class IdentitySegmentCorrection:
    """Authoritative, append-only operator correction over an observation range.

    Operator truth is immutable: the raw inference in ``identity_decisions`` is
    never edited. Applying a correction writes one or more
    :class:`IdentityRevisionRange` rows and an :class:`IdentityRevisionJob`.

    ``target_identity_id is None`` together with ``set_unknown`` expresses an
    explicit "Set to Unknown". An empty identity that is not ``set_unknown`` is
    rejected at the service boundary.
    """

    correction_id: CorrectionId
    ph_id: PHId
    actor: str
    reason_code: CorrectionReasonCode
    observation_start: datetime
    observation_end: datetime
    base_ph_version: int
    revision_id: RevisionId
    target_identity_id: IdentityId | None = None
    set_unknown: bool = False
    kind: CorrectionKind = "label"
    frame_only: bool = False
    note: str | None = None
    source_view: str | None = None
    reviewed_frame_id: str | None = None
    reviewed_bbox: dict[str, Any] | None = None
    base_revision_id: RevisionId | None = None
    compensates_correction_id: CorrectionId | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class IdentityRevisionRange:
    """Effective identity for a PH over an explicit time range.

    Operator ranges (``authority == 'operator'``) are authoritative inside their
    bounds and cannot be superseded by inferred ranges. A live range has
    ``superseded_by_range_id is None``.
    """

    range_id: str
    revision_id: RevisionId
    ph_id: PHId
    authority: RevisionAuthority
    range_start: datetime
    range_end: datetime
    effective_identity_id: IdentityId | None = None  # None == Unknown
    correction_id: CorrectionId | None = None
    supersedes_range_id: str | None = None
    superseded_by_range_id: str | None = None
    compensated_by_revision_id: RevisionId | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class IdentityRevisionJob:
    """Projection lifecycle for one revision.

    A correction is complete only after every entry in ``required_projections``
    acknowledges the same ``revision_id``. Retries are idempotent; an accepted
    correction is never rolled back to recover from a projection failure.
    """

    job_id: str
    revision_id: RevisionId
    status: RevisionJobStatus = "pending"
    required_projections: tuple[str, ...] = ()
    correction_id: CorrectionId | None = None
    attempts: int = 0
    last_error: str | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class ProjectionAck:
    """One consumer's acknowledgement that it applied a revision."""

    revision_id: RevisionId
    consumer: str
    schema_version: str
    status: ProjectionAckStatus = "acked"
    counts: dict[str, int] = field(default_factory=dict)
    applied_at: datetime | None = None


@dataclass(frozen=True)
class SegmentBoundary:
    """One boundary of a proposed segment, with the reason it was chosen."""

    observation_id: str
    captured_at: datetime
    reason: str  # association_discontinuity | identity_anchor | split | merge |
    #             operator_revision | segment_edge


@dataclass(frozen=True)
class SegmentProposal:
    """Advisory proposed correction segment for caregiver review.

    The proposal is advisory only; applying requires an explicit confirmed
    start/end. ``ph_version`` is the optimistic token; a stale token forces
    recompute (HTTP 409).
    """

    ph_id: PHId
    observation_ids: list[str]
    start: SegmentBoundary
    end: SegmentBoundary
    ph_version: int
    effective_identity_id: IdentityId | None = None


PrivacyPolicy = Literal["drop_detection", "blur_region", "mask_region"]


@dataclass(frozen=True)
class PrivacyZone:
    """Operator-defined privacy mask for a camera."""

    zone_id: PrivacyZoneId
    camera_id: CameraId
    name: str
    polygon: list[tuple[int, int]]
    policy: PrivacyPolicy = "drop_detection"
    enabled: bool = True


# ---------------------------------------------------------------------------
# Camera / Stream Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a physical camera."""

    camera_id: CameraId
    name: str = ""
    rtsp_url: str = ""
    location: str = ""
    resolution_width: int = 1920
    resolution_height: int = 1080
    floor_plan: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass(frozen=True)
class OverlapGroup:
    """A group of cameras that share a physical field of view."""

    group_id: str
    name: str = ""
    camera_ids: tuple[str, ...] = ()
    source: str = "operator"  # "operator" | "learned"


# ---------------------------------------------------------------------------
# Camera adjacency topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraTopologyEdge:
    """Learned transit-time statistics for a directed camera pair.

    Stores a running count, mean, and variance of observed handoff transit
    times, updated online via the Welford algorithm. The topology model
    service (``app/tracking/world/topology.py``) computes a plausibility
    score from these statistics.
    """

    from_camera: str
    to_camera: str
    observation_count: int = 0
    mean_transit_s: float = 0.0
    variance_transit_s2: float = 0.0
    last_updated_at: datetime | None = None


@dataclass(frozen=True)
class CoPresenceLink:
    """Identity-level co-presence link between two Person Hypotheses.

    Written when two open PHs in the same overlap group share a committed
    identity.  The CHECK constraint ph_id_a < ph_id_b prevents duplicate
    directional pairs in storage.
    """

    id: str
    group_id: str
    ph_id_a: str
    ph_id_b: str
    identity_id: str
    first_observed_at: datetime
    last_observed_at: datetime
    observation_count: int = 1


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for a logical processing stream."""

    stream_id: StreamId
    camera_id: CameraId
    frame_rate: float = 5.0
    resolution_width: int = 640
    resolution_height: int = 480
    is_active: bool = True


# ---------------------------------------------------------------------------
# Stream Assignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamAssignment:
    """Maps a stream to a room and zone for spatial tracking."""

    stream_id: StreamId
    room_id: str = ""
    zone_id: str = ""


# ---------------------------------------------------------------------------
# Dementia Activity Layer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Trajectory, dwell, and keyframe types
# ---------------------------------------------------------------------------

PostureType = Literal["standing", "sitting", "walking", "lying", "unknown"]
TagReason = Literal["periodic", "identity_changed", "hazard", "dwell_start"]


@dataclass(frozen=True)
class PersonTrajectoryPoint:
    """One confirmed ground-plane position row in person_trajectories.

    ``identity_id`` is nullable: when the Bayesian resolver has not yet
    committed an identity, the trajectory point is still written with
    ``identity_id=None`` so the track remains visible on dashboards and
    can be retroactively labelled when identity is resolved (Phase 5).
    """

    identity_id: IdentityId | None
    ph_id: PHId
    observed_at: datetime
    room_name: str = ""
    ground_x: float = 0.0  # meters, floor-plan frame
    ground_y: float = 0.0  # meters, floor-plan frame
    posture: PostureType = "unknown"
    identity_confidence: float = 0.0
    # sqrt of larger eigenvalue of PH position covariance, in metres
    position_sigma_m: float = 0.0
    # stabilized best-view camera selected by the world tracker
    primary_camera_id: str = ""
    # number of camera observations fused into this point for the frame
    contributing_camera_count: int = 1
    # representative observation footpoint reliability for this frame
    footpoint_reliable: bool = True
    # mean keypoint velocity at this point; None when pose unavailable
    motion_energy: float | None = None
    # Kalman floor speed in m/s; None when camera is uncalibrated
    floor_speed_m_s: float | None = None


@dataclass(frozen=True)
class RoomDwell:
    """Contiguous interval a person spent in one room."""

    dwell_id: str
    identity_id: IdentityId | None
    ph_id: PHId
    room_name: str
    entered_at: datetime
    exited_at: datetime | None = None
    duration_seconds: int | None = None
    entry_confidence: float = 0.0
    primary_posture: PostureType = "unknown"
    min_motion_energy: float | None = None  # lowest motion energy observed during the dwell
    still_seconds: int = 0  # accumulated contiguous low-motion time within the dwell
    activity_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaggedKeyframe:
    """A periodic or triggered frame sample tagged with tracking annotations."""

    keyframe_id: str
    ph_id: PHId
    camera_id: CameraId
    minio_key: str
    captured_at: datetime
    annotations: dict[str, Any]  # bbox, person_id, posture, activity, confidence
    tag_reason: TagReason
    expires_at: datetime


@dataclass(frozen=True)
class BboxAnnotation:
    """YOLO bounding box for one tracked person in one keyframe."""

    keyframe_id: str  # FK to tagged_keyframes.id
    ph_id: str
    camera_id: str
    x1: float  # pixels, top-left x, in original frame resolution
    y1: float  # pixels, top-left y
    x2: float  # pixels, bottom-right x
    y2: float  # pixels, bottom-right y
    detection_confidence: float
    frame_width: int  # original frame width (needed for normalisation in frontend)
    frame_height: int  # original frame height
    identity_id: str | None = None  # None if not yet resolved
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: str | None = None  # DB-generated UUID; None for new annotations before persist
    # Frames since the contributing detection. 0 = same-frame.
    bbox_age_frames: int = 0
    # User-drawn override bbox
    override_x1: float | None = None
    override_y1: float | None = None
    override_x2: float | None = None
    override_y2: float | None = None
    override_by: str | None = None
    override_at: datetime | None = None


ActivityType = Literal[
    "entry",
    "exit",
    "linger",
    "loop",
    "fall_detected",
    "area_entered",
    "area_exited",
    "pacing",
    "sundowning",
    "bathroom_anomaly",
    "stillness",
    "nighttime_movement",
    "absence",
]


@dataclass(frozen=True)
class PersonActivity:
    """A dementia-relevant activity record."""

    activity_id: ActivityId
    identity_id: IdentityId
    camera_id: CameraId
    activity_type: ActivityType
    occurred_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    related_event_id: TrackingEventId = ""


# ---------------------------------------------------------------------------
# Dementia Signal types
# ---------------------------------------------------------------------------

DementiaSignalKind = Literal[
    "pacing",
    "bathroom_dwell_anomaly",
    "sundowning_index",
    "nighttime_movement",
    "stillness_anomaly",
    "absence",
    "inferred_dwell_exceeded",
    "presumed_location_unknown",
    "identity_disagreement",
    "fall_suspected",
    "gait_slowing",
    "agitation_index",
]

DementiaSignalSeverity = Literal["info", "warning", "emergency"]


@dataclass(frozen=True)
class DementiaSignal:
    """A computed dementia-relevant signal.

    Emitted periodically by :class:`DementiaSignalWorker`.  Each signal
    covers a time window and carries a severity and z-score relative to
    the person's historical baseline.
    """

    signal_id: str
    identity_id: IdentityId
    signal_kind: DementiaSignalKind
    severity: DementiaSignalSeverity
    value: float
    baseline: float | None = None
    z_score: float | None = None
    window_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    window_end: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: dict[str, Any] = field(default_factory=dict)
    emitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    algorithm_version: int = 1  # incremented when detector logic changes
    algorithm_name: str = ""  # human-readable name for clinical documentation
    evidence_grade: str = (
        # clinical_review | observational_study | caregiver_guidance
        # | local_baseline_only | experimental
        ""
    )
    algorithm_spec_json: str = ""  # JSON-serialized algorithm specification


@dataclass(frozen=True)
class Keyframe:
    """A MinIO-stored keyframe linked to a world observation or PH."""

    observation_id: str
    observed_at: datetime
    camera_id: str
    minio_key: str
    floor_x_mm: float | None = None
    floor_y_mm: float | None = None
    pose_class: str | None = None
    reid_confidence: float | None = None
