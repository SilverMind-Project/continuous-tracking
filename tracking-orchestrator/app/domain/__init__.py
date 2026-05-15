"""Domain types for the continuous tracking system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

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
class FloorPoint:
    """2D ground-plane point in millimeters."""

    x_mm: int
    y_mm: int
    calibrated: bool = False


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
    global_track_id: GlobalTrackId = ""
    floor_point: FloorPoint = field(default_factory=lambda: FloorPoint(0, 0))


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
    state: Literal["active", "closed"] = "active"


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
class IdentityRevision:
    """Bayesian posterior update for a global track at a point in time.

    Emitted when a GlobalTrack's identity assignment changes. Covers all
    tracklets within the revision horizon that were part of the same
    GlobalTrack at the time of the decision.
    """

    revision_id: RevisionId
    global_track_id: GlobalTrackId
    tracklet_ids: list[TrackletId]
    candidates: list[IdentityCandidate]
    map_identity_id: IdentityId
    posterior_entropy: float
    previous_identity_id: IdentityId | None = None
    new_identity_id: IdentityId | None = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    revision_time: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class FaceAnchor:
    """A face-confirmed identity anchor from the face ID service.

    Face anchors are the strongest evidence source in the Bayesian
    posterior. They carry a person_id from the face recognition service
    along with confidence and quality metrics.
    """

    person_id: IdentityId
    confidence: float
    quality: float = 1.0
    tracklet_id: TrackletId = ""
    camera_id: CameraId = ""
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))


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

    global_track_id: GlobalTrackId
    identity_id: IdentityId | None
    posterior: PosteriorDist
    revises_previous: bool
    previous_identity_id: IdentityId | None = None
    reason: str = ""


@dataclass(frozen=True)
class Identity:
    """The opaque per-person identity record."""

    identity_id: IdentityId
    display_name: str
    enrolled_at: datetime
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


@dataclass(frozen=True)
class IdentityCorrection:
    """Manual identity correction or merge decision."""

    correction_id: CorrectionId
    global_track_id: GlobalTrackId
    from_identity_id: IdentityId
    to_identity_id: IdentityId
    corrected_at: datetime
    corrected_by: str = ""
    reason: str = ""


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
# M6: Trajectory, dwell, and keyframe types
# ---------------------------------------------------------------------------

PostureType = Literal["standing", "sitting", "walking", "lying", "unknown"]
TagReason = Literal["periodic", "identity_changed", "hazard", "dwell_start"]


@dataclass(frozen=True)
class PersonTrajectoryPoint:
    """One confirmed ground-plane position row in person_trajectories."""

    identity_id: IdentityId
    global_track_id: GlobalTrackId
    observed_at: datetime
    room_name: str = ""
    ground_x: float = 0.0  # meters, floor-plan frame
    ground_y: float = 0.0  # meters, floor-plan frame
    posture: PostureType = "unknown"
    identity_confidence: float = 0.0
    # mean keypoint velocity at this point; None when pose unavailable
    motion_energy: float | None = None


@dataclass(frozen=True)
class RoomDwell:
    """Contiguous interval a person spent in one room."""

    dwell_id: str
    identity_id: IdentityId
    global_track_id: GlobalTrackId
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
    tracklet_id: TrackletId
    global_track_id: GlobalTrackId
    camera_id: CameraId
    minio_key: str
    captured_at: datetime
    annotations: dict[str, Any]  # bbox, person_id, posture, activity, confidence
    tag_reason: TagReason
    expires_at: datetime


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
