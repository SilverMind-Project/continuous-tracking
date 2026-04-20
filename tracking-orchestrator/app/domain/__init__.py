"""Domain types for the continuous tracking system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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


@dataclass(frozen=True)
class GlobalTrack:
    """A global track persists across cameras and time."""

    global_track_id: GlobalTrackId
    camera_ids: list[CameraId]
    tracklet_ids: list[TrackletId]
    started_at: datetime
    last_seen_at: datetime
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
    """Bayesian posterior update for a global track at a point in time."""

    revision_id: RevisionId
    global_track_id: GlobalTrackId
    candidates: list[IdentityCandidate]
    map_identity_id: IdentityId
    posterior_entropy: float
    revision_time: datetime


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


@dataclass(frozen=True)
class PrivacyZone:
    """Operator-defined privacy mask for a camera."""

    zone_id: PrivacyZoneId
    camera_id: CameraId
    name: str
    polygon: list[tuple[int, int]]
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
