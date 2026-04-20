"""Domain types for the continuous tracking system.

All types are frozen dataclasses to enforce immutability within the core.
They should never be serialized to or deserialized from directly at
boundaries — use Pydantic models at the edges.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

CameraId = str
DetectionId = str
TrackletId = str
GlobalTrackId = str
IdentityId = str
StreamId = str


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
    detection_id: DetectionId = field(default_factory=lambda: DetectionId(uuid.uuid4()))
    camera_id: CameraId = ""
    bbox: BoundingBox = field(default_factory=BoundingBox)
    embedding: list[float] = field(default_factory=list)
    confidence: float = 1.0
    tracklet_id: TrackletId = ""
    global_track_id: GlobalTrackId = ""
    floor_point: FloorPoint = field(default_factory=FloorPoint)
    capture_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Tracklet / GlobalTrack
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tracklet:
    """A tracklet is a short-lived trajectory within a single camera view.
    It may span multiple frames but is broken by occlusion, exit, etc.
    """
    tracklet_id: TrackletId = field(default_factory=lambda: TrackletId(uuid.uuid4()))
    camera_id: CameraId = ""
    detection_ids: list[DetectionId] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    state: Literal["active", "terminated"] = "active"


@dataclass(frozen=True)
class GlobalTrack:
    """A global track persists across cameras and time.
    It is the entity that identity resolution operates on.
    """
    global_track_id: GlobalTrackId = field(default_factory=lambda: GlobalTrackId(uuid.uuid4()))
    camera_ids: list[CameraId] = field(default_factory=list)
    tracklet_ids: list[TrackletId] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: Literal["active", "closed"] = "active"


# ---------------------------------------------------------------------------
# Identity
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
    global_track_id: GlobalTrackId
    candidates: list[IdentityCandidate]
    map_identity_id: IdentityId
    posterior_entropy: float
    revision_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class GalleryEntry:
    """A known person's gallery record (enrolled identity)."""
    identity_id: IdentityId = field(default_factory=lambda: IdentityId(uuid.uuid4()))
    display_name: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    enrolled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


# ---------------------------------------------------------------------------
# TrackingEvent (domain-level, the result of frame processing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackingEvent:
    """The domain-level output of processing one frame.
    Mirrors the protobuf message but with Python types.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: CameraId = ""
    event_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frame_index: int = 0
    detections: list[Detection] = field(default_factory=list)
    identity_revisions: list[IdentityRevision] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Camera / Stream Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a physical camera."""
    camera_id: CameraId
    name: str = ""
    rtsp_url: str = ""
    location: str = ""  # e.g. "hallway-east"
    floor_plan: dict = field(default_factory=dict)
    is_active: bool = True


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for a logical processing stream (derived from a camera)."""
    stream_id: StreamId
    camera_id: CameraId
    frame_rate: float = 5.0
    resolution_width: int = 640
    resolution_height: int = 480
    is_active: bool = True


# ---------------------------------------------------------------------------
# Dementia Activity Layer
# ---------------------------------------------------------------------------

ActivityType = Literal[
    "entry", "exit", "linger", "loop", "fall_detected", "area_entered", "area_exited",
]


@dataclass(frozen=True)
class PersonActivity:
    """A dementia-relevant activity record, written by the activity layer."""
    activity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    identity_id: IdentityId = ""
    camera_id: CameraId = ""
    activity_type: ActivityType = "entry"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    confidence: float = 1.0
    related_event_id: str = ""
