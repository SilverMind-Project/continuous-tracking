"""Pydantic v2 schemas for the Person Hypothesis API (N1).

Every response model has an explicit ``from_domain()`` classmethod
so the API surface is deliberate and privacy-first (universal rule 16).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..domain import IdentityRevision, PersonHypothesis, WorldObservation

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class ObservationResponse(BaseModel):
    observation_id: str = ""
    camera_id: str = ""
    frame_index: int = 0
    captured_at: datetime | None = None
    floor_x_m: float = 0.0
    floor_y_m: float = 0.0
    detection_confidence: float = 0.0
    height_m: float | None = None

    @classmethod
    def from_domain(cls, obs: WorldObservation) -> ObservationResponse:
        return cls(
            camera_id=obs.camera_id,
            frame_index=obs.frame_index,
            captured_at=obs.captured_at,
            floor_x_m=obs.floor_point.x_mm / 1000.0,
            floor_y_m=obs.floor_point.y_mm / 1000.0,
            detection_confidence=obs.detection_confidence,
            height_m=obs.height_estimate_m,
        )


# ---------------------------------------------------------------------------
# PH summary (list item)
# ---------------------------------------------------------------------------


class PHSummary(BaseModel):
    ph_id: str
    born_at: datetime | None = None
    last_seen_at: datetime | None = None
    closed_at: datetime | None = None
    observation_count: int = 0
    current_identity_id: str | None = None
    active_cameras: list[str] = Field(default_factory=list)
    last_floor_speed_m_s: float = 0.0
    last_posture: str | None = None

    @classmethod
    def from_domain(cls, ph: PersonHypothesis) -> PHSummary:
        return cls(
            ph_id=ph.ph_id,
            born_at=ph.born_at,
            last_seen_at=ph.last_seen_at,
            closed_at=ph.closed_at,
            observation_count=ph.observation_count,
            current_identity_id=ph.current_identity_id,
            active_cameras=list(ph.active_cameras),
            last_floor_speed_m_s=ph.last_floor_speed_m_s,
            last_posture=ph.last_posture,
        )


# ---------------------------------------------------------------------------
# PH detail
# ---------------------------------------------------------------------------


class PHDetail(BaseModel):
    ph_id: str
    born_at: datetime | None = None
    last_seen_at: datetime | None = None
    closed_at: datetime | None = None
    observation_count: int = 0
    current_identity_id: str | None = None
    current_identity_committed_at: datetime | None = None
    active_cameras: list[str] = Field(default_factory=list)
    last_seen_camera: str = ""
    last_floor_speed_m_s: float = 0.0
    last_posture: str | None = None
    height_estimate_m: float | None = None
    state_mean: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, ph: PersonHypothesis) -> PHDetail:
        return cls(
            ph_id=ph.ph_id,
            born_at=ph.born_at,
            last_seen_at=ph.last_seen_at,
            closed_at=ph.closed_at,
            observation_count=ph.observation_count,
            current_identity_id=ph.current_identity_id,
            current_identity_committed_at=ph.current_identity_committed_at,
            active_cameras=list(ph.active_cameras),
            last_seen_camera=ph.last_seen_camera,
            last_floor_speed_m_s=ph.last_floor_speed_m_s,
            last_posture=ph.last_posture,
            height_estimate_m=ph.height_estimate_m,
            state_mean=list(ph.state_mean),
            metadata=dict(ph.metadata),
        )


# ---------------------------------------------------------------------------
# Trail point
# ---------------------------------------------------------------------------


class TrailPointResponse(BaseModel):
    camera_id: str = ""
    floor_x_m: float = 0.0
    floor_y_m: float = 0.0
    captured_at: datetime | None = None

    @classmethod
    def from_observation(cls, obs: WorldObservation) -> TrailPointResponse:
        return cls(
            camera_id=obs.camera_id,
            floor_x_m=obs.floor_point.x_mm / 1000.0,
            floor_y_m=obs.floor_point.y_mm / 1000.0,
            captured_at=obs.captured_at,
        )


# ---------------------------------------------------------------------------
# Revision
# ---------------------------------------------------------------------------


class RevisionResponse(BaseModel):
    revision_id: str
    ph_id: str
    previous_identity_id: str | None = None
    new_identity_id: str | None = None
    actor: str = ""
    reason: str = ""
    kind: str = ""
    applied_at: datetime | None = None
    rewritten_rows: int = 0

    @classmethod
    def from_domain(cls, rev: IdentityRevision, kind: str = "auto") -> RevisionResponse:
        return cls(
            revision_id=rev.revision_id,
            ph_id=rev.ph_id,
            previous_identity_id=rev.previous_identity_id,
            new_identity_id=rev.new_identity_id,
            actor=rev.actor,
            reason=rev.reason,
            kind=kind,
            applied_at=rev.applied_at,
            rewritten_rows=rev.rewritten_rows,
        )


# ---------------------------------------------------------------------------
# Paginated response wrapper
# ---------------------------------------------------------------------------


class PaginatedPHList(BaseModel):
    items: list[PHSummary]
    total: int
    limit: int
    offset: int


class PHObservationsList(BaseModel):
    ph_id: str
    items: list[ObservationResponse]
    count: int


# ---------------------------------------------------------------------------
# Correction request bodies
# ---------------------------------------------------------------------------


class CorrectIdentityRequest(BaseModel):
    new_identity_id: str | None = Field(default=None, max_length=128)
    reason: str = Field(default="manual", max_length=512)


class MergeRequest(BaseModel):
    source_ph_id: str = Field(..., min_length=1, max_length=128)
    target_ph_id: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(default="manual", max_length=512)


class SplitRequest(BaseModel):
    at_observation_id: str = Field(..., min_length=1, max_length=256)
    reason: str = Field(default="manual", max_length=512)


class BatchCorrectItem(BaseModel):
    ph_id: str = Field(..., min_length=1, max_length=128)
    new_identity_id: str | None = Field(default=None, max_length=128)
    reason: str = Field(default="manual", max_length=512)


class BatchCorrectRequest(BaseModel):
    corrections: list[BatchCorrectItem] = Field(..., min_length=1, max_length=50)


# ---------------------------------------------------------------------------
# Correction response bodies
# ---------------------------------------------------------------------------


class CorrectIdentityResponse(BaseModel):
    revision: RevisionResponse


class MergeResponse(BaseModel):
    revision: RevisionResponse
    source_ph_id: str
    target_ph_id: str


class SplitResponse(BaseModel):
    original_ph_id: str
    new_ph_id: str


class BatchCorrectResponse(BaseModel):
    revisions: list[RevisionResponse]
    applied: int
    errors: list[dict[str, str]] = Field(default_factory=list)


class RevisionsFeedResponse(BaseModel):
    items: list[RevisionResponse]
    has_more: bool
