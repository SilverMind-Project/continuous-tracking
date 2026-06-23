"""Pydantic manifest model and JSON Schema for private identity replay datasets.

Only the schema definition is committed. Populated manifests referencing
household images must be listed in .gitignore and never committed.

Required replay cases (from M11 spec):
  - both persons visible with clear faces
  - both visible with one profile/back-facing
  - crossings and partial occlusion
  - seated/standing arrangement from confirmed incident
  - camera overlap and transitions
  - face absent, weak ReID, prior expiry
  - explicit Unknown frames
  - duplicate PH candidates
  - bad geometry/covariance and appearance outliers
  - operator correction and handoff boundary examples
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class Split(StrEnum):
    train = "train"
    val = "val"
    test = "test"


class BBox(BaseModel):
    """Bounding box in pixel coordinates (top-left origin)."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class FrameAnnotation(BaseModel):
    """Annotation for a single frame within an episode."""

    frame_index: int = Field(ge=0)
    camera_id: str
    # sha256 hex digest of the raw frame bytes; must not be empty.
    frame_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    identities: list[str] = Field(default_factory=list)
    bboxes: list[BBox] = Field(default_factory=list)
    # True when a person is visible but their identity is explicitly unknown.
    has_explicit_unknown: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def _bbox_identity_lengths_match(self) -> FrameAnnotation:
        if self.identities and self.bboxes and len(self.identities) != len(self.bboxes):
            raise ValueError(
                f"identities ({len(self.identities)}) and bboxes ({len(self.bboxes)}) "
                "must have the same length when both are provided"
            )
        return self


class Episode(BaseModel):
    """A contiguous sequence of frames forming one replay scenario."""

    episode_id: str = Field(min_length=1, max_length=64)
    description: str = ""
    split: Split
    # Identity IDs that appear in this episode (subset of manifest.identities).
    identity_ids: list[str] = Field(min_length=1)
    frames: list[FrameAnnotation] = Field(min_length=1)

    @field_validator("episode_id")
    @classmethod
    def _no_whitespace(cls, v: str) -> str:
        if any(c.isspace() for c in v):
            raise ValueError("episode_id must not contain whitespace")
        return v


class IdentityEntry(BaseModel):
    """Synthetic or real identity entry (use fictional names only in examples)."""

    identity_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    # sha256 of the enrollment embedding bytes (must not be empty for populated manifests).
    embedding_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


class ReplayManifest(BaseModel):
    """Root manifest for a private identity replay dataset.

    Commit only example manifests with fictional identities.
    Production manifests referencing household data must be git-ignored.
    """

    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+$")] = "1.0"
    dataset_name: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    identities: list[IdentityEntry] = Field(min_length=1)
    episodes: list[Episode] = Field(min_length=1)

    @model_validator(mode="after")
    def _identity_ids_unique(self) -> ReplayManifest:
        ids = [i.identity_id for i in self.identities]
        if len(ids) != len(set(ids)):
            dupes = [x for x in set(ids) if ids.count(x) > 1]
            raise ValueError(f"duplicate identity_ids: {dupes}")
        return self

    @model_validator(mode="after")
    def _episode_ids_unique(self) -> ReplayManifest:
        ids = [e.episode_id for e in self.episodes]
        if len(ids) != len(set(ids)):
            dupes = [x for x in set(ids) if ids.count(x) > 1]
            raise ValueError(f"duplicate episode_ids: {dupes}")
        return self

    @model_validator(mode="after")
    def _episode_identities_declared(self) -> ReplayManifest:
        known = {i.identity_id for i in self.identities}
        for ep in self.episodes:
            unknown = set(ep.identity_ids) - known
            if unknown:
                raise ValueError(
                    f"episode {ep.episode_id!r} references undeclared "
                    f"identity_ids: {sorted(unknown)}"
                )
        return self
