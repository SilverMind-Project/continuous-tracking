"""Domain types for the household appearance classifier and its training samples.

Freezes the multi-day body-appearance training sample record, the feature-extractor
seam (v1 SOLIDER-REID embeddings; future DINOv2 extractor via a new feature_version),
and the closed health/mode vocabularies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from app.domain import OrientationBin, ViewPrototype

HouseholdClassifierMode = Literal["off", "shadow", "enabled"]
HOUSEHOLD_CLASSIFIER_MODES: frozenset[str] = frozenset({"off", "shadow", "enabled"})


class AppearanceLabelSource(StrEnum):
    """Closed provenance vocabulary for multi-day body-appearance training samples."""

    WEARABLE_ATTESTED = "wearable_attested"
    DIRECT_FACE = "direct_face"
    OPERATOR_VERIFIED = "operator_verified"
    NEGATIVE = "negative"


class HouseholdClassifierHealthState(StrEnum):
    """Closed health-state vocabulary mirroring the face calibration evaluator."""

    OFF = "off"
    READY = "ready"
    DEGRADED_MISSING = "degraded_missing"
    DEGRADED_INCOMPATIBLE = "degraded_incompatible"
    DEGRADED_INVALID = "degraded_invalid"


@dataclass(frozen=True)
class AppearanceTrainingSample:
    """One labeled body-appearance embedding captured for offline classifier training.

    ``observed_at`` is the timezone-aware capture time of the observation.
    ``day`` is the calendar date in the household timezone (``app.timezone``),
    used by the offline trainer for strictly non-overlapping day splits.
    ``identity_id`` is ``None`` if and only if ``label_source == NEGATIVE``.
    """

    sample_id: str
    observed_at: datetime
    day: date
    identity_id: str | None
    label_source: AppearanceLabelSource
    camera_id: str
    orientation: OrientationBin
    quality: float
    feature_version: str
    embedding: tuple[float, ...]
    seed_association_id: str | None = None
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("AppearanceTrainingSample.observed_at must be timezone-aware")
        if self.revoked_at is not None and self.revoked_at.tzinfo is None:
            raise ValueError("AppearanceTrainingSample.revoked_at must be timezone-aware")
        if self.label_source == AppearanceLabelSource.NEGATIVE:
            if self.identity_id is not None:
                raise ValueError("negative appearance sample must have identity_id=None")
        elif not self.identity_id:
            raise ValueError("positive appearance sample must carry a non-empty identity_id")
        if not (0.0 <= self.quality <= 1.0) or not math.isfinite(self.quality):
            raise ValueError("AppearanceTrainingSample.quality must be finite in [0, 1]")
        if not self.feature_version:
            raise ValueError("AppearanceTrainingSample.feature_version must be non-empty")
        if not self.embedding or not all(math.isfinite(v) for v in self.embedding):
            raise ValueError("AppearanceTrainingSample.embedding must be non-empty and finite")


@dataclass(frozen=True)
class AppearanceFeature:
    """Normalized body-appearance feature vector extracted for runtime classification."""

    embedding: tuple[float, ...]
    orientation: OrientationBin
    feature_version: str
    quality: float


class AppearanceFeatureExtractor(Protocol):
    """Protocol separating body-embedding extraction from the classifier head.

    v1 reads existing SOLIDER-REID embeddings produced by ``InferenceStage`` and
    stored on PH view prototypes. Replacing SOLIDER with a DINOv2 Triton model
    implements this protocol under a new ``feature_version``, invalidating older
    classifier artifacts cleanly via version mismatch.
    """

    @property
    def feature_version(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    def extract(
        self,
        *,
        view_prototypes: Sequence[ViewPrototype],
        current_orientation: OrientationBin,
        observation_embedding: Sequence[float] | None,
        gallery_mean: Sequence[float] | None,
        quality: float,
    ) -> AppearanceFeature | None: ...
