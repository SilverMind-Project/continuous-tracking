"""Runtime household appearance classifier evaluator, feature extractor, and evidence channel.

Evaluates an offline-trained, versioned JSON multinomial logistic regression head on
frozen L2-normalized body embeddings using NumPy only (never scikit-learn or pickle).
The channel corroborates identity across days and outfits without ever granting
authority, refreshing the independent-evidence clock, ranking in the duplicate guard,
or entering world association.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from structlog import get_logger

from app.domain import (
    HOUSEHOLD_CLASSIFIER_MODES,
    AppearanceFeature,
    HouseholdClassifierHealthState,
    HouseholdClassifierMode,
    IdentityResolvableEntity,
    OrientationBin,
    PosteriorDist,
    ViewPrototype,
)
from app.observability import metrics as _metrics
from app.observability.metrics import HouseholdClassifierHealthLabel
from app.tracking.identity.channels import ChannelLikelihood, ResolverInputs
from app.tracking.identity.evidence_grades import HOUSEHOLD_CLASSIFIER

logger = get_logger(__name__)

_DEFAULT_EMBEDDING_DIM = 768
_NORM_EPS = 1e-8


@dataclass(frozen=True)
class HouseholdClassifierConfig:
    """Configuration for the household appearance classifier and sample capture.

    Dimensional fields declare their units (``_s`` seconds).
    """

    # Rollout mode: "off" skips evaluation and health checks report off; "shadow" evaluates
    # counterfactual outcomes and the Part E classifier arm without altering live decisions;
    # "enabled" includes the classifier channel in the live Bayesian posterior.
    mode: HouseholdClassifierMode = "off"

    # Separate opt-in gate for recording multi-day training embeddings to Postgres.
    capture_enabled: bool = False

    # Filesystem path to the versioned JSON classifier artifact (empty means unconfigured).
    artifact_path: str = ""

    # Pre-softmax logit scale factor applied as (weight_multiplier / temperature) before softmax
    # with support_multiplier = 1.0 so the weight does not cancel on normalization.
    weight_multiplier: float = 1.0

    # Minimum UNKNOWN probability mass reserved in the channel distribution for open-set safety.
    unknown_floor: float = 0.2

    # Calibrated class probability floor required before a class counts as commit evidence.
    commit_evidence_min_probability: float = 0.8

    # Minimum crop quality [0, 1] required to record a training sample.
    min_quality: float = 0.35

    # Per-(PH, orientation) capture-time deduplication interval in seconds.
    dedup_window_s: float = 10.0

    # Daily cap on recorded open-set negative training samples.
    max_negatives_per_day: int = 500

    def __post_init__(self) -> None:
        if self.mode not in HOUSEHOLD_CLASSIFIER_MODES:
            raise ValueError(
                f"household_classifier.mode must be one of {HOUSEHOLD_CLASSIFIER_MODES}, "
                f"got {self.mode!r}"
            )
        if not math.isfinite(self.weight_multiplier) or self.weight_multiplier <= 0.0:
            raise ValueError("household_classifier.weight_multiplier must be positive and finite")
        if not (0.0 < self.unknown_floor < 1.0):
            raise ValueError("household_classifier.unknown_floor must be in (0, 1)")
        if not (0.0 < self.commit_evidence_min_probability <= 1.0):
            raise ValueError(
                "household_classifier.commit_evidence_min_probability must be in (0, 1]"
            )
        if not (0.0 <= self.min_quality <= 1.0):
            raise ValueError("household_classifier.min_quality must be in [0, 1]")
        if self.dedup_window_s < 0.0:
            raise ValueError("household_classifier.dedup_window_s must be >= 0")
        if self.max_negatives_per_day < 0:
            raise ValueError("household_classifier.max_negatives_per_day must be >= 0")


class HouseholdClassifierArtifactMetrics(BaseModel):
    """Held-out evaluation metrics recorded in the classifier artifact."""

    model_config = ConfigDict(extra="forbid")

    macro_top1_at_0_8: float
    per_class_recall_at_0_8: dict[str, float]
    ece: float
    open_set_false_match_rate_at_0_8: float
    selected_c: float
    cv_log_loss: float
    eval_positive_samples: int
    eval_negative_samples: int


class HouseholdClassifierArtifact(BaseModel):
    """Versioned JSON artifact produced by ``scripts/train_household_classifier.py``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    artifact_id: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    classes: list[str] = Field(min_length=2)
    weights: list[list[float]] = Field(min_length=2)
    bias: list[float] = Field(min_length=2)
    temperature: float
    trained_days: list[str] = Field(min_length=1)
    eval_days: list[str] = Field(min_length=1)
    metrics: HouseholdClassifierArtifactMetrics
    tool_version: str = Field(min_length=1)
    seed: int
    created_at: str = Field(default="")


def _l2_normalize(values: Sequence[float]) -> npt.NDArray[np.float64] | None:
    """Return a finite L2-normalized float64 vector, or None if degenerate/non-finite."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        return None
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= _NORM_EPS:
        return None
    return arr / norm


def _stable_softmax(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    shifted = logits - float(np.max(logits))
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    if not math.isfinite(total) or total <= 0.0:
        return np.full_like(logits, 1.0 / logits.size, dtype=np.float64)
    return exps / total


class SoliderFeatureExtractor:
    """v1 ``AppearanceFeatureExtractor`` reading SOLIDER-REID embeddings.

    Prefers the PH's view prototype matching the current observation's orientation,
    falling back to the current observation's embedding and then ``gallery_mean``.
    """

    def __init__(
        self,
        *,
        feature_version: str,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self._feature_version = feature_version
        self._embedding_dim = embedding_dim

    @property
    def feature_version(self) -> str:
        return self._feature_version

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def extract(
        self,
        *,
        view_prototypes: Sequence[ViewPrototype],
        current_orientation: OrientationBin,
        observation_embedding: Sequence[float] | None,
        gallery_mean: Sequence[float] | None,
        quality: float,
    ) -> AppearanceFeature | None:
        """Select and L2-normalize the best available body embedding for a PH."""
        candidate: Sequence[float] | None = None
        selected_orientation = current_orientation

        if current_orientation != OrientationBin.UNKNOWN:
            for proto in view_prototypes:
                if proto.orientation == current_orientation and proto.embedding:
                    candidate = proto.embedding
                    break

        if candidate is None and observation_embedding:
            candidate = observation_embedding

        if candidate is None and gallery_mean:
            candidate = gallery_mean

        if candidate is None and view_prototypes:
            best_proto = max(view_prototypes, key=lambda p: p.count)
            if best_proto.embedding:
                candidate = best_proto.embedding
                selected_orientation = best_proto.orientation

        if candidate is None:
            return None

        normed = _l2_normalize(candidate)
        if normed is None or normed.size != self._embedding_dim:
            return None
        return AppearanceFeature(
            embedding=tuple(float(v) for v in normed),
            orientation=selected_orientation,
            feature_version=self._feature_version,
            quality=float(quality),
        )


def _set_health_gauge(active_state: HouseholdClassifierHealthState) -> None:
    """Publish one-hot readiness gauge across the bounded health-state vocabulary."""
    gauge = _metrics.metrics.cts_household_classifier_health
    for label in HouseholdClassifierHealthLabel:
        gauge.labels(state=str(label)).set(1.0 if str(label) == str(active_state) else 0.0)


def _validate_artifact_arrays(
    artifact: HouseholdClassifierArtifact,
    *,
    expected_embedding_dim: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str | None]:
    """Validate shapes, class names, finiteness, and temperature; return ``(W, b, error)``."""
    classes = artifact.classes
    if len(classes) < 2:
        return np.empty((0, 0)), np.empty(0), "artifact must contain at least 2 classes"
    if len(set(classes)) != len(classes):
        return np.empty((0, 0)), np.empty(0), "artifact contains duplicate class ids"
    if any(not c or not c.strip() or c == "UNKNOWN" for c in classes):
        return np.empty((0, 0)), np.empty(0), "artifact contains empty or reserved class id"

    if not math.isfinite(artifact.temperature) or artifact.temperature <= 0.0:
        return np.empty((0, 0)), np.empty(0), "temperature must be positive and finite"

    n_classes = len(classes)
    if len(artifact.weights) != n_classes or len(artifact.bias) != n_classes:
        return np.empty((0, 0)), np.empty(0), "weights/bias row count does not match classes"

    row_lengths = {len(row) for row in artifact.weights}
    if len(row_lengths) != 1:
        return np.empty((0, 0)), np.empty(0), "weights matrix rows have inconsistent lengths"
    dim = next(iter(row_lengths))
    if dim <= 0:
        return np.empty((0, 0)), np.empty(0), "weights matrix has zero feature dimension"
    if expected_embedding_dim is not None and dim != expected_embedding_dim:
        return (
            np.empty((0, 0)),
            np.empty(0),
            f"embedding dimension {dim} does not match expected {expected_embedding_dim}",
        )

    weights = np.asarray(artifact.weights, dtype=np.float64)
    bias = np.asarray(artifact.bias, dtype=np.float64)
    if not bool(np.all(np.isfinite(weights))) or not bool(np.all(np.isfinite(bias))):
        return np.empty((0, 0)), np.empty(0), "weights or bias contain non-finite values"

    return weights, bias, None


class HouseholdClassifierEvaluator:
    """Stateless NumPy evaluator for a validated :class:`HouseholdClassifierArtifact`."""

    def __init__(
        self,
        *,
        artifact: HouseholdClassifierArtifact | None,
        weights: npt.NDArray[np.float64] | None,
        bias: npt.NDArray[np.float64] | None,
        health_state: HouseholdClassifierHealthState,
        reason: str = "",
    ) -> None:
        self._artifact = artifact
        self._weights = weights
        self._bias = bias
        self._health_state = health_state
        self._reason = reason
        _set_health_gauge(health_state)

    @classmethod
    def from_artifact(
        cls,
        artifact: HouseholdClassifierArtifact,
        *,
        expected_feature_version: str,
        expected_embedding_dim: int | None = None,
    ) -> HouseholdClassifierEvaluator:
        """Build an evaluator directly from a parsed artifact model."""
        if expected_feature_version and artifact.feature_version != expected_feature_version:
            logger.warning(
                "household_classifier_artifact_incompatible",
                expected_feature_version=expected_feature_version,
                artifact_feature_version=artifact.feature_version,
            )
            return cls(
                artifact=None,
                weights=None,
                bias=None,
                health_state=HouseholdClassifierHealthState.DEGRADED_INCOMPATIBLE,
                reason="feature_version mismatch",
            )
        weights, bias, err = _validate_artifact_arrays(
            artifact, expected_embedding_dim=expected_embedding_dim
        )
        if err is not None:
            logger.warning("household_classifier_artifact_invalid", reason=err)
            return cls(
                artifact=None,
                weights=None,
                bias=None,
                health_state=HouseholdClassifierHealthState.DEGRADED_INVALID,
                reason=err,
            )
        logger.info(
            "household_classifier_artifact_ready",
            artifact_id=artifact.artifact_id,
            feature_version=artifact.feature_version,
            class_count=len(artifact.classes),
            embedding_dim=int(weights.shape[1]),
        )
        return cls(
            artifact=artifact,
            weights=weights,
            bias=bias,
            health_state=HouseholdClassifierHealthState.READY,
            reason="",
        )

    @classmethod
    def from_artifact_path(
        cls,
        artifact_path: str | Path | None,
        *,
        expected_feature_version: str,
        mode: HouseholdClassifierMode = "off",
        expected_embedding_dim: int | None = _DEFAULT_EMBEDDING_DIM,
    ) -> HouseholdClassifierEvaluator:
        """Load and validate a classifier JSON artifact from disk, failing closed on any error."""
        if mode == "off":
            return cls(
                artifact=None,
                weights=None,
                bias=None,
                health_state=HouseholdClassifierHealthState.OFF,
                reason="household_classifier.mode is off",
            )

        if not artifact_path:
            logger.warning("household_classifier_artifact_missing", reason="no_path_configured")
            return cls(
                artifact=None,
                weights=None,
                bias=None,
                health_state=HouseholdClassifierHealthState.DEGRADED_MISSING,
                reason="no artifact_path configured",
            )

        path = Path(artifact_path)
        if not path.is_file():
            logger.warning("household_classifier_artifact_missing", path=str(path))
            return cls(
                artifact=None,
                weights=None,
                bias=None,
                health_state=HouseholdClassifierHealthState.DEGRADED_MISSING,
                reason=f"file not found: {path}",
            )

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("household_classifier_artifact_unreadable", error=str(exc))
            return cls(
                artifact=None,
                weights=None,
                bias=None,
                health_state=HouseholdClassifierHealthState.DEGRADED_INVALID,
                reason=f"read error: {exc}",
            )

        try:
            artifact = HouseholdClassifierArtifact.model_validate(raw)
        except ValidationError as exc:
            logger.warning("household_classifier_artifact_schema_invalid", error=str(exc))
            return cls(
                artifact=None,
                weights=None,
                bias=None,
                health_state=HouseholdClassifierHealthState.DEGRADED_INVALID,
                reason="schema validation failed",
            )

        return cls.from_artifact(
            artifact,
            expected_feature_version=expected_feature_version,
            expected_embedding_dim=expected_embedding_dim,
        )

    @property
    def health_state(self) -> HouseholdClassifierHealthState:
        return self._health_state

    @property
    def is_ready(self) -> bool:
        return (
            self._health_state == HouseholdClassifierHealthState.READY
            and self._artifact is not None
            and self._weights is not None
            and self._bias is not None
        )

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def artifact_id(self) -> str | None:
        return self._artifact.artifact_id if self._artifact is not None else None

    @property
    def feature_version(self) -> str | None:
        return self._artifact.feature_version if self._artifact is not None else None

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self._artifact.classes) if self._artifact is not None else ()

    @property
    def embedding_dim(self) -> int | None:
        return int(self._weights.shape[1]) if self._weights is not None else None

    def health_summary(self) -> dict[str, object]:
        """Non-secret health check payload suitable for ``GET /health`` (never exposes IDs)."""
        if self._health_state == HouseholdClassifierHealthState.OFF:
            status = "info"
        elif self._health_state == HouseholdClassifierHealthState.READY:
            status = "ok"
        else:
            status = "degraded"
        return {
            "status": status,
            "state": str(self._health_state),
            "message": self._reason or str(self._health_state),
            "artifact_id": self.artifact_id,
            "feature_version": self.feature_version,
            "class_count": len(self.classes),
        }

    def predict_proba(
        self,
        embedding: Sequence[float],
        *,
        weight_multiplier: float = 1.0,
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Return ``(calibrated_probs, weighted_probs)`` over trained classes, or None.

        ``calibrated_probs`` uses ``softmax((W @ x + b) / temperature)`` and drives the
        commit-evidence threshold and UNKNOWN mass calculation. ``weighted_probs`` scales
        logits by ``weight_multiplier / temperature`` before softmax so the channel weight
        controls posterior sharpness without canceling under normalization.
        """
        if not self.is_ready:
            return None
        assert self._artifact is not None
        assert self._weights is not None
        assert self._bias is not None

        if not math.isfinite(weight_multiplier) or weight_multiplier <= 0.0:
            return None

        vec = _l2_normalize(embedding)
        if vec is None or vec.size != self._weights.shape[1]:
            return None

        logits = self._weights @ vec + self._bias
        if not bool(np.all(np.isfinite(logits))):
            return None

        temperature = self._artifact.temperature
        p_cal = _stable_softmax(logits / temperature)
        p_weighted = (
            p_cal
            if weight_multiplier == 1.0
            else _stable_softmax(logits * (weight_multiplier / temperature))
        )

        classes = self._artifact.classes
        cal_map = {cls_id: float(p) for cls_id, p in zip(classes, p_cal, strict=True)}
        weighted_map = {cls_id: float(p) for cls_id, p in zip(classes, p_weighted, strict=True)}
        return cal_map, weighted_map


@dataclass(frozen=True)
class HouseholdClassifierEvaluation:
    """Result of evaluating :class:`HouseholdClassifierChannel` for one entity."""

    likelihood: ChannelLikelihood
    active: bool
    calibrated_probs: Mapping[str, float]


def _empty_classifier_likelihood(channel_name: str) -> ChannelLikelihood:
    return ChannelLikelihood(
        channel=channel_name,
        distribution=PosteriorDist({}),
        support_multiplier=1.0,
        evidence_at=None,
        clock_evidence_ids=frozenset(),
        counts_as_commit_evidence=HOUSEHOLD_CLASSIFIER.counts_as_commit_evidence,
        commit_evidence_ids=frozenset(),
    )


class HouseholdClassifierChannel:
    """Corroborating closed-set appearance classifier channel with an open-set UNKNOWN floor."""

    name: str = "household_classifier"

    def __init__(
        self,
        *,
        config: HouseholdClassifierConfig,
        evaluator: HouseholdClassifierEvaluator | None = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator

    @property
    def is_ready(self) -> bool:
        return self._evaluator is not None and self._evaluator.is_ready

    def evaluate_classifier(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> HouseholdClassifierEvaluation:
        """Compute the classifier likelihood and gated commit-evidence IDs for *entity*."""
        evaluator = self._evaluator
        if evaluator is None or not evaluator.is_ready:
            return HouseholdClassifierEvaluation(
                likelihood=_empty_classifier_likelihood(self.name),
                active=False,
                calibrated_probs={},
            )

        feature = inputs.ph_appearance_features.get(entity.entity_id)
        if feature is None:
            return HouseholdClassifierEvaluation(
                likelihood=_empty_classifier_likelihood(self.name),
                active=False,
                calibrated_probs={},
            )

        if evaluator.feature_version and feature.feature_version != evaluator.feature_version:
            return HouseholdClassifierEvaluation(
                likelihood=_empty_classifier_likelihood(self.name),
                active=False,
                calibrated_probs={},
            )

        pred = evaluator.predict_proba(
            feature.embedding,
            weight_multiplier=self._config.weight_multiplier,
        )
        if pred is None:
            return HouseholdClassifierEvaluation(
                likelihood=_empty_classifier_likelihood(self.name),
                active=False,
                calibrated_probs={},
            )

        cal_probs, weighted_probs = pred
        max_cal_prob = max(cal_probs.values(), default=0.0)
        unknown_mass = max(self._config.unknown_floor, 1.0 - max_cal_prob)
        known_mass = max(0.0, 1.0 - unknown_mass)

        dist: dict[str, float] = {
            cls_id: known_mass * prob for cls_id, prob in weighted_probs.items()
        }
        dist["UNKNOWN"] = unknown_mass

        commit_ids = (
            frozenset(
                cls_id
                for cls_id, prob in cal_probs.items()
                if prob >= self._config.commit_evidence_min_probability
            )
            if HOUSEHOLD_CLASSIFIER.counts_as_commit_evidence
            else frozenset()
        )

        # Grade invariant: never refreshes the independent-evidence clock.
        assert not HOUSEHOLD_CLASSIFIER.refreshes_evidence_clock
        likelihood = ChannelLikelihood(
            channel=self.name,
            distribution=PosteriorDist(dist),
            support_multiplier=1.0,
            evidence_at=None,
            clock_evidence_ids=frozenset(),
            counts_as_commit_evidence=HOUSEHOLD_CLASSIFIER.counts_as_commit_evidence,
            commit_evidence_ids=commit_ids,
        )
        return HouseholdClassifierEvaluation(
            likelihood=likelihood,
            active=True,
            calibrated_probs=cal_probs,
        )

    async def evaluate(
        self,
        entity: IdentityResolvableEntity,
        inputs: ResolverInputs,
    ) -> ChannelLikelihood:
        """Implement :class:`EvidenceChannel`."""
        return self.evaluate_classifier(entity, inputs).likelihood
