"""PH-local appearance-update policy (M03 tasks 7-9).

A geometrically-valid association decides *that* a PH matched an observation.
This policy decides whether that observation's embedding is allowed to update
the PH's appearance state — ``gallery_mean``, the per-orientation view
prototypes, and the ``mean_quality`` EMA. The two decisions are independent: a
rejected embedding still advances the Kalman state and ``observation_count`` (the
person was there), it simply may not pollute appearance.

An embedding is accepted only when it is:

- finite (no NaN/inf),
- non-degenerate (non-zero norm, so it can be L2-normalised; real SOLIDER
  vectors are already unit length and pass trivially),
- quality-qualified (crop quality ≥ threshold),
- orientation-valid, AND
- either consistent with the existing prototype for its orientation (cosine
  similarity ≥ floor) or initialising a *new* qualified orientation
  (orientation confidence ≥ threshold).

A rejected embedding is reported with a typed reason for diagnostics; it is
never labelled with the PH identity. Pure functions only — no I/O, no metrics,
no datetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ...domain import OrientationBin, ViewPrototype
from .config import WorldTrackerConfig
from .helpers import cosine_similarity


class AppearanceRejectReason(StrEnum):
    """Why a matched observation's embedding was not allowed to update appearance."""

    NO_EMBEDDING = "no_embedding"
    NOT_FINITE = "not_finite"
    DEGENERATE_NORM = "degenerate_norm"
    LOW_QUALITY = "low_quality"
    UNKNOWN_ORIENTATION = "unknown_orientation"
    LOW_ORIENTATION_CONFIDENCE = "low_orientation_confidence"
    CROSS_PERSON_OUTLIER = "cross_person_outlier"


@dataclass(frozen=True)
class AppearanceDecision:
    """Outcome of the appearance-update policy for one matched observation."""

    accept: bool
    reason: AppearanceRejectReason | None = None  # set iff accept is False


_ACCEPT = AppearanceDecision(accept=True)


def evaluate_appearance_update(
    embedding: list[float] | None,
    orientation: OrientationBin,
    orientation_confidence: float,
    quality: float,
    existing_prototypes: tuple[ViewPrototype, ...],
    cfg: WorldTrackerConfig,
) -> AppearanceDecision:
    """Decide whether *embedding* may update PH-local appearance state.

    When ``cfg.enable_appearance_outlier_rejection`` is False this is a
    behaviour-preserving pass-through that accepts any non-empty embedding (the
    kill-switch path). Otherwise it applies the full M03 contamination guard.
    """
    if not embedding:
        return AppearanceDecision(accept=False, reason=AppearanceRejectReason.NO_EMBEDDING)

    if not cfg.enable_appearance_outlier_rejection:
        return _ACCEPT

    vec = np.asarray(embedding, dtype=np.float32)
    if not bool(np.all(np.isfinite(vec))):
        return AppearanceDecision(accept=False, reason=AppearanceRejectReason.NOT_FINITE)

    norm = float(np.linalg.norm(vec))
    if norm <= cfg.appearance_embedding_norm_tol:
        # Degenerate (zero / near-zero) embedding: cannot be normalised, carries
        # no appearance information. A real SOLIDER vector is unit length.
        return AppearanceDecision(accept=False, reason=AppearanceRejectReason.DEGENERATE_NORM)

    if quality < cfg.appearance_min_quality:
        return AppearanceDecision(accept=False, reason=AppearanceRejectReason.LOW_QUALITY)

    if orientation == OrientationBin.UNKNOWN:
        return AppearanceDecision(accept=False, reason=AppearanceRejectReason.UNKNOWN_ORIENTATION)

    matching = next((p for p in existing_prototypes if p.orientation == orientation), None)
    if matching is None:
        # Initialising a new orientation prototype: gate on orientation
        # confidence so we do not anchor a view on a low-confidence pose.
        if orientation_confidence < cfg.appearance_new_orientation_min_confidence:
            return AppearanceDecision(
                accept=False, reason=AppearanceRejectReason.LOW_ORIENTATION_CONFIDENCE
            )
        return _ACCEPT

    # Established orientation: reject abrupt cross-person jumps. Compare on
    # L2-normalised vectors so the cosine threshold is meaningful even when the
    # incoming embedding is not yet unit length (prototypes are always stored
    # normalised by update_view_prototypes).
    unit_vec = (vec / norm).tolist()
    sim = cosine_similarity(list(matching.embedding), unit_vec)
    if sim < cfg.appearance_outlier_min_sim:
        return AppearanceDecision(accept=False, reason=AppearanceRejectReason.CROSS_PERSON_OUTLIER)

    return _ACCEPT
