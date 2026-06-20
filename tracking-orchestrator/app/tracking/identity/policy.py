"""Authority-independent policy configuration for identity evaluation.

Defines the full ``CommitPolicy`` dataclass that drives all commit
evaluation, maintenance windows, quality gates, and face-lock management.
Importing this module is safe from any layer — no storage, transport, or
pipeline imports are allowed here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommitPolicy:
    """Complete configuration for identity commit evaluation.

    Defaults match the live ``ResolverConfig`` values so an unparameterised
    ``CommitPolicy()`` replicates current production behaviour.

    Grouping convention (fields are logically grouped, not alphabetical):
      - Commit thresholds (prob, margin, dense variants)
      - Maintenance windows (prior, face-lock)
      - Face-lock gating
      - Quality gate
      - Flip debounce
      - Contradiction detection
      - Sticky maintenance
    """

    # --- Commit thresholds -------------------------------------------------
    commit_prob: float = 0.65
    commit_margin: float = 0.15
    commit_prob_dense: float = 0.80
    commit_margin_dense: float = 0.20

    # --- Maintenance windows -----------------------------------------------
    prior_maintenance_max_age_s: float = 120.0
    face_lock_maintenance_max_age_s: float = 300.0

    # --- Face-lock gating --------------------------------------------------
    face_commit_min_confidence: float = 0.70
    min_quality_to_face_lock: float = 0.45

    # --- Quality gate -------------------------------------------------------
    min_quality_to_commit: float = 0.35
    enable_quality_gate: bool = False

    # --- Flip debounce ------------------------------------------------------
    flip_debounce_window_s: float = 10.0
    enable_flip_debounce: bool = False

    # --- Contradiction detection --------------------------------------------
    contradiction_face_confidence: float = 0.70
    contradiction_posterior_prob: float = 0.80
    contradiction_posterior_margin: float = 0.20

    # --- Sticky maintenance -------------------------------------------------
    enable_sticky_maintenance: bool = False
