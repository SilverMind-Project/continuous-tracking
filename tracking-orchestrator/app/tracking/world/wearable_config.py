"""Configuration for wearable-to-PH association and wearable identity evidence."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain import WEARABLE_MODES, WearableMode


@dataclass(frozen=True)
class WearableConfig:
    """Parameters governing wearable fix intake, association, attestation, and scoring."""

    # Rollout switch: "off" skips fix consumption and association; "shadow" runs
    # association and counterfactual evaluation without affecting live identity;
    # "enabled" feeds accepted wearable evidence into the Bayesian resolver.
    mode: WearableMode = "off"

    # Sliding association window duration (seconds) and minimum fixes per pair.
    window_s: float = 10.0
    min_fixes: int = 5
    min_window_llr: float = 6.0
    min_margin_llr: float = 4.0

    # Per-fix trajectory gating and innovation covariance floor terms.
    gate_chi2: float = 9.21
    gate_penalty_llr: float = -4.0
    timing_sigma_s: float = 0.5
    bias_floor_m: float = 0.15

    # Fix staleness and history interpolation caps.
    max_fix_age_s: float = 15.0
    buffer_max_age_s: float = 30.0
    max_extrapolation_s: float = 0.5
    max_fix_buffer_per_tag: int = 128

    # Covariance-based evidence class routing and room-class thresholds.
    trajectory_max_sigma_m: float = 0.75
    room_min_mass: float = 0.7
    room_polygon_min_mass: float = 0.05
    room_match_eps: float = 0.1

    # Parked-tag and multi-tag disagreement guards.
    parked_after_s: float = 600.0
    tag_disagreement_window_s: float = 10.0

    # Graded attestation thresholds (trajectory and room).
    trajectory_attest_min_duration_s: float = 10.0
    trajectory_attest_min_fixes: int = 8
    room_attest_min_duration_s: float = 60.0
    room_entry_window_s: float = 20.0
    attest_min_llr: float = 15.0

    # Wearable evidence channel likelihood parameters.
    weight_multiplier: float = 2.0
    p_max: float = 0.97
    p_max_room: float = 0.90
    llr_mid: float = 8.0
    llr_scale: float = 3.0

    # Duplicate-active guard ranking threshold and swap-suspected escalation.
    duplicate_guard_min_probability: float = 0.90
    swap_conflict_count: int = 3
    swap_conflict_window_s: float = 600.0

    def __post_init__(self) -> None:
        if self.mode not in WEARABLE_MODES:
            raise ValueError(f"wearable.mode must be one of {WEARABLE_MODES}, got {self.mode!r}")
        if self.window_s <= 0.0:
            raise ValueError("wearable.window_s must be positive")
        if self.min_fixes < 1:
            raise ValueError("wearable.min_fixes must be >= 1")
        if self.trajectory_max_sigma_m <= 0.0:
            raise ValueError("wearable.trajectory_max_sigma_m must be positive")
        if not 0.0 < self.p_max < 1.0:
            raise ValueError("wearable.p_max must be in (0, 1)")
        if not 0.0 < self.p_max_room <= self.p_max:
            raise ValueError("wearable.p_max_room must be in (0, p_max]")
        if self.llr_scale <= 0.0:
            raise ValueError("wearable.llr_scale must be positive")
        if self.weight_multiplier <= 0.0:
            raise ValueError("wearable.weight_multiplier must be positive")
        if not 0.0 < self.room_match_eps < 0.5:
            raise ValueError("wearable.room_match_eps must be in (0, 0.5)")
        if not 0.0 < self.duplicate_guard_min_probability <= 1.0:
            raise ValueError("wearable.duplicate_guard_min_probability must be in (0, 1]")
        if self.swap_conflict_count < 1:
            raise ValueError("wearable.swap_conflict_count must be >= 1")
