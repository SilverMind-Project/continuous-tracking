"""Configuration for wearable-to-PH association and wearable identity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.domain import WEARABLE_MODES, WearableMode

FaceAttestationMode = Literal["off", "enabled"]
SpatialFusionMode = Literal["off", "shadow", "enabled"]
SPATIAL_FUSION_MODES: tuple[SpatialFusionMode, ...] = ("off", "shadow", "enabled")


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

    # Parked-tag and multi-tag disagreement guards. A device's room counts toward a
    # disagreement only when it holds at least this share of that tag's window room mass.
    parked_after_s: float = 600.0
    tag_disagreement_window_s: float = 10.0
    tag_disagreement_min_room_share: float = 0.6

    # How long a segment stays open, emitting no evidence, after its last accepted round.
    # Longer than a duty-cycled watch's burst cooldown, so the next burst resumes the same
    # segment and its latched grade instead of re-attesting.
    segment_hold_s: float = 75.0

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

    # Image-only evaluation: the share of attested PH resolutions that are re-scored without the
    # wearable channel, once without wearable-taught gallery rows (control) and once with them
    # (treatment); in both, rows seen in the last ``evaluation_exclusion_s`` do not vote. 0.0 is
    # off.
    evaluation_sample_rate: float = 0.0
    evaluation_exclusion_s: float = 600.0
    # A gap between two sampled evaluations of one PH longer than this counts no time (the PH
    # or its tag was away); the next evaluation starts a new interval.
    evaluation_max_gap_s: float = 30.0

    # Offer emitted faces of attested PHs to person-identification-service's operator-gated
    # attested-exemplar queue. "off" sends nothing.
    face_attestation: FaceAttestationMode = "off"

    # Automatically revoke what a tag taught and hold teaching when a segment is first marked
    # swap_suspected. Acts only when a wearable channel is running.
    revocation_auto_on_swap: bool = True

    # Spatial fusion of trajectory-attested position fixes into the PH Kalman state.
    # "off" never touches PH state; "shadow" measures each gated update without applying it
    # and changes nothing else; "enabled" applies it and switches PH floor history to raw
    # camera measurements, so association never scores a wearable against its own fixes.
    spatial_fusion_mode: SpatialFusionMode = "off"
    # Only fixes this tight (major-axis sigma, metres) are fused.
    spatial_fusion_max_sigma_m: float = 0.35
    # Per-sensor systematic error added to every fused fix and kept as a floor: a fused
    # fix never leaves the PH position more certain than this (metres).
    spatial_fusion_bias_floor_m: float = 0.12
    # Largest gap between the fix and the PH state time that is fused (seconds).
    spatial_fusion_max_dt_s: float = 3.0
    # Two-dof chi-square gate on the time-aligned innovation (99%).
    spatial_fusion_gate_chi2: float = 9.21
    # How long fused fixes may keep a camera-occluded PH open after its last camera
    # observation (seconds). 0 disables the bridge.
    coasting_bridge_max_s: float = 10.0

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
        if not 0.0 <= self.evaluation_sample_rate <= 1.0:
            raise ValueError("wearable.evaluation.sample_rate must be in [0, 1]")
        if self.evaluation_exclusion_s < 0.0:
            raise ValueError("wearable.evaluation.exclusion_s must be >= 0")
        if self.evaluation_max_gap_s <= 0.0:
            raise ValueError("wearable.evaluation.max_gap_s must be positive")
        if self.face_attestation not in ("off", "enabled"):
            raise ValueError("wearable.supervision.face_attestation must be 'off' or 'enabled'")
        if self.weight_multiplier <= 0.0:
            raise ValueError("wearable.weight_multiplier must be positive")
        if not 0.0 < self.room_match_eps < 0.5:
            raise ValueError("wearable.room_match_eps must be in (0, 0.5)")
        if not 0.0 < self.duplicate_guard_min_probability <= 1.0:
            raise ValueError("wearable.duplicate_guard_min_probability must be in (0, 1]")
        if self.swap_conflict_count < 1:
            raise ValueError("wearable.swap_conflict_count must be >= 1")
        if not 0.5 <= self.tag_disagreement_min_room_share <= 1.0:
            raise ValueError("wearable.tag_disagreement_min_room_share must be in [0.5, 1]")
        if self.segment_hold_s < 0.0:
            raise ValueError("wearable.segment_hold_s must be >= 0")
        if self.spatial_fusion_mode not in SPATIAL_FUSION_MODES:
            raise ValueError(
                f"wearable.spatial_fusion.mode must be one of {SPATIAL_FUSION_MODES}, "
                f"got {self.spatial_fusion_mode!r}"
            )
        if self.spatial_fusion_mode != "off" and self.mode == "off":
            raise ValueError("wearable.spatial_fusion.mode needs wearable.mode shadow or enabled")
        if self.spatial_fusion_max_sigma_m <= 0.0:
            raise ValueError("wearable.spatial_fusion.max_sigma_m must be positive")
        if self.spatial_fusion_bias_floor_m <= 0.0:
            raise ValueError("wearable.spatial_fusion.bias_floor_m must be positive")
        if self.spatial_fusion_max_dt_s <= 0.0:
            raise ValueError("wearable.spatial_fusion.max_dt_s must be positive")
        if self.spatial_fusion_gate_chi2 <= 0.0:
            raise ValueError("wearable.spatial_fusion.gate_chi2 must be positive")
        if self.coasting_bridge_max_s < 0.0:
            raise ValueError("wearable.spatial_fusion.coasting_bridge_max_s must be >= 0")
