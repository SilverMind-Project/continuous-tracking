"""World tracker configuration — single frozen dataclass with all tunables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldTrackerConfig:
    """All tunable parameters for the world-coordinate person tracker.

    Defaults are initial values; tune during acceptance testing against
    the testcontainer integration scenarios.
    """

    # ---- Geometric gate ----
    gate_chi2: float = 9.21  # chi-squared 99% with 2 dof
    observation_noise_m: float = 0.25  # calibration residual 95th percentile
    process_noise_accel_m_s2: float = 0.5  # white-noise acceleration std dev

    # ---- Kalman initialisation ----
    initial_position_sigma_m: float = 1.0
    initial_velocity_sigma_m_s: float = 2.0

    # ---- Velocity decay for unobserved PHs ----
    velocity_decay_s: float = 3.0

    # ---- Cost matrix weights (must sum roughly to 1.0) ----
    alpha_geo: float = 0.5
    alpha_app: float = 0.4
    alpha_height: float = 0.1

    # ---- Appearance ----
    height_sigma_m: float = 0.15

    # ---- Identity conflict hard gate ----
    face_conflict_threshold: float = 0.70

    # ---- PH lifecycle ----
    ph_close_grace_s: float = 5.0
    min_observations_to_publish: int = 3

    # ---- Inferred handoff ----
    inferred_handoff_max_s: float = 600.0
    inferred_handoff_max_distance_m: float = 5.0

    # ---- PH revival ----
    # When enabled, an unmatched observation revives a recently-closed same-camera
    # PH whose appearance matches instead of spawning a brand-new UNKNOWN PH.
    enable_ph_revival: bool = False
    revive_max_age_s: float = 30.0
    revive_max_distance_m: float = 2.0
    revive_appearance_min_sim: float = 0.55

    # ---- Uncalibrated association relaxation ----
    # When enabled, uncalibrated-camera observations use a wider geometric gate
    # and appearance-weighted cost so synthetic floor-point jitter does not
    # force a close and respawn.
    enable_uncalibrated_gate_relax: bool = False
    uncalibrated_gate_chi2: float = 21.0
    uncalibrated_alpha_app: float = 0.7

    # ---- Multi-view ReID association ----
    # When enabled, pair_cost uses max-over-view-prototypes cosine similarity
    # instead of the single gallery_mean.  Ships shadow-first (off by default).
    enable_multiview_association: bool = False

    # ---- Cross-camera revival ----
    # When enabled, unmatched observations can revive recently-closed PHs from
    # a different camera via topology and multi-view appearance gating.
    enable_cross_camera_revival: bool = False
    cross_camera_min_plausibility: float = 0.05
    cross_camera_revive_appearance_min_sim: float = 0.60

    # ---- Group appearance dedup ----
    # When enabled, uncalibrated observations in a declared overlap group are
    # deduped by appearance similarity instead of being skipped.
    enable_group_appearance_dedup: bool = False
    dedup_group_appearance_min_sim: float = 0.75

    # ---- Cross-camera dedup ----
    # Pre-association floor-point dedup for observations from overlapping cameras.
    # Two observations from different cameras within this distance on the floor plane
    # and not in identity conflict are merged into one representative before the
    # Hungarian assignment runs, preventing one person from spawning two PHs.
    dedup_enabled: bool = True
    dedup_max_distance_m: float = 0.6
    dedup_residual_coeff_k: float = 1.0
    dedup_max_distance_ceiling_m: float = 1.5
    dedup_require_no_face_conflict: bool = True

    # ---- Low-confidence detection recovery (BYTE-style second association) ----
    # When enabled, YOLO detections in [low_confidence_floor, detector_confidence) are
    # kept and offered in a second association pass against already-open PHs only, with a
    # tightened geometric gate (recovery_gate_chi2, 95% chi-squared with 2 dof vs the
    # primary 99%). They can never spawn a new PH, seed the gallery, update appearance
    # EMAs, or contribute identity evidence.  Guardrail: the only ghost-persistence risk
    # is "PH closes a few seconds later than it should" — never a phantom person or
    # identity contamination. Ships dark (default false); enable per-deployment after
    # fixture proofs pass, same as other CTS robustness flags.
    enable_low_confidence_recovery: bool = False
    low_confidence_floor: float = 0.25
    recovery_gate_chi2: float = 5.99  # chi-squared 95% with 2 dof
