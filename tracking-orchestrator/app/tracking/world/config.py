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

    # ---- Adaptive zero-velocity update (ZUPT) ----
    # Enter below a sustained near-still speed; exit at the bottom of the
    # clinically relevant slow-shuffle band so 0.2-0.4 m/s gait is not clamped.
    zupt_speed_enter_m_s: float = 0.12
    zupt_speed_exit_m_s: float = 0.20
    # Debounce brief pauses, and require the measurement to agree with the
    # prediction before counting a frame as stationary.
    zupt_consecutive_frames: int = 5
    zupt_innov_chi2: float = 2.0
    # Smaller sigma pulls velocity harder toward zero once stationarity is gated.
    zupt_velocity_sigma_m_s: float = 0.05

    # ---- Cost matrix weights (must sum roughly to 1.0) ----
    alpha_geo: float = 0.5
    alpha_app: float = 0.4
    alpha_height: float = 0.1

    # ---- Appearance ----
    height_sigma_m: float = 0.15

    # ---- PH-local appearance-update contamination guard ----
    # One bad association must not immediately pollute a PH's appearance state.
    # An embedding is EMA'd into gallery_mean / view prototypes / mean_quality
    # only when it is finite, unit-normalised, quality-qualified, orientation-
    # valid, and either consistent with the matching orientation prototype or
    # initialising a new qualified orientation. Rejected embeddings advance the
    # Kalman state and observation_count but touch NO appearance state, and are
    # never labelled as the PH's identity.
    #
    # Ships ON: completion criterion ("one bad association cannot
    # immediately pollute PH-local prototypes") is a property of the shipped
    # milestone, not of a flag that has to be flipped. The flag is a kill-switch
    # that DEFAULTS ON, for rollback only.
    enable_appearance_outlier_rejection: bool = True
    # Minimum crop quality [0,1] for an embedding to update appearance state.
    appearance_min_quality: float = 0.30
    # An embedding whose L2 norm is at/below this is degenerate (zero vector)
    # and carries no appearance information; it is rejected. Real SOLIDER
    # embeddings are unit length and pass; non-unit vectors are normalised
    # internally for the cross-person cosine comparison.
    appearance_embedding_norm_tol: float = 1e-6
    # Cosine similarity floor against the matching orientation prototype. Below
    # this an established orientation's embedding is treated as a cross-person
    # outlier and rejected. New (not-yet-seen) orientations bypass this and are
    # admitted on quality + orientation confidence alone.
    appearance_outlier_min_sim: float = 0.35
    # Minimum orientation confidence to initialise a NEW orientation prototype.
    appearance_new_orientation_min_confidence: float = 0.30

    # ---- Identity conflict hard gate ----
    face_conflict_threshold: float = 0.70

    # ---- Association covariance/point validation ----
    # Fail-closed validation of floor points and observation covariance before
    # the Mahalanobis gate. Non-finite points/covariance always gate out (pure
    # safety, not flagged). This flag additionally enforces the symmetry/PSD
    # checks and the trace cap below; defaults ON because an invalid covariance
    # must never gate-match. Flip OFF only for development A/B comparison.
    enable_covariance_validation: bool = True
    # Reject observation covariance whose position-trace exceeds this (m²).
    # Above any legitimate observation covariance, below jump-rescuing inflation.
    covariance_max_trace_m2: float = 20.0
    covariance_symmetry_tol_m2: float = 1e-6
    covariance_psd_tol_m2: float = -1e-9

    # ---- Typed authoritative identity evidence in association ----
    # Operator-confirmed PH identity is absolute authority: a conflicting
    # recognized face hard-gates regardless of face_conflict_threshold, and a
    # sub-threshold same-identity face cannot weaken it. Wired from PH metadata.
    # Verified-ReID disagreement is a configurable STRONG COST, never a hard
    # gate. The input (a per-observation verified-ReID identity) is plumbed by
    # governed-gallery; until then no verified-ReID id reaches association,
    # so this is inert. Default OFF for that cross-milestone dependency, NOT as
    # a "ship dark" robustness toggle.
    enable_reid_disagreement_cost: bool = False
    reid_disagreement_cost: float = 0.6  # added to pair cost on verified disagreement
    # Minimum gallery cosine similarity for treating a top operator_verified
    # gallery hit as an observation's verified-ReID identity.
    # Only consulted when enable_reid_disagreement_cost is true, so the
    # per-observation gallery lookup adds zero cost while the flag is off.
    reid_disagreement_min_similarity: float = 0.5
    # Hard vote-age cutoff for the disagreement probe's gallery lookup, in
    # seconds. None means no cutoff.
    # Production default is 43200s/12h via settings.yaml, mirroring
    # resolver.gallery_vote_max_age_s so the probe sees the same corpus the
    # resolver votes with even though the flag above is off in production
    # today; enabling it later inherits correct temporal behavior for free.
    reid_disagreement_max_age_s: float | None = None

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

    # ---- Observation uncertainty model (used by information-form fusion) ----
    # These mirror the module-level constants in observation_model.py and make
    # them tunable per-deployment.  Tune against acceptance fixtures.
    k_cal: float = 1.0  # R_cal/bias-floor = (k_cal · residual_m)² · I  (m²)
    base_footpoint_sigma_px: float = 4.0  # detector bbox-bottom localization noise (px, 1 sigma)
    occluded_footpoint_inflation: float = 8.0  # sigma multiplier when feet are hidden/truncated

    # ---- Stabilized primary camera ----
    # A challenger camera must be the best-view camera for this many consecutive
    # tracker frames before the PH's displayed/keyframe camera switches.
    primary_switch_frames: int = 5

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

    # ---- Low-confidence band measurement (diagnostic, read-only) ----
    # When enabled (and recovery off), the detect stage decodes at
    # low_confidence_floor and records how often a present person is visible only
    # in [low_confidence_floor, detector_confidence) — i.e. whether detection
    # gaps are caused by the confidence cut. Does NOT feed the low band to the
    # tracker, so tracking behavior is unchanged. Off by default.
    measure_low_confidence_band: bool = False
