"""World tracker configuration — single frozen dataclass with all tunables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldTrackerConfig:
    """All tunable parameters for the world-coordinate person tracker.

    Defaults are initial values; tune during acceptance testing against
    the testcontainer scenarios as documented in the M1 appendix.
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

    # ---- Inferred handoff (M4 consumer) ----
    inferred_handoff_max_s: float = 600.0
    inferred_handoff_max_distance_m: float = 5.0

    # ---- Identity evidence window ----
    evidence_window_s: float = 30.0
