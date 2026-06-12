"""Algorithm specifications and data quality for dementia signals.

Each signal has an ``AlgorithmSpec`` describing its scientific basis,
evidence grade, and required inputs.  ``DataQuality`` gates signal
emission on identity confidence, coverage, and baseline readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EvidenceGrade = Literal[
    "clinical_review",
    "observational_study",
    "caregiver_guidance",
    "local_baseline_only",
    "experimental",
]

NON_DIAGNOSTIC_DISCLAIMER = (
    "This signal is a behavioural pattern alert for caregiver context. "
    "It is not a diagnosis of dementia, Alzheimer's disease, delirium, "
    "infection, or any other medical condition."
)


@dataclass(frozen=True)
class AlgorithmSpec:
    """Metadata for one dementia signal detection algorithm."""

    name: str
    version: int
    evidence_grade: EvidenceGrade
    clinical_label: str
    disclaimer: str = NON_DIAGNOSTIC_DISCLAIMER
    required_inputs: tuple[str, ...] = ()
    min_baseline_samples: int = 3
    min_observation_coverage_ratio: float = 0.0  # fraction of expected frames


# ---------------------------------------------------------------------------
# Signal specs
# ---------------------------------------------------------------------------

PACING_SPEC = AlgorithmSpec(
    name="pacing-v3",
    version=3,
    evidence_grade="observational_study",
    clinical_label="Pacing / Repetitive Movement",
    required_inputs=("room_transitions", "trajectory_points"),
    min_baseline_samples=7,
)

EVENING_ACTIVITY_SPEC = AlgorithmSpec(
    name="evening-activity-change-v1",
    version=1,
    evidence_grade="clinical_review",
    clinical_label="Evening Activity Change (Sundowning Index)",
    required_inputs=("trajectory_points", "time_of_day"),
    min_baseline_samples=14,
)

NIGHTTIME_MOVEMENT_SPEC = AlgorithmSpec(
    name="nighttime-movement-v1",
    version=1,
    evidence_grade="caregiver_guidance",
    clinical_label="Nighttime Movement",
    required_inputs=("room_transitions", "time_of_day"),
    min_baseline_samples=7,
)

STILLNESS_SPEC = AlgorithmSpec(
    name="stillness-v2",
    version=2,
    evidence_grade="observational_study",
    clinical_label="Prolonged Stillness",
    required_inputs=("trajectory_points", "motion_energy", "posture"),
    min_baseline_samples=3,
)

UNOBSERVED_GAP_SPEC = AlgorithmSpec(
    name="unobserved-gap-v1",
    version=1,
    evidence_grade="caregiver_guidance",
    clinical_label="Unobserved Gap (not absence without coverage proof)",
    required_inputs=("trajectory_points", "camera_coverage"),
    min_baseline_samples=7,
)

BATHROOM_DWELL_SPEC = AlgorithmSpec(
    name="bathroom-dwell-v1",
    version=1,
    evidence_grade="caregiver_guidance",
    clinical_label="Bathroom Dwell Anomaly",
    required_inputs=("room_dwells", "room_taxonomy"),
    min_baseline_samples=5,
)

FALL_SUSPECTED_SPEC = AlgorithmSpec(
    name="fall-pose-heuristic-v1",
    version=1,
    evidence_grade="observational_study",
    clinical_label="Suspected Fall (Pose Heuristic)",
    # Lightweight skeleton-based detection; see VISION.md §3.1 citations.
    required_inputs=("pose_keypoints", "bbox", "posture_scores", "kalman_floor_speed"),
    # Fast-path: emits on first trigger without baseline comparison.
    min_baseline_samples=0,
)

GAIT_SLOWING_SPEC = AlgorithmSpec(
    name="gait-slowing-v1",
    version=1,
    evidence_grade="observational_study",
    clinical_label="Gait Slowing",
    # Sustained gait speed decline is among the best-validated predictors of cognitive
    # decline (PMC8968722; Frontiers fdgth.2025.1698551).  Passive in-home camera
    # measurement validated in cognitively impaired populations (Hegde et al.,
    # Alzheimer's & Dementia 2025, dad2.70085).  Within-person two-window comparison
    # (recent 28 d vs prior 28 d); absolute floor ~0.8 m/s and meaningful decline on
    # the order of 0.1 m/s are clinically relevant (PMC8968722).
    required_inputs=("gait_daily", "floor_speed_m_s"),
    min_baseline_samples=10,
)


@dataclass(frozen=True)
class IdentitySignalContext:
    """Identity-level context for signal computation."""

    identity_id: str
    identity_confidence_mean: float  # mean confidence across window
    observation_count: int  # number of trajectory points in window
    coverage_ratio: float  # observed / expected frame count
    baseline_sample_count: int
    algorithm_version: int


@dataclass(frozen=True)
class DataQuality:
    """Data quality assessment for one signal computation window."""

    identity_confidence_ok: bool
    coverage_ok: bool
    baseline_ready: bool
    reason: str = ""

    @property
    def sufficient(self) -> bool:
        return self.identity_confidence_ok and self.coverage_ok and self.baseline_ready
