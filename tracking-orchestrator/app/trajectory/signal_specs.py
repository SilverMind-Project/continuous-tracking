"""Algorithm specifications and data quality for dementia signals.

Each signal has an ``AlgorithmSpec`` describing its scientific basis,
evidence grade, and required inputs.  ``DataQuality`` gates signal
emission on identity confidence, coverage, and baseline readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..domain import DementiaSignal

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

AGITATION_MOTOR_SPEC = AlgorithmSpec(
    name="agitation-motor-heuristic-v1",
    version=1,
    # Experimental: no ground-truth-labelled dataset for this deployment.
    # Validated approaches exist (skeletal + physiological fusion PMC12741316;
    # privacy-protecting behaviours-of-risk detection s12938-023-01065-3) but
    # direct comparison against CMAI scores has not yet been performed here.
    evidence_grade="experimental",
    clinical_label="Restlessness Elevated (Agitation Motor Index)",
    required_inputs=(
        "trajectory_points",
        "motion_energy",
        "floor_speed_m_s",
        "agitation_window_baseline",
    ),
    min_baseline_samples=5,
)

SAME_CLOTHES_SPEC = AlgorithmSpec(
    name="same-clothes-appearance-v1",
    version=1,
    # Experimental: a day-over-day appearance-similarity prefilter, not yet
    # backtested against caregiver-labelled same-clothes days for this
    # deployment (DL-M07 Part D). CC's VLM confirm + shower-proxy join
    # (DL-M08) is the evidenced-alert layer; this spec is the CTS-side
    # prefilter's own metadata.
    evidence_grade="experimental",
    clinical_label="Same Clothes Suspected (Appearance Prefilter)",
    required_inputs=("gallery_mean", "mean_quality"),
    min_baseline_samples=0,  # day-over-day comparison, not a z-score baseline
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


# ---------------------------------------------------------------------------
# Spec lookup + metadata application (shared by DementiaSignalWorker and any
# other producer of a DementiaSignal, e.g. AppearanceEvaluator)
# ---------------------------------------------------------------------------

SIGNAL_SPEC: dict[str, AlgorithmSpec] = {
    "pacing": PACING_SPEC,
    "sundowning_index": EVENING_ACTIVITY_SPEC,
    "nighttime_movement": NIGHTTIME_MOVEMENT_SPEC,
    "stillness_anomaly": STILLNESS_SPEC,
    "absence": UNOBSERVED_GAP_SPEC,
    "bathroom_dwell_anomaly": BATHROOM_DWELL_SPEC,
    "fall_suspected": FALL_SUSPECTED_SPEC,
    "gait_slowing": GAIT_SLOWING_SPEC,
    "agitation_index": AGITATION_MOTOR_SPEC,
    "same_clothes_suspected": SAME_CLOTHES_SPEC,
}


def apply_algorithm_metadata(signal: DementiaSignal, signal_kind: str) -> DementiaSignal:
    """Return a copy of *signal* stamped with its AlgorithmSpec metadata."""
    spec = SIGNAL_SPEC.get(signal_kind)
    if spec is not None:
        return replace(
            signal,
            algorithm_name=spec.name,
            evidence_grade=spec.evidence_grade,
            algorithm_spec_json=(
                f'{{"name": "{spec.name}", "version": {spec.version}, '
                f'"evidence_grade": "{spec.evidence_grade}", '
                f'"clinical_label": "{spec.clinical_label}", '
                f'"disclaimer": "{spec.disclaimer}", '
                f'"min_baseline_samples": {spec.min_baseline_samples}}}'
            ),
        )
    return replace(
        signal,
        algorithm_name="unknown",
        evidence_grade="experimental",
        algorithm_spec_json="{}",
    )
