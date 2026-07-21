"""M09 acceptance gate tests — quantitative CI proofs for the spatial-fusion milestones.

Each gate compares a measurement from the anisotropic-fusion (current) tracker
against a legacy-mode baseline (isotropic + ZUPT disabled) or against an
absolute bound.  Thresholds are justified by a baseline run; see the inline
comments that cite the recorded numbers.

Marked @pytest.mark.integration; run under ``make ci``.  All tests use
InMemoryPHRepository so no Postgres testcontainer is required.
"""

from __future__ import annotations

import math

import pytest

from app.domain import ObservationGeometry, OrientationBin
from app.tracking.world.observation_model import posture_view_weight
from app.trajectory.posture import GlobalPostureTracker, PostureScores
from tests.integration._fusion_metrics import compute_eigen_ratio, run_replay
from tests.integration._replay import FIXTURES_DIR, load_fixture, load_truth

# ── Gate constants ─────────────────────────────────────────────────────────────
# All thresholds justified by the implementation baseline run (2026-06-18).

# Baseline (legacy=isotropic+no-ZUPT): step jitter = 0.0279 m
# Baseline (fused):                    step jitter = 0.0073 m  → ratio = 0.262
# Gate: fused jitter ≤ 50% of legacy jitter (ratio ≤ 0.50); measured 0.26.
STATIONARY_VARIANCE_DROP_MIN = 0.50

# Baseline: mean_speed (frames 3+) = 0.2854 m/s vs truth 0.30 m/s (error=0.015)
# Tolerance = 0.05 m/s covers Kalman warmup and discretization at 2 Hz.
SLOW_SHUFFLE_SPEED_TOL = 0.05  # m/s

# Baseline: position jump at cam-B dropout = 0.0197 m (inter-camera bias = 0.5 m)
# The bias floor limits the Kalman gain so the single-step response is ≈ 0.02 m.
# Gate < 0.15 m — 7.5x the measured value, robust to fixture noise variations.
ANTIJUMP_MAX_STEP_M = 0.15  # m

# Posture agreement: ≥ 95% of frames should show "sitting".  Fixture designed so
# equal view weights give sw=0.522 > sitting=0.452 (test would fail); geometry
# weights (front=0.64, side=1.0) flip to sitting=0.511 (test passes).
# Hysteresis holds frame 0 (1/2 consecutive); measured = 29/30 = 96.7%.
POSTURE_AGREEMENT_MIN = 0.95

# Baseline: eigen-ratio = 4.125 at step 39 (converged equilibrium after 40 frames
# of isotropic Q dilution from the initial sigma_x=0.4m, sigma_y=0.1m R).
# Gate = 3.5 (≫ 1 proves anisotropy survived; safely below the measured 4.13).
OBLIQUE_EIGEN_RATIO_MIN = 3.5

# Baseline: max lag = 0 frames during walk phase; ZUPT speed = 0.000 m/s when stopped.
MOVING_LAG_MAX_FRAMES = 2


def _geo(orientation: OrientationBin) -> ObservationGeometry:
    return ObservationGeometry(
        footpoint_px=(100.0, 200.0),
        floor_residual_m=0.02,
        footpoint_reliable=True,
        detection_confidence=0.9,
        crop_quality=0.9,
        orientation=orientation,
        orientation_confidence=0.9,
    )


# ── Gate tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stationary_variance_drops() -> None:
    """Fused-mode step-to-step jitter ≤ 50% of legacy-mode jitter (stationary person).

    NOTE: The fixture uses exact constant positions per camera (no per-frame noise
    injection), so the "jitter" measured here is Kalman transient + ZUPT engagement,
    not steady-state noise rejection.  The test proves that M04 (anisotropic fusion)
    + M05 (ZUPT) together reduce jitter vs the legacy isotropic/no-ZUPT baseline,
    not that the filter rejects Gaussian pixel noise more tightly.

    Baseline recorded (legacy): step jitter = 0.0279 m
    Baseline recorded (fused):  step jitter = 0.0073 m  (ratio = 0.262)
    Gate: ratio ≤ 0.50
    """
    path = FIXTURES_DIR / "stationary_two_camera.bin"
    assert path.exists(), "Run scripts/synthesize_replay_fixture.py to regenerate fixtures"
    steps = load_fixture(path)

    fused = await run_replay(steps)
    legacy = await run_replay(steps, legacy_mode=True)

    assert legacy.step_jitter_m > 0, "Legacy jitter must be non-zero to compute ratio"
    ratio = fused.step_jitter_m / legacy.step_jitter_m
    assert ratio <= STATIONARY_VARIANCE_DROP_MIN, (
        f"Fused jitter / legacy jitter = {ratio:.3f} (fused={fused.step_jitter_m:.4f} m, "
        f"legacy={legacy.step_jitter_m:.4f} m); expected ratio ≤ {STATIONARY_VARIANCE_DROP_MIN}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stationary_no_systematic_jump_on_camera_dropout() -> None:
    """Anti-jump proof: position does not jump at cam-B dropout.

    When cam-B (-0.2 m offset) disappears, the single-step position change
    must be < ANTIJUMP_MAX_STEP_M (0.15 m), far less than the 0.5 m
    inter-camera systematic bias.  This proves the non-shrinking bias floor
    (M04 / dedup.py) correctly moderates the Kalman gain for cam-A-only frames.

    The bias floor sets R_A_total ≈ 0.0925 m², giving Kalman gain K ≈ 0.10
    from the converged state, so the instantaneous position jump ≈ 0.025 m.
    """
    path = FIXTURES_DIR / "stationary_two_camera.bin"
    assert path.exists(), "Run scripts/synthesize_replay_fixture.py"
    steps = load_fixture(path)
    truth = load_truth(path)

    dropout_step: int = truth.get("cam_b_dropout_at_step", 30)

    result = await run_replay(steps)

    # Dedup must have collapsed both cameras to one PH throughout.
    # If dedup regressed and two PHs spawned, primary_ph selection would silently
    # track one and the jump assertion could pass spuriously.
    unique_ph_ids = {s.ph_id for s in result.steps if s.ph_id is not None}
    assert len(unique_ph_ids) == 1, (
        f"Expected exactly 1 PH (dedup collapsed both cameras); "
        f"got {len(unique_ph_ids)}: {unique_ph_ids}"
    )

    # Find positions immediately before and immediately after dropout.
    pos_before = next(
        (s.position_m for s in reversed(result.steps[:dropout_step]) if s.position_m is not None),
        None,
    )
    pos_after = next(
        (s.position_m for s in result.steps[dropout_step:] if s.position_m is not None),
        None,
    )

    assert pos_before is not None, "No PH position before dropout step"
    assert pos_after is not None, "No PH position after dropout step"

    jump_m = math.sqrt((pos_after[0] - pos_before[0]) ** 2 + (pos_after[1] - pos_before[1]) ** 2)
    assert jump_m < ANTIJUMP_MAX_STEP_M, (
        f"Position jumped {jump_m:.3f} m at cam-B dropout (step {dropout_step}); "
        f"expected < {ANTIJUMP_MAX_STEP_M} m (inter-camera bias = 0.5 m)"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slow_shuffle_speed_preserved() -> None:
    """ZUPT must NOT clamp a 0.30 m/s slow shuffle.

    The ZUPT bands are zupt_speed_enter=0.12 m/s and zupt_speed_exit=0.20 m/s.
    A 0.30 m/s walk is above both thresholds, so ZUPT must never fire.  The
    Kalman-estimated mean speed must be within SLOW_SHUFFLE_SPEED_TOL of truth.

    Baseline recorded: mean_speed ~0.29 m/s after warmup (frames 3+).
    """
    path = FIXTURES_DIR / "slow_shuffle.bin"
    assert path.exists(), "Run scripts/synthesize_replay_fixture.py"
    steps = load_fixture(path)
    truth = load_truth(path)

    truth_speed: float = truth.get("truth_speed_m_s", 0.30)

    result = await run_replay(steps)

    # Skip the first 3 frames (Kalman warmup: velocity state starts at 0).
    steady_speeds = [s.speed_m_s for s in result.steps[3:] if s.position_m is not None]
    assert steady_speeds, "No PH positions in steady-state window"

    mean_speed = sum(steady_speeds) / len(steady_speeds)
    error = abs(mean_speed - truth_speed)
    assert error <= SLOW_SHUFFLE_SPEED_TOL, (
        f"Estimated speed {mean_speed:.3f} m/s differs from truth {truth_speed} m/s "
        f"by {error:.3f} m/s; tolerance {SLOW_SHUFFLE_SPEED_TOL} m/s. "
        "ZUPT may be clamping a 0.30 m/s shuffle — check zupt_speed_exit_m_s."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_oblique_camera_r_anisotropic() -> None:
    """Oblique camera produces an anisotropic posterior covariance.

    The fixture sets floor_cov_random = [0.16, 0, 0, 0.01]  (sigma_x=0.4 m,
    sigma_y=0.1 m, raw R ratio = 16).  After _finalize_singleton adds the bias
    floor and 40 frames of isotropic Q accumulate, the converged eigen-ratio
    is 4.125 (baseline recorded 2026-06-18).

    Gate ≥ 3.5 (≫ 1 proves anisotropy survived; isotropic fallback yields 1).
    """
    path = FIXTURES_DIR / "oblique_single_camera.bin"
    assert path.exists(), "Run scripts/synthesize_replay_fixture.py"
    steps = load_fixture(path)

    result = await run_replay(steps)

    # Skip step 0 (PH spawn: covariance is the isotropic initialize prior,
    # not yet updated with the anisotropic R).  Use the last converged step.
    converged = [s for s in result.steps if s.position_cov_flat is not None and s.step >= 1]
    assert converged, "No PH with anisotropic covariance found after step 0"

    last = converged[-1]
    assert last.position_cov_flat is not None
    ratio = compute_eigen_ratio(last.position_cov_flat)
    assert ratio >= OBLIQUE_EIGEN_RATIO_MIN, (
        f"Position covariance eigen-ratio {ratio:.2f} < {OBLIQUE_EIGEN_RATIO_MIN}; "
        "anisotropic R may not be propagating through the Kalman update."
    )


@pytest.mark.integration
def test_posture_four_camera_agreement() -> None:
    """Four-camera posture fusion: geometry weighting makes 'sitting' win, not 3-vs-1 majority.

    The fixture is designed so that under equal view weights, the high-kp front
    camera (kp=0.95, sw=0.90) dominates and standing_walking wins (sw=0.522 vs
    sitting=0.452).  Only with the actual frontal view-weight penalty (0.64) does
    sitting recover the lead (sitting=0.511 vs sw=0.459).

    Drives GlobalPostureTracker directly with scores from the truth sidecar.
    Setting all cam_weights to 1.0 would cause this test to fail — proving the
    gate cannot pass on majority vote alone.

    Baseline: 96.7% agreement (29/30 frames; frame 0 held pending hysteresis commit).
    Gate: ≥ 95% of frames 0-29 return 'sitting'.
    """
    bin_path = FIXTURES_DIR / "posture_disagreement_four_camera.bin"
    assert bin_path.exists(), "Run scripts/synthesize_replay_fixture.py"
    truth = load_truth(bin_path)

    per_camera_raw: dict = truth.get("per_camera_posture_scores", {})
    posture_truth: str = truth.get("posture_truth", "sitting")

    # Build per-camera PostureScores and view weights from known orientations.
    cam_orientations: dict[str, OrientationBin] = {
        "cam-left": OrientationBin.LEFT,
        "cam-right": OrientationBin.RIGHT,
        "cam-back": OrientationBin.BACK,
        "cam-front": OrientationBin.FRONT,
    }
    cam_scores: dict[str, PostureScores] = {}
    cam_weights: dict[str, float] = {}
    for cam_id, ori in cam_orientations.items():
        raw = per_camera_raw[cam_id]
        cam_scores[cam_id] = PostureScores(
            lying=raw["lying"],
            sitting=raw["sitting"],
            standing_walking=raw["standing_walking"],
            keypoint_confidence=raw["keypoint_confidence"],
        )
        cam_weights[cam_id] = posture_view_weight(_geo(ori))

    all_cameras = list(cam_orientations.keys())
    # cam-left is the "representative" (highest quality in fixture).
    representative = "cam-left"
    non_rep = [c for c in all_cameras if c != representative]

    tracker = GlobalPostureTracker(required_consecutive=2)
    n_frames = 30
    matching = 0

    for _frame in range(n_frames):
        for cam_id in non_rep:
            tracker.record_snapshot(
                "p1",
                cam_id,
                cam_scores[cam_id],
                view_weight=cam_weights[cam_id],
            )
        committed = tracker.update(
            "p1",
            representative,
            cam_scores[representative],
            active_camera_ids=all_cameras,
            view_weight=cam_weights[representative],
        )
        if committed == posture_truth:
            matching += 1

    accuracy = matching / n_frames
    assert accuracy >= POSTURE_AGREEMENT_MIN, (
        f"Posture accuracy {accuracy:.2%} ({matching}/{n_frames} frames matched "
        f"'{posture_truth}'); expected ≥ {POSTURE_AGREEMENT_MIN:.0%}. "
        "Check GlobalPostureTracker view-weight geometry or resolve margin."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_moving_lag_not_regressed() -> None:
    """During the walk phase, position lag is bounded to ≤ MOVING_LAG_MAX_FRAMES.

    The fixture has 20 walk frames (150 mm/step) then 40 stationary frames.
    At each walk step the truth position advances 150 mm; the Kalman estimate
    must track within MOVING_LAG_MAX_FRAMES frames (i.e. at most 2 steps behind).

    Lag is measured as the smallest N such that the estimate at step i+N is
    within one step-width (150 mm) of the truth at step i.

    Also asserts ZUPT engages during the stationary phase (speed drops to near 0
    within zupt_consecutive_frames = 5 frames of the stop).
    """
    path = FIXTURES_DIR / "moving_then_stop.bin"
    assert path.exists(), "Run scripts/synthesize_replay_fixture.py"
    steps = load_fixture(path)
    truth = load_truth(path)

    truth_traj_raw: list = truth.get("truth_trajectory_mm", [])
    walk_end: int = truth.get("walk_end_step", 20)

    truth_traj_m = [(x / 1000.0, y / 1000.0) for x, y in truth_traj_raw]
    result = await run_replay(steps, truth_trajectory=truth_traj_m)

    # ── Walk-phase lag check ───────────────────────────────────────────────────
    step_width_m = 0.150  # 150 mm per frame at 0.3 m/s * 0.5 s
    walk_positions = {s.step: s.position_m for s in result.steps if s.position_m is not None}

    max_lag = 0
    for truth_step in range(2, walk_end):  # skip first 2 (Kalman warmup)
        truth_x, truth_y = truth_traj_m[truth_step]
        for lag in range(MOVING_LAG_MAX_FRAMES + 1):
            est_step = truth_step + lag
            est_pos = walk_positions.get(est_step)
            if est_pos is None:
                continue
            dist = math.sqrt((est_pos[0] - truth_x) ** 2 + (est_pos[1] - truth_y) ** 2)
            if dist <= step_width_m:
                max_lag = max(max_lag, lag)
                break
        else:
            # Could not match within MOVING_LAG_MAX_FRAMES: compute actual lag.
            for lag in range(MOVING_LAG_MAX_FRAMES + 1, len(steps)):
                est_pos = walk_positions.get(truth_step + lag)
                if est_pos is not None:
                    dist = math.sqrt((est_pos[0] - truth_x) ** 2 + (est_pos[1] - truth_y) ** 2)
                    if dist <= step_width_m:
                        max_lag = max(max_lag, lag)
                        break

    assert max_lag <= MOVING_LAG_MAX_FRAMES, (
        f"Walk-phase tracking lag = {max_lag} frames; expected ≤ {MOVING_LAG_MAX_FRAMES}. "
        "Fusion may be too slow to track a 0.30 m/s walk."
    )

    # ── ZUPT engagement check (stationary phase) ──────────────────────────────
    zupt_window_start = walk_end + 5 + 3  # ZUPT fires at frame walk_end+5; check 3 later
    stationary_speeds = [
        s.speed_m_s for s in result.steps[zupt_window_start:] if s.position_m is not None
    ]
    if stationary_speeds:
        mean_stop_speed = sum(stationary_speeds) / len(stationary_speeds)
        # ZUPT should drive velocity close to zero; ≤ zupt_speed_exit_m_s (0.20 m/s).
        assert mean_stop_speed < 0.20, (
            f"Mean speed during stationary phase = {mean_stop_speed:.3f} m/s; "
            "ZUPT may not have engaged.  Check zupt_consecutive_frames and thresholds."
        )
