"""Fusion measurement harness for M09 acceptance gate tests.

Replays WorldObservation fixtures through WorldTracker.step() and returns
per-step position data and aggregate metrics.  Provides a legacy-mode toggle
that strips anisotropic covariance and disables ZUPT, producing the old-fusion
baseline so tests can assert a quantified improvement delta.

Legacy mode is a TEST-ONLY switch — it is NOT a production flag.

Usage::

    from tests.integration._fusion_metrics import run_replay, ReplayResult
    from tests.integration._replay import FIXTURES_DIR, load_fixture

    steps = load_fixture(FIXTURES_DIR / "stationary_two_camera.bin")
    result = await run_replay(steps)
    legacy = await run_replay(steps, legacy_mode=True)
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker
from tests.integration._replay import _ROOM_POLYGONS  # noqa: F401 — re-exported for test use

# Extend the room polygon to cover all fixture coordinate ranges.
_ROOM_POLYGONS_WIDE: dict[str, list[tuple[float, float]]] = {
    "living_room": [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
}

_BASE_TIME = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)
_FRAME_INTERVAL_S = 0.5


@dataclass
class StepResult:
    """Tracker output for one WorldTracker.step() call."""

    step: int
    ph_id: str | None
    position_m: tuple[float, float] | None  # (x, y) in metres
    # 16-element row-major 4x4 Kalman covariance; None when no PH exists.
    position_cov_flat: tuple[float, ...] | None
    speed_m_s: float
    active_cameras: frozenset[str]


@dataclass
class ReplayResult:
    """Full replay output: per-step data plus aggregate metrics."""

    steps: list[StepResult]

    # Aggregate metrics (computed from steps that have a PH position).
    position_rms_m: float  # RMS error vs truth_trajectory (NaN when no truth)
    step_jitter_m: float  # RMS step-to-step position displacement
    mean_speed_m_s: float  # mean Kalman-estimated floor speed


def compute_eigen_ratio(ph_cov_flat: tuple[float, ...]) -> float:
    """Return the eigen-ratio (λ_max / λ_min) of the 2x2 position block."""
    cov4x4 = np.array(ph_cov_flat, dtype=np.float64).reshape(4, 4)
    pos_block = cov4x4[:2, :2]
    evals = np.linalg.eigvalsh(pos_block)
    evals = np.abs(evals)
    min_e = float(np.min(evals))
    if min_e < 1e-12:
        return float("inf")
    return float(np.max(evals) / min_e)


async def run_replay(
    steps: list[list],
    cfg: WorldTrackerConfig | None = None,
    *,
    legacy_mode: bool = False,
    truth_trajectory: list[tuple[float, float]] | None = None,
    room_polygons: dict[str, list[tuple[float, float]]] | None = None,
) -> ReplayResult:
    """Replay fixture steps through WorldTracker and collect per-step results.

    Args:
        steps:            Output of load_fixture() — list of per-frame observation lists.
        cfg:              WorldTrackerConfig override (defaults to WorldTrackerConfig()).
        legacy_mode:      When True, strip floor_cov_random from all observations
                          (forcing isotropic fallback) and disable ZUPT by setting
                          zupt_consecutive_frames to a huge value.  TEST-ONLY.
        truth_trajectory: Optional list of (x_m, y_m) ground-truth positions, one per
                          step.  When provided, position_rms_m is computed.
        room_polygons:    Room polygon dict; defaults to a 30x30 m room covering all
                          built-in fixture coordinates.
    """
    if room_polygons is None:
        room_polygons = _ROOM_POLYGONS_WIDE

    if cfg is None:
        cfg = WorldTrackerConfig()

    if legacy_mode:
        # Disable ZUPT: require far more consecutive frames than any fixture has.
        cfg = dataclasses.replace(cfg, zupt_consecutive_frames=99999)

    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    step_results: list[StepResult] = []

    for i, frame_obs in enumerate(steps):
        now = _BASE_TIME + timedelta(seconds=i * _FRAME_INTERVAL_S)

        if legacy_mode:
            # Strip anisotropic covariance: forces isotropic_cov(observation_noise_m).
            frame_obs = [dataclasses.replace(obs, floor_cov_random=None) for obs in frame_obs]

        result = await tracker.step(
            observations=frame_obs,
            now=now,
            room_polygons=room_polygons,
        )

        # Identify the primary PH for this step (highest observation count).
        primary_ph = None
        if result.updated_phs:
            primary_ph = max(result.updated_phs, key=lambda ph: ph.observation_count)

        if primary_ph is not None:
            x_m = float(primary_ph.state_mean[0])
            y_m = float(primary_ph.state_mean[1])
            speed = float(primary_ph.last_floor_speed_m_s)
            cov_flat: tuple[float, ...] = tuple(float(v) for v in primary_ph.state_cov)
            cameras = frozenset(primary_ph.active_cameras)
            step_results.append(
                StepResult(
                    step=i,
                    ph_id=primary_ph.ph_id,
                    position_m=(x_m, y_m),
                    position_cov_flat=cov_flat,
                    speed_m_s=speed,
                    active_cameras=cameras,
                )
            )
        else:
            step_results.append(
                StepResult(
                    step=i,
                    ph_id=None,
                    position_m=None,
                    position_cov_flat=None,
                    speed_m_s=0.0,
                    active_cameras=frozenset(),
                )
            )

    # ── aggregate metrics ──────────────────────────────────────────────────────
    positions = [s.position_m for s in step_results if s.position_m is not None]

    # RMS vs truth trajectory
    rms = float("nan")
    if truth_trajectory is not None and positions:
        sq_errors: list[float] = []
        for s in step_results:
            if s.position_m is None:
                continue
            idx = s.step
            if idx >= len(truth_trajectory):
                continue
            tx, ty = truth_trajectory[idx]
            dx = s.position_m[0] - tx
            dy = s.position_m[1] - ty
            sq_errors.append(dx * dx + dy * dy)
        rms = math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else float("nan")

    # Step-to-step jitter (consecutive displacement)
    jitter = 0.0
    if len(positions) >= 2:
        sq_jumps: list[float] = []
        prev = positions[0]
        for cur in positions[1:]:
            dx = cur[0] - prev[0]
            dy = cur[1] - prev[1]
            sq_jumps.append(dx * dx + dy * dy)
            prev = cur
        jitter = math.sqrt(sum(sq_jumps) / len(sq_jumps)) if sq_jumps else 0.0

    # Mean Kalman speed
    speeds = [s.speed_m_s for s in step_results if s.position_m is not None]
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0

    return ReplayResult(
        steps=step_results,
        position_rms_m=rms,
        step_jitter_m=jitter,
        mean_speed_m_s=mean_speed,
    )
