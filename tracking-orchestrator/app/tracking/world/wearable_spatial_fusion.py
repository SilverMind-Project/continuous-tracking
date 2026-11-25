"""Loosely coupled fusion of trajectory-attested wearable position fixes into PH state.

Pure functions: no I/O, no wall clock, no tracker state. ``WorldTracker`` decides which PH
may be fused (open, trajectory-attested this round, camera-tracked or briefly occluded)
and owns persistence; this module selects the fix and computes the gated update.

Time model. A PH's stored state is the posterior at ``last_seen_at``. A fix older than
that (the usual case: a watch publishes 3 s micro-batches) is moved to the state time
along the PH velocity instead of rewinding the filter, and the velocity uncertainty over
that gap is added to its covariance, so the stored state stays the posterior at
``last_seen_at``. A fix newer than the state time (a camera-occluded PH) predicts the state
to the fix time first; the caller then advances ``last_seen_at`` to it.

Covariance. Every fused fix carries its own random covariance plus the configured
systematic floor ``b^2 I``. Treating that floor as independent across fixes would still
let a run of fixes shrink the PH covariance far below it, so after each update the
smallest position eigenvalue is lifted back to ``min(b^2, its value before the update)``:
wearable fixes never make a PH more certain than the wearable's systematic error, while
camera updates may.

This module never branches on radio technology: eligibility is fix kind, covariance
major-axis sigma, and time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from cts_contracts import FixKind

from app.domain import WearableFix, tuple_to_cov2x2
from app.tracking.world.kalman import (
    KalmanState,
    apply_position_floor,
    isotropic_cov,
    mahalanobis2_position,
    predict,
    update,
)
from app.tracking.world.wearable_association import covariance_major_sigma_m
from app.tracking.world.wearable_config import WearableConfig


@dataclass(frozen=True)
class MotionModel:
    """The world tracker's Kalman prediction parameters."""

    process_noise_accel_m_s2: float
    velocity_decay_s: float
    damped: bool


@dataclass(frozen=True)
class FusionResult:
    """One fix's fusion outcome against one PH state.

    ``state`` is the posterior (at ``state.updated_at``) when ``gated`` is False, else
    ``None``. ``shift_m`` is how far the update moved the position estimate.
    """

    gated: bool
    state: KalmanState | None
    shift_m: float
    mahalanobis2: float


def select_fusion_fix(
    fixes: Sequence[WearableFix],
    *,
    tag_id: str,
    state_time: datetime,
    fused_through: datetime | None,
    allow_newer: bool,
    cfg: WearableConfig,
) -> WearableFix | None:
    """Return the newest fix of *tag_id* that may be fused into a PH state at *state_time*.

    Eligible: a position fix with finite covariance no wider than
    ``spatial_fusion_max_sigma_m``, newer than *fused_through* (each fix is fused once),
    within ``spatial_fusion_max_dt_s`` of the state time, and not newer than the state
    unless *allow_newer* (only a camera-occluded PH may be advanced by a fix).
    """
    best: WearableFix | None = None
    for fix in fixes:
        if fix.tag_id != tag_id or fix.kind != FixKind.POSITION:
            continue
        if fix.x_m is None or fix.y_m is None or fix.cov_m2 is None:
            continue
        if not all(math.isfinite(v) for v in (fix.x_m, fix.y_m, *fix.cov_m2)):
            continue
        if covariance_major_sigma_m(fix.cov_m2) > cfg.spatial_fusion_max_sigma_m:
            continue
        if fused_through is not None and fix.measured_at <= fused_through:
            continue
        dt_s = (fix.measured_at - state_time).total_seconds()
        if abs(dt_s) > cfg.spatial_fusion_max_dt_s:
            continue
        if dt_s > 0.0 and not allow_newer:
            continue
        if best is None or fix.measured_at > best.measured_at:
            best = fix
    return best


def fuse_position_fix(
    state: KalmanState,
    fix: WearableFix,
    *,
    cfg: WearableConfig,
    motion: MotionModel,
) -> FusionResult:
    """Gate and apply one position fix to *state* (see the module docstring)."""
    assert fix.x_m is not None
    assert fix.y_m is not None
    assert fix.cov_m2 is not None
    r = tuple_to_cov2x2(fix.cov_m2) + isotropic_cov(cfg.spatial_fusion_bias_floor_m)

    if fix.measured_at > state.updated_at:
        prior = predict(
            state,
            fix.measured_at,
            motion.process_noise_accel_m_s2,
            motion.velocity_decay_s,
            damped=motion.damped,
        )
        z_x, z_y = fix.x_m, fix.y_m
    else:
        prior = state
        lag_s = (state.updated_at - fix.measured_at).total_seconds()
        # Move the fix to the state time along the PH velocity; the velocity's own
        # uncertainty over the lag widens the shifted measurement.
        z_x = fix.x_m + float(state.mean[2]) * lag_s
        z_y = fix.y_m + float(state.mean[3]) * lag_s
        r = r + (lag_s * lag_s) * np.asarray(state.covariance, dtype=np.float64)[2:4, 2:4]

    d2 = mahalanobis2_position(prior, z_x, z_y, r)
    if not math.isfinite(d2) or d2 > cfg.spatial_fusion_gate_chi2:
        return FusionResult(gated=True, state=None, shift_m=0.0, mahalanobis2=d2)

    posterior = update(prior, z_x, z_y, r)
    posterior = apply_position_floor(prior, posterior, cfg.spatial_fusion_bias_floor_m**2)
    shift_m = math.hypot(
        float(posterior.mean[0] - prior.mean[0]),
        float(posterior.mean[1] - prior.mean[1]),
    )
    return FusionResult(gated=False, state=posterior, shift_m=shift_m, mahalanobis2=d2)
