"""Pure numpy/scipy multilateration solver with covariance and outlier rejection.

The solver weights every range measurement strictly by its reported
``variance_m2`` and never branches on adapter or radio technology names.
Angular telemetry with ``azimuth_reference_frame != "floor_plan"`` (including
``"device_local"`` wrist azimuth) is completely ignored so distance-only solves
remain rotationally invariant to wearer wrist orientation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.stats import chi2

from rf_ingress.domain import RangeEpoch, RangeMeasurement, RfNodeGeometry, SolveResult


@dataclass(frozen=True)
class MultilaterationConfig:
    """Configuration for :class:`MultilaterationSolver`."""

    assumed_tag_height_m: float = 1.0
    tag_height_sigma_m: float = 0.3
    min_vertical_spread_m: float = 0.5
    max_condition: float = 1000.0
    quality_scale_m: float = 0.5
    # Keep None by default so wide-covariance fixes (e.g. BLE RSSI) are still
    # published with their true covariance rather than dropped at ingress.
    max_sigma_m: float | None = None
    outlier_confidence: float = 0.99
    soft_l1_f_scale: float = 1.5


def _wrap_angle_rad(delta_rad: float) -> float:
    return math.atan2(math.sin(delta_rad), math.cos(delta_rad))


def _initial_guess(
    anchors_xyz: NDArray[np.float64],
    distances_m: NDArray[np.float64],
    variances_m2: NDArray[np.float64],
    *,
    mode_3d: bool,
    assumed_z_m: float,
) -> NDArray[np.float64]:
    weights = 1.0 / (variances_m2 * np.maximum(distances_m, 0.25))
    weight_sum = float(np.sum(weights))
    centroid = np.sum(anchors_xyz * weights[:, None], axis=0) / max(weight_sum, 1e-12)
    if mode_3d:
        return np.array([centroid[0], centroid[1], assumed_z_m], dtype=np.float64)
    return np.array([centroid[0], centroid[1]], dtype=np.float64)


def _anchors_horizontally_collinear(anchors_xyz: NDArray[np.float64]) -> bool:
    """Return True when anchor (x, y) positions collapse onto a 1D line.

    Collinear anchors produce an exact mirror ambiguity across the anchor line
    even when the local range Jacobian at an off-line point is full rank.
    """
    xy = anchors_xyz[:, :2]
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if len(singular_values) < 2 or float(singular_values[0]) < 1e-6:
        return True
    return bool(float(singular_values[1]) / float(singular_values[0]) < 1e-3)


def _effective_2d_sigmas(
    anchors_z_m: NDArray[np.float64],
    distances_m: NDArray[np.float64],
    variances_m2: NDArray[np.float64],
    assumed_z_m: float,
    tag_height_sigma_m: float,
) -> NDArray[np.float64]:
    if tag_height_sigma_m <= 0.0:
        return np.sqrt(variances_m2)
    dz = assumed_z_m - anchors_z_m
    denom = np.maximum(distances_m, 0.1)
    jz = dz / denom
    var_eff = variances_m2 + (jz * tag_height_sigma_m) ** 2
    return np.sqrt(var_eff)


class MultilaterationSolver:
    """Per-epoch nonlinear least-squares multilateration solver."""

    def __init__(self, config: MultilaterationConfig | None = None) -> None:
        self._cfg: MultilaterationConfig = config or MultilaterationConfig()

    def solve(
        self,
        epoch: RangeEpoch,
        nodes: Mapping[str, RfNodeGeometry],
    ) -> SolveResult | None:
        """Solve one range epoch against known RF node geometries.

        Returns ``None`` when fewer than 3 nodes are available (unless 2 nodes
        carry calibrated ``floor_plan`` bearings), when optimization fails to
        converge, or when node geometry is degenerate/collinear.
        """
        valid: list[tuple[RangeMeasurement, RfNodeGeometry]] = []
        for measurement in epoch.measurements:
            node = nodes.get(measurement.anchor_id)
            if node is not None:
                valid.append((measurement, node))

        if len(valid) < 2:
            return None

        if len(valid) == 2:
            return self._solve_two_anchor_with_floor_plan_bearing(valid)

        return self._solve_distance_only(valid)

    def _solve_distance_only(
        self,
        valid: list[tuple[RangeMeasurement, RfNodeGeometry]],
    ) -> SolveResult | None:
        n_anchors = len(valid)
        z_values = [node.z_m for _, node in valid]
        vertical_spread_m = max(z_values) - min(z_values)
        try_3d = n_anchors >= 4 and vertical_spread_m >= self._cfg.min_vertical_spread_m

        if try_3d:
            result_3d = self._solve_subset(valid, mode_3d=True, allow_outlier_drop=True)
            if result_3d is not None:
                return result_3d

        return self._solve_subset(valid, mode_3d=False, allow_outlier_drop=True)

    def _solve_subset(
        self,
        valid: list[tuple[RangeMeasurement, RfNodeGeometry]],
        *,
        mode_3d: bool,
        allow_outlier_drop: bool,
        dropped_anchor_id: str | None = None,
    ) -> SolveResult | None:
        n_anchors = len(valid)
        n_params = 3 if mode_3d else 2
        if n_anchors < 3 or n_anchors < n_params:
            return None

        anchors_xyz = np.array([[n.x_m, n.y_m, n.z_m] for _, n in valid], dtype=np.float64)
        if _anchors_horizontally_collinear(anchors_xyz):
            return None
        distances_m = np.array([m.distance_m for m, _ in valid], dtype=np.float64)
        variances_m2 = np.array([m.variance_m2 for m, _ in valid], dtype=np.float64)

        if mode_3d:
            sigmas_m = np.sqrt(variances_m2)
        else:
            sigmas_m = _effective_2d_sigmas(
                anchors_xyz[:, 2],
                distances_m,
                variances_m2,
                self._cfg.assumed_tag_height_m,
                self._cfg.tag_height_sigma_m,
            )

        x0 = _initial_guess(
            anchors_xyz,
            distances_m,
            variances_m2,
            mode_3d=mode_3d,
            assumed_z_m=self._cfg.assumed_tag_height_m,
        )

        def residuals(state: NDArray[np.float64]) -> NDArray[np.float64]:
            if mode_3d:
                diff = state[None, :] - anchors_xyz
            else:
                state_3d = np.array(
                    [state[0], state[1], self._cfg.assumed_tag_height_m],
                    dtype=np.float64,
                )
                diff = state_3d[None, :] - anchors_xyz
            pred = np.linalg.norm(diff, axis=1)
            return (pred - distances_m) / sigmas_m

        # Use soft_l1 when outlier rejection is still eligible so a single bad
        # anchor cannot drag the initial solution away from the inlier consensus.
        loss_name = "soft_l1" if allow_outlier_drop and n_anchors >= 4 else "linear"
        opt = least_squares(
            residuals,
            x0,
            method="trf",
            loss=loss_name,
            f_scale=self._cfg.soft_l1_f_scale,
            max_nfev=200,
        )
        if not opt.success or not np.all(np.isfinite(opt.x)):
            return None

        state_hat = opt.x.astype(np.float64)
        if mode_3d:
            pos_3d = state_hat
        else:
            pos_3d = np.array(
                [state_hat[0], state_hat[1], self._cfg.assumed_tag_height_m],
                dtype=np.float64,
            )

        diff_hat = pos_3d[None, :] - anchors_xyz
        pred_dist = np.linalg.norm(diff_hat, axis=1)
        if not mode_3d and self._cfg.tag_height_sigma_m > 0.0:
            # Re-evaluate vertical-Jacobian variance inflation at the converged point.
            sigmas_m = _effective_2d_sigmas(
                anchors_xyz[:, 2],
                pred_dist,
                variances_m2,
                self._cfg.assumed_tag_height_m,
                self._cfg.tag_height_sigma_m,
            )

        raw_res = pred_dist - distances_m
        std_res = raw_res / sigmas_m
        chi2_val = float(np.sum(std_res**2))
        dof = n_anchors - n_params

        # Single-outlier rejection: requires >= 4 anchors so >= 3 remain.
        # In 4-anchor 3D mode (dof == 1), shifting z can absorb a single bad range
        # onto another anchor; evaluate leave-one-out subsets and pick the lowest
        # residual consensus so the true outlier anchor is isolated reliably.
        if allow_outlier_drop and n_anchors >= 4 and dof > 0:
            chi2_crit = float(chi2.ppf(self._cfg.outlier_confidence, dof))
            if chi2_val > chi2_crit:
                candidate_indices = (
                    list(range(n_anchors))
                    if n_anchors <= 8
                    else [int(idx) for idx in np.argsort(-np.abs(std_res))[:4]]
                )
                best_result: SolveResult | None = None
                for idx in candidate_indices:
                    reduced = [pair for i, pair in enumerate(valid) if i != idx]
                    next_mode_3d = mode_3d and len(reduced) >= 4
                    cand = self._solve_subset(
                        reduced,
                        mode_3d=next_mode_3d,
                        allow_outlier_drop=False,
                        dropped_anchor_id=valid[idx][1].anchor_id,
                    )
                    if cand is not None and (
                        best_result is None or cand.residual_rms_m < best_result.residual_rms_m
                    ):
                        best_result = cand
                return best_result

        # Analytic whitened range Jacobian: J_w[i] = ((p - a_i) / r_i) / sigma_i.
        safe_dist = np.maximum(pred_dist, 1e-6)
        jac_full = diff_hat / (safe_dist[:, None] * sigmas_m[:, None])
        jac = jac_full if mode_3d else jac_full[:, :2]

        info = jac.T @ jac
        if not np.all(np.isfinite(info)):
            return None
        cond = float(np.linalg.cond(info))
        if not math.isfinite(cond) or cond > self._cfg.max_condition:
            return None

        try:
            cov_unscaled = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            return None

        chi2_red = (chi2_val / dof) if dof > 0 else 1.0
        scale = max(1.0, chi2_red)
        cov_2d = cov_unscaled[:2, :2] * scale
        cov_xx = float(cov_2d[0, 0])
        cov_xy = float(0.5 * (cov_2d[0, 1] + cov_2d[1, 0]))
        cov_yy = float(cov_2d[1, 1])

        if not (math.isfinite(cov_xx) and math.isfinite(cov_xy) and math.isfinite(cov_yy)):
            return None
        det = cov_xx * cov_yy - cov_xy * cov_xy
        if cov_xx <= 0.0 or cov_yy <= 0.0 or det <= 0.0:
            return None

        horizontal_sigma_m = math.sqrt(0.5 * (cov_xx + cov_yy))
        if self._cfg.max_sigma_m is not None and horizontal_sigma_m > self._cfg.max_sigma_m:
            return None

        quality = max(0.0, min(1.0, math.exp(-horizontal_sigma_m / self._cfg.quality_scale_m)))
        residual_rms_m = float(math.sqrt(float(np.mean(raw_res**2))))

        return SolveResult(
            x_m=float(pos_3d[0]),
            y_m=float(pos_3d[1]),
            z_m=float(pos_3d[2]) if mode_3d else None,
            cov_xx_m2=cov_xx,
            cov_xy_m2=cov_xy,
            cov_yy_m2=cov_yy,
            horizontal_sigma_m=horizontal_sigma_m,
            quality=quality,
            anchors_used=n_anchors,
            residual_rms_m=residual_rms_m,
            dropped_outlier_anchor_id=dropped_anchor_id,
            mode_3d=mode_3d,
        )

    def _solve_two_anchor_with_floor_plan_bearing(
        self,
        valid: list[tuple[RangeMeasurement, RfNodeGeometry]],
    ) -> SolveResult | None:
        """Optional 2-anchor 2D solve when fixed anchors report ``floor_plan`` bearings."""
        bearing_entries: list[tuple[RfNodeGeometry, float, float]] = []
        for meas, node in valid:
            if (
                meas.azimuth_reference_frame == "floor_plan"
                and meas.azimuth_deg is not None
                and meas.azimuth_variance_deg2 is not None
                and math.isfinite(meas.azimuth_deg)
                and math.isfinite(meas.azimuth_variance_deg2)
                and meas.azimuth_variance_deg2 > 0.0
            ):
                theta_rad = math.radians(meas.azimuth_deg)
                sigma_rad = math.radians(math.sqrt(meas.azimuth_variance_deg2))
                bearing_entries.append((node, theta_rad, sigma_rad))

        if not bearing_entries:
            return None

        anchors_xyz = np.array([[n.x_m, n.y_m, n.z_m] for _, n in valid], dtype=np.float64)
        distances_m = np.array([m.distance_m for m, _ in valid], dtype=np.float64)
        variances_m2 = np.array([m.variance_m2 for m, _ in valid], dtype=np.float64)
        sigmas_m = _effective_2d_sigmas(
            anchors_xyz[:, 2],
            distances_m,
            variances_m2,
            self._cfg.assumed_tag_height_m,
            self._cfg.tag_height_sigma_m,
        )

        # Initialize along the first anchor's floor-plan bearing ray.
        first_node, first_theta, _ = bearing_entries[0]
        first_dist = next(m.distance_m for m, n in valid if n.anchor_id == first_node.anchor_id)
        horiz_r = math.sqrt(
            max(0.01, first_dist**2 - (self._cfg.assumed_tag_height_m - first_node.z_m) ** 2)
        )
        x0 = np.array(
            [
                first_node.x_m + horiz_r * math.cos(first_theta),
                first_node.y_m + horiz_r * math.sin(first_theta),
            ],
            dtype=np.float64,
        )

        def residuals(state: NDArray[np.float64]) -> NDArray[np.float64]:
            state_3d = np.array(
                [state[0], state[1], self._cfg.assumed_tag_height_m], dtype=np.float64
            )
            pred_d = np.linalg.norm(state_3d[None, :] - anchors_xyz, axis=1)
            range_res = (pred_d - distances_m) / sigmas_m
            bearing_res = np.array(
                [
                    _wrap_angle_rad(math.atan2(state[1] - node.y_m, state[0] - node.x_m) - theta)
                    / sigma_theta
                    for node, theta, sigma_theta in bearing_entries
                ],
                dtype=np.float64,
            )
            return np.concatenate([range_res, bearing_res])

        opt = least_squares(
            residuals,
            x0,
            method="trf",
            loss="soft_l1",
            f_scale=self._cfg.soft_l1_f_scale,
            max_nfev=200,
        )
        if not opt.success or not np.all(np.isfinite(opt.x)):
            return None

        state_hat = opt.x.astype(np.float64)
        pos_3d = np.array(
            [state_hat[0], state_hat[1], self._cfg.assumed_tag_height_m], dtype=np.float64
        )
        diff_hat = pos_3d[None, :] - anchors_xyz
        pred_dist = np.linalg.norm(diff_hat, axis=1)
        raw_range_res = pred_dist - distances_m

        # Build combined range + bearing Jacobian.
        safe_dist = np.maximum(pred_dist, 1e-6)
        jac_rows: list[NDArray[np.float64]] = [
            diff_hat[i, :2] / (safe_dist[i] * sigmas_m[i]) for i in range(len(valid))
        ]
        for node, _theta, sigma_theta in bearing_entries:
            dx = float(state_hat[0] - node.x_m)
            dy = float(state_hat[1] - node.y_m)
            r2 = max(dx * dx + dy * dy, 1e-6)
            jac_rows.append(np.array([-dy / r2, dx / r2], dtype=np.float64) / sigma_theta)
        jac = np.vstack(jac_rows)

        info = jac.T @ jac
        cond = float(np.linalg.cond(info))
        if not math.isfinite(cond) or cond > self._cfg.max_condition:
            return None
        try:
            cov_unscaled = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            return None

        std_res = residuals(state_hat)
        dof = len(std_res) - 2
        chi2_red = (float(np.sum(std_res**2)) / dof) if dof > 0 else 1.0
        cov_2d = cov_unscaled * max(1.0, chi2_red)
        cov_xx = float(cov_2d[0, 0])
        cov_xy = float(0.5 * (cov_2d[0, 1] + cov_2d[1, 0]))
        cov_yy = float(cov_2d[1, 1])
        det = cov_xx * cov_yy - cov_xy * cov_xy
        if not (math.isfinite(cov_xx) and math.isfinite(cov_xy) and math.isfinite(cov_yy)):
            return None
        if cov_xx <= 0.0 or cov_yy <= 0.0 or det <= 0.0:
            return None

        horizontal_sigma_m = math.sqrt(0.5 * (cov_xx + cov_yy))
        if self._cfg.max_sigma_m is not None and horizontal_sigma_m > self._cfg.max_sigma_m:
            return None
        quality = max(0.0, min(1.0, math.exp(-horizontal_sigma_m / self._cfg.quality_scale_m)))
        residual_rms_m = float(math.sqrt(float(np.mean(raw_range_res**2))))

        return SolveResult(
            x_m=float(state_hat[0]),
            y_m=float(state_hat[1]),
            z_m=None,
            cov_xx_m2=cov_xx,
            cov_xy_m2=cov_xy,
            cov_yy_m2=cov_yy,
            horizontal_sigma_m=horizontal_sigma_m,
            quality=quality,
            anchors_used=len(valid),
            residual_rms_m=residual_rms_m,
            dropped_outlier_anchor_id=None,
            mode_3d=False,
        )
