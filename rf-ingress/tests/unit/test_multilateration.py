"""Unit and property tests for MultilaterationSolver accuracy, covariance, outliers, and R30."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
from cts_contracts.wearables import RangingTechnology
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chi2

from rf_ingress.domain import RangeEpoch, RangeMeasurement, RfNodeGeometry
from rf_ingress.solver.multilateration import MultilaterationConfig, MultilaterationSolver

_T0 = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)


def _four_anchors_3d() -> dict[str, RfNodeGeometry]:
    return {
        "a1": RfNodeGeometry(anchor_id="a1", x_m=0.0, y_m=0.0, z_m=2.4),
        "a2": RfNodeGeometry(anchor_id="a2", x_m=6.0, y_m=0.0, z_m=1.2),
        "a3": RfNodeGeometry(anchor_id="a3", x_m=6.0, y_m=5.0, z_m=2.6),
        "a4": RfNodeGeometry(anchor_id="a4", x_m=0.0, y_m=5.0, z_m=1.4),
    }


def _three_anchors_elevated() -> dict[str, RfNodeGeometry]:
    return {
        "a1": RfNodeGeometry(anchor_id="a1", x_m=0.0, y_m=0.0, z_m=2.4),
        "a2": RfNodeGeometry(anchor_id="a2", x_m=6.0, y_m=0.0, z_m=2.4),
        "a3": RfNodeGeometry(anchor_id="a3", x_m=3.0, y_m=5.0, z_m=2.4),
    }


def _make_epoch(
    anchors: dict[str, RfNodeGeometry],
    true_xyz: tuple[float, float, float],
    *,
    sigma_m: float,
    rng: np.random.Generator | None = None,
    bias_by_anchor: dict[str, float] | None = None,
    azimuth_by_anchor: dict[str, tuple[float, float | None, str]] | None = None,
    technology: RangingTechnology = RangingTechnology.UWB,
) -> RangeEpoch:
    measurements: list[RangeMeasurement] = []
    for aid, node in anchors.items():
        d_true = math.sqrt(
            (true_xyz[0] - node.x_m) ** 2
            + (true_xyz[1] - node.y_m) ** 2
            + (true_xyz[2] - node.z_m) ** 2
        )
        noise = float(rng.normal(0.0, sigma_m)) if rng is not None else 0.0
        bias = (bias_by_anchor or {}).get(aid, 0.0)
        az_deg = None
        az_var = None
        az_frame = None
        if azimuth_by_anchor and aid in azimuth_by_anchor:
            az_deg, az_var, raw_frame = azimuth_by_anchor[aid]
            assert raw_frame in ("device_local", "floor_plan")
            az_frame = raw_frame
        measurements.append(
            RangeMeasurement(
                tag_id="tag-1",
                anchor_id=aid,
                distance_m=max(0.05, d_true + bias + noise),
                variance_m2=sigma_m * sigma_m,
                technology=technology,
                measured_at=_T0,
                azimuth_deg=az_deg,
                azimuth_variance_deg2=az_var,
                azimuth_reference_frame=az_frame,  # type: ignore[arg-type]
                adapter_id="sim-1",
                epoch_seq=1,
            )
        )
    return RangeEpoch(
        tag_id="tag-1",
        adapter_id="sim-1",
        epoch_seq=1,
        measured_at=_T0,
        measurements=tuple(measurements),
    )


def test_monte_carlo_four_anchors_accuracy_and_covariance_consistency() -> None:
    anchors = _four_anchors_3d()
    solver = MultilaterationSolver()
    rng = np.random.default_rng(20261119)
    true_xyz = (2.7, 2.3, 1.0)
    sigma_m = 0.10

    errors_m: list[float] = []
    nees_values: list[float] = []
    for _ in range(1000):
        epoch = _make_epoch(anchors, true_xyz, sigma_m=sigma_m, rng=rng)
        res = solver.solve(epoch, anchors)
        assert res is not None
        err = np.array([res.x_m - true_xyz[0], res.y_m - true_xyz[1]], dtype=np.float64)
        errors_m.append(float(np.linalg.norm(err)))
        cov = np.array(
            [[res.cov_xx_m2, res.cov_xy_m2], [res.cov_xy_m2, res.cov_yy_m2]],
            dtype=np.float64,
        )
        nees_values.append(float(err.T @ np.linalg.inv(cov) @ err))

    median_err_m = float(np.median(errors_m))
    mean_nees = float(np.mean(nees_values))
    lo_95 = float(chi2.ppf(0.025, 2))
    hi_95 = float(chi2.ppf(0.975, 2))

    assert median_err_m < 0.12, f"median horizontal error {median_err_m:.4f} m >= 0.12 m"
    assert lo_95 <= mean_nees <= hi_95, (
        f"mean NEES {mean_nees:.3f} outside 95% chi2(2) [{lo_95:.3f}, {hi_95:.3f}]"
    )
    # Tighter check: covariance is genuinely consistent, not grossly inflated or shrunken.
    assert 1.2 <= mean_nees <= 2.6, (
        f"mean NEES {mean_nees:.3f} indicates poorly calibrated covariance"
    )


def test_three_anchors_2d_mode_inflates_covariance_for_height_uncertainty() -> None:
    anchors = _three_anchors_elevated()
    true_xyz = (2.8, 2.1, 1.0)
    epoch = _make_epoch(anchors, true_xyz, sigma_m=0.10)

    solver_with_height = MultilaterationSolver(
        MultilaterationConfig(assumed_tag_height_m=1.0, tag_height_sigma_m=0.30)
    )
    solver_no_height = MultilaterationSolver(
        MultilaterationConfig(assumed_tag_height_m=1.0, tag_height_sigma_m=0.0)
    )

    res_with = solver_with_height.solve(epoch, anchors)
    res_without = solver_no_height.solve(epoch, anchors)
    assert res_with is not None and res_without is not None
    assert res_with.mode_3d is False
    assert res_with.z_m is None
    assert math.hypot(res_with.x_m - true_xyz[0], res_with.y_m - true_xyz[1]) < 0.005
    assert res_with.horizontal_sigma_m > res_without.horizontal_sigma_m * 1.25


def test_collinear_anchors_yield_no_fix() -> None:
    collinear = {
        "a1": RfNodeGeometry(anchor_id="a1", x_m=0.0, y_m=2.0, z_m=2.0),
        "a2": RfNodeGeometry(anchor_id="a2", x_m=3.0, y_m=2.0, z_m=2.0),
        "a3": RfNodeGeometry(anchor_id="a3", x_m=6.0, y_m=2.0, z_m=2.0),
    }
    epoch = _make_epoch(collinear, (3.0, 4.0, 1.0), sigma_m=0.10)
    solver = MultilaterationSolver()
    assert solver.solve(epoch, collinear) is None


def test_single_anchor_nlos_outlier_is_rejected_and_error_stays_under_0_3m() -> None:
    anchors = _four_anchors_3d()
    true_xyz = (3.1, 2.4, 1.0)
    epoch = _make_epoch(
        anchors,
        true_xyz,
        sigma_m=0.10,
        bias_by_anchor={"a3": 1.0},
    )
    solver = MultilaterationSolver()
    res = solver.solve(epoch, anchors)
    assert res is not None
    assert res.dropped_outlier_anchor_id == "a3"
    assert res.anchors_used == 3
    err_m = math.hypot(res.x_m - true_xyz[0], res.y_m - true_xyz[1])
    assert err_m < 0.30, f"outlier-rejected error {err_m:.4f} m >= 0.30 m"


def test_wide_ble_rssi_fixes_are_published_with_true_covariance() -> None:
    anchors = _four_anchors_3d()
    epoch = _make_epoch(
        anchors,
        (3.0, 2.5, 1.0),
        sigma_m=2.5,
        technology=RangingTechnology.BLE_RSSI,
    )
    solver = MultilaterationSolver()
    res = solver.solve(epoch, anchors)
    assert res is not None
    assert res.horizontal_sigma_m > 1.5
    assert res.cov_xx_m2 * res.cov_yy_m2 - res.cov_xy_m2**2 > 0.0


def test_device_local_azimuth_is_bit_identical_to_none_on_3_and_4_anchors() -> None:
    solver = MultilaterationSolver()
    for anchors in (_three_anchors_elevated(), _four_anchors_3d()):
        true_xyz = (2.9, 2.2, 1.0)
        epoch_none = _make_epoch(anchors, true_xyz, sigma_m=0.10)
        epoch_local = _make_epoch(
            anchors,
            true_xyz,
            sigma_m=0.10,
            azimuth_by_anchor={
                aid: (float(-140 + 73 * idx), 25.0, "device_local")
                for idx, aid in enumerate(anchors)
            },
        )
        res_none = solver.solve(epoch_none, anchors)
        res_local = solver.solve(epoch_local, anchors)
        assert res_none is not None and res_local is not None
        assert round(res_local.x_m * 1000.0) == round(res_none.x_m * 1000.0)
        assert round(res_local.y_m * 1000.0) == round(res_none.y_m * 1000.0)
        assert res_local.x_m == res_none.x_m
        assert res_local.y_m == res_none.y_m
        assert res_local.cov_xx_m2 == res_none.cov_xx_m2
        assert res_local.cov_xy_m2 == res_none.cov_xy_m2
        assert res_local.cov_yy_m2 == res_none.cov_yy_m2


def test_two_anchors_solve_only_with_floor_plan_bearing() -> None:
    two_anchors = {
        "a1": RfNodeGeometry(anchor_id="a1", x_m=0.0, y_m=0.0, z_m=2.0),
        "a2": RfNodeGeometry(anchor_id="a2", x_m=6.0, y_m=0.0, z_m=2.0),
    }
    true_xyz = (2.5, 3.0, 1.0)
    solver = MultilaterationSolver()

    epoch_no_bearing = _make_epoch(two_anchors, true_xyz, sigma_m=0.10)
    assert solver.solve(epoch_no_bearing, two_anchors) is None

    epoch_device_local = _make_epoch(
        two_anchors,
        true_xyz,
        sigma_m=0.10,
        azimuth_by_anchor={"a1": (45.0, 16.0, "device_local")},
    )
    assert solver.solve(epoch_device_local, two_anchors) is None

    true_bearing_deg = math.degrees(math.atan2(true_xyz[1] - 0.0, true_xyz[0] - 0.0))
    epoch_floor_bearing = _make_epoch(
        two_anchors,
        true_xyz,
        sigma_m=0.10,
        azimuth_by_anchor={"a1": (true_bearing_deg, 9.0, "floor_plan")},
    )
    res = solver.solve(epoch_floor_bearing, two_anchors)
    assert res is not None
    assert math.hypot(res.x_m - true_xyz[0], res.y_m - true_xyz[1]) < 0.05


@settings(max_examples=60, deadline=None)
@given(
    x_m=st.floats(min_value=0.5, max_value=5.5, allow_nan=False, allow_infinity=False),
    y_m=st.floats(min_value=0.5, max_value=4.5, allow_nan=False, allow_infinity=False),
)
def test_property_noise_free_inside_hull_error_under_1mm(x_m: float, y_m: float) -> None:
    anchors = _four_anchors_3d()
    solver = MultilaterationSolver()
    epoch = _make_epoch(anchors, (x_m, y_m, 1.0), sigma_m=0.10)
    res = solver.solve(epoch, anchors)
    assert res is not None
    err_m = math.hypot(res.x_m - x_m, res.y_m - y_m)
    assert err_m < 0.001, f"noise-free error {err_m:.6f} m >= 1 mm at ({x_m}, {y_m})"
