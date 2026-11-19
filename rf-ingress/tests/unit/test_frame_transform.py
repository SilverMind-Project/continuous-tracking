"""Unit tests for 2D similarity frame transform fitting and covariance propagation."""

from __future__ import annotations

import math

import pytest

from rf_ingress.solver.frame_transform import fit_similarity_transform


def test_fit_recovers_known_similarity_transform_from_three_points() -> None:
    true_scale = 1.25
    true_angle = math.radians(37.0)
    true_tx = 2.4
    true_ty = -1.3
    c = math.cos(true_angle)
    s = math.sin(true_angle)

    native = [(0.0, 0.0), (4.0, 0.0), (1.5, 3.0)]
    floor = [
        (true_scale * (c * x - s * y) + true_tx, true_scale * (s * x + c * y) + true_ty)
        for x, y in native
    ]

    tf = fit_similarity_transform(native, floor)
    assert tf.point_count == 3
    assert tf.rms_residual_m < 1e-12
    assert math.isclose(tf.scale, true_scale, rel_tol=1e-9)
    assert math.isclose(tf.rotation_rad, true_angle, abs_tol=1e-9)
    assert math.isclose(tf.tx_m, true_tx, abs_tol=1e-9)
    assert math.isclose(tf.ty_m, true_ty, abs_tol=1e-9)

    px, py = tf.transform_point(2.0, 1.0)
    expected_x = true_scale * (c * 2.0 - s * 1.0) + true_tx
    expected_y = true_scale * (s * 2.0 + c * 1.0) + true_ty
    assert math.isclose(px, expected_x, abs_tol=1e-9)
    assert math.isclose(py, expected_y, abs_tol=1e-9)

    xx, xy, yy = tf.transform_covariance(0.04, 0.01, 0.09)
    assert xx > 0.0 and yy > 0.0 and (xx * yy - xy * xy) > 0.0


def test_fit_reports_nonzero_residual_on_noisy_survey_points() -> None:
    native = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
    floor = [(0.0, 0.0), (4.1, 0.0), (0.0, 3.9)]
    tf = fit_similarity_transform(native, floor)
    assert tf.rms_residual_m > 0.01


def test_fit_rejects_fewer_than_two_or_coincident_points() -> None:
    with pytest.raises(ValueError, match="At least 2"):
        fit_similarity_transform([(0.0, 0.0)], [(1.0, 1.0)])
    with pytest.raises(ValueError, match="coincident"):
        fit_similarity_transform([(1.0, 1.0), (1.0, 1.0)], [(0.0, 0.0), (2.0, 2.0)])
