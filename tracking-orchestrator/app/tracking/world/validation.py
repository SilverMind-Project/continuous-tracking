"""Pure validators for floor coordinates and covariance, run before the
cost matrix is built.

These functions exist so that no ``NaN``, infinite, mis-shaped, non-symmetric,
non-positive-semidefinite, or pathologically large covariance ever reaches the
Mahalanobis gate, the Hungarian solver, the Kalman update, or trajectory
persistence (M03 completion criteria). They are numpy-only and import nothing
from the rest of the tracker so ``cost_matrix → validation`` stays acyclic.

A floor point or covariance that fails validation is *not* a match candidate:
the caller fails the pair closed to ``GATE_INF`` and records the reason.

The covariance trace cap (``DEFAULT_MAX_COV_TRACE_M2``) is the guard that keeps
*"large uncertainty cannot convert an implausible jump into a match"* from being
true while *"existing valid Mahalanobis boundary cases remain unchanged"* also
holds: it sits above any legitimate observation covariance (the largest green
fixture is ``4.0·I`` → trace 8.0 m²; real homography covariances are sub-m²) and
below the inflated magnitudes that would otherwise rescue an implausible jump.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NDArrayF8 = npt.NDArray[np.float64]

# Symmetric within this absolute tolerance on the off-diagonal difference (m²).
DEFAULT_COV_SYMMETRY_TOL_M2: float = 1e-6
# Eigenvalues may dip this far below zero before the matrix is rejected as
# non-PSD; absorbs floating-point noise without admitting indefinite matrices.
DEFAULT_COV_PSD_TOL_M2: float = -1e-9
# Maximum allowed position-covariance trace (sum of the two variances) in m².
# See module docstring for how this value is bounded by the fixtures.
DEFAULT_MAX_COV_TRACE_M2: float = 20.0


def is_finite_point(x_m: float, y_m: float) -> bool:
    """Return True when both floor coordinates are finite (no NaN/inf)."""
    return bool(np.isfinite(x_m) and np.isfinite(y_m))


def is_valid_covariance(
    cov_m2: NDArrayF8,
    *,
    symmetry_tol_m2: float = DEFAULT_COV_SYMMETRY_TOL_M2,
    psd_tol_m2: float = DEFAULT_COV_PSD_TOL_M2,
    max_trace_m2: float = DEFAULT_MAX_COV_TRACE_M2,
) -> bool:
    """Return True when *cov_m2* is a usable 2x2 floor covariance.

    Rejects (returns False) when the matrix is:

    - not finite (any NaN/inf entry),
    - not shape (2, 2),
    - not symmetric within ``symmetry_tol_m2``,
    - not positive semidefinite (smallest eigenvalue below ``psd_tol_m2``),
    - over the trace cap ``max_trace_m2`` (pathologically uncertain).

    Pure: no exceptions escape — an eigenvalue solver failure is treated as
    invalid (fail closed), never propagated.
    """
    arr = np.asarray(cov_m2, dtype=np.float64)
    if arr.shape != (2, 2):
        return False
    if not bool(np.all(np.isfinite(arr))):
        return False
    if abs(float(arr[0, 1] - arr[1, 0])) > symmetry_tol_m2:
        return False
    if float(arr[0, 0] + arr[1, 1]) > max_trace_m2:
        return False
    try:
        # eigvalsh on the symmetric part; cheap and robust for 2x2.
        eigenvalues = np.linalg.eigvalsh(0.5 * (arr + arr.T))
    except np.linalg.LinAlgError:
        return False
    return bool(float(eigenvalues.min()) >= psd_tol_m2)
