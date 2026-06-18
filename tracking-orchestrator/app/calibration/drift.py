"""Camera-drift detection via ORB feature matching.

Primary metric: ORB feature match + RANSAC homography inlier ratio.
  - Binary descriptors are intensity-invariant, so day/night lighting changes
    do not produce false positives.  This is the dominant failure mode for
    scalar SSIM-only detectors in care-home deployments.

Secondary metric: Gaussian-windowed mean SSIM (via cv2, not scikit-image).
  - Corroborates ORB findings only; never the sole driver of a drift flag.

Rotation/translation check: cv2.estimateAffinePartial2D on RANSAC inliers.
  - Avoids cv2.decomposeHomographyMat which requires camera intrinsics K.
  - A similarity transform (scale + rotate + translate) is sufficient to detect
    physical camera bumps without needing a calibrated focal length.

Insufficient-features guard: when ORB finds too few good matches the ratio
is unreliable noise.  Return drifted=False + reason="insufficient_features"
rather than risk a false positive.  Operator false positives are the primary
nuisance in care-home settings.

Human-in-the-loop: this module ONLY scores and flags.  It never reads or
writes any homography or calibration state.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Default thresholds – conservative to minimise false positives.
# ---------------------------------------------------------------------------

_DEFAULT_MIN_INLIER_RATIO: float = 0.35
_DEFAULT_MIN_ROTATION_DEG: float = 1.5
_DEFAULT_MIN_TRANSLATION_PX: float = 15.0

# Below this many Lowe-filtered matches the inlier ratio is unreliable noise.
_MIN_GOOD_MATCHES: int = 20


@dataclass(frozen=True)
class DriftResult:
    """Score returned by :func:`drift_score`."""

    inlier_ratio: float
    ssim: float
    drifted: bool
    reason: str


def _mean_ssim(
    gray_ref: npt.NDArray[np.uint8],
    gray_cur: npt.NDArray[np.uint8],
) -> float:
    """Gaussian-windowed mean SSIM computed with cv2.GaussianBlur.

    Implements Wang et al. (2004) without scikit-image.  Uses the same
    default constants (K1=0.01, K2=0.03, L=255) as the reference implementation.
    """
    r = gray_ref.astype(np.float32)
    c = gray_cur.astype(np.float32)

    k_size = (11, 11)
    sigma = 1.5
    C1 = (0.01 * 255.0) ** 2  # 6.5025
    C2 = (0.03 * 255.0) ** 2  # 58.5225

    mu_r = cv2.GaussianBlur(r, k_size, sigma)
    mu_c = cv2.GaussianBlur(c, k_size, sigma)
    mu_r2 = mu_r * mu_r
    mu_c2 = mu_c * mu_c
    mu_rc = mu_r * mu_c

    sigma_r2 = cv2.GaussianBlur(r * r, k_size, sigma) - mu_r2
    sigma_c2 = cv2.GaussianBlur(c * c, k_size, sigma) - mu_c2
    sigma_rc = cv2.GaussianBlur(r * c, k_size, sigma) - mu_rc

    numerator = (2.0 * mu_rc + C1) * (2.0 * sigma_rc + C2)
    denominator = (mu_r2 + mu_c2 + C1) * (sigma_r2 + sigma_c2 + C2)

    ssim_map: npt.NDArray[np.float32] = numerator / denominator
    return float(np.mean(ssim_map))


def drift_score(
    reference_bgr: npt.NDArray[np.uint8],
    current_bgr: npt.NDArray[np.uint8],
    *,
    min_inlier_ratio: float = _DEFAULT_MIN_INLIER_RATIO,
    min_rotation_deg: float = _DEFAULT_MIN_ROTATION_DEG,
    min_translation_px: float = _DEFAULT_MIN_TRANSLATION_PX,
) -> DriftResult:
    """Score a current frame against a calibration reference for camera drift.

    Both arrays must be BGR uint8 (standard cv2 convention).  Convert from
    RGB with ``arr[:, :, ::-1]`` before calling.

    Args:
        reference_bgr: Snapshot captured at the time of the last calibration.
        current_bgr: Recent frame from the same camera.
        min_inlier_ratio: Inlier ratio below which drift is declared.
        min_rotation_deg: Rotation angle above which drift is declared.
        min_translation_px: Translation magnitude above which drift is declared.

    Returns:
        DriftResult with drifted=False and reason="insufficient_features" when
        ORB finds too few matches to make a reliable determination.
    """
    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    # Resize current to reference dimensions before matching and SSIM.
    if current_bgr.shape[:2] != reference_bgr.shape[:2]:
        current_bgr = cv2.resize(
            current_bgr,
            (reference_bgr.shape[1], reference_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    cur_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)

    ssim = _mean_ssim(ref_gray, cur_gray)

    # ORB: binary descriptors tolerate brightness/contrast changes.
    orb = cv2.ORB_create(nfeatures=1000)
    kp_ref, desc_ref = orb.detectAndCompute(ref_gray, None)
    kp_cur, desc_cur = orb.detectAndCompute(cur_gray, None)

    if desc_ref is None or desc_cur is None or len(kp_ref) < 10 or len(kp_cur) < 10:
        return DriftResult(
            inlier_ratio=0.0,
            ssim=ssim,
            drifted=False,
            reason="insufficient_features",
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desc_ref, desc_cur, k=2)

    # Lowe's ratio test (Lowe 2004): discard ambiguous matches.
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]

    if len(good) < _MIN_GOOD_MATCHES:
        return DriftResult(
            inlier_ratio=0.0,
            ssim=ssim,
            drifted=False,
            reason="insufficient_features",
        )

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return DriftResult(
            inlier_ratio=0.0,
            ssim=ssim,
            drifted=False,
            reason="insufficient_features",
        )

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good)

    # Rotation/translation from the RANSAC inlier set via partial affine.
    # estimateAffinePartial2D needs no camera intrinsics (unlike decomposeHomographyMat).
    rotation_deg = 0.0
    translation_px = 0.0
    inlier_src = src_pts[mask.ravel() == 1]
    inlier_dst = dst_pts[mask.ravel() == 1]
    if len(inlier_src) >= 3:
        affine, _ = cv2.estimateAffinePartial2D(inlier_src, inlier_dst)
        if affine is not None:
            # affine[:,0] = [cos(θ)·s, sin(θ)·s]
            rotation_deg = float(abs(np.degrees(np.arctan2(float(affine[0, 1]), float(affine[0, 0])))))
            translation_px = float(np.hypot(float(affine[0, 2]), float(affine[1, 2])))

    if inlier_ratio < min_inlier_ratio:
        return DriftResult(
            inlier_ratio=inlier_ratio,
            ssim=ssim,
            drifted=True,
            reason=f"low_inlier_ratio:{inlier_ratio:.3f}",
        )

    if rotation_deg > min_rotation_deg:
        return DriftResult(
            inlier_ratio=inlier_ratio,
            ssim=ssim,
            drifted=True,
            reason=f"rotation:{rotation_deg:.2f}deg",
        )

    if translation_px > min_translation_px:
        return DriftResult(
            inlier_ratio=inlier_ratio,
            ssim=ssim,
            drifted=True,
            reason=f"translation:{translation_px:.1f}px",
        )

    return DriftResult(
        inlier_ratio=inlier_ratio,
        ssim=ssim,
        drifted=False,
        reason="ok",
    )
