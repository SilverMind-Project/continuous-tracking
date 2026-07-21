# Operations: Camera Drift Detection & Recalibration

## What is camera drift?

A fixed camera is considered "drifted" when its physical position has changed
(been bumped, accidentally rotated, or repositioned) since its last homography
calibration was committed.  Because the homography maps image pixels to
floor-plan coordinates, even a small positional change degrades all
localization for that camera — tracked positions will be offset, cross-camera
deduplication may fail, and dwell-time calculations become inaccurate.

Today there is no automatic signal that a camera has moved.  This feature adds
a periodic automated check that compares a recent scene frame against a
snapshot taken at calibration time and flags the camera for operator review
when significant geometric change is detected.

---

## How drift is detected

### Primary metric: ORB feature match + RANSAC inlier ratio

The system uses ORB (Oriented FAST and Rotated BRIEF) to extract binary
feature descriptors from both the stored reference frame and a recent live
frame.  It then matches them with a Brute-Force matcher (Hamming distance) and
applies Lowe's ratio test to discard ambiguous matches.  A RANSAC homography is
fitted to the surviving matches; the **inlier ratio** (inliers / total good
matches) is the primary drift metric.

- **Threshold**: `inlier_ratio < 0.35` → drift flagged.
- **Why ORB is primary**: ORB descriptors are binary and intensity-invariant.
  A camera that goes from daytime to nighttime lighting will still produce high
  inlier ratios because the descriptors do not depend on absolute brightness.
  SSIM alone would false-positive on every day/night transition.

### Secondary check: rotation and translation (via partial affine)

Once RANSAC inliers are identified, `cv2.estimateAffinePartial2D` fits a
similarity transform (rotation + scale + translation) to the inlier
point-pairs.  This avoids needing camera intrinsics (which `decomposeHomographyMat`
requires).

- Rotation > 1.5° → drift flagged.
- Translation > 15 px → drift flagged.

### Secondary metric: mean SSIM (corroboration only)

A Gaussian-windowed mean SSIM score is computed using `cv2.GaussianBlur`
(no scikit-image).  This is **never** the sole driver of a drift flag; it is
included in the response payload as a corroborating signal for operator review.
SSIM false-positives on lighting changes even when geometry is unchanged, which
is why ORB is authoritative.

### Insufficient-features guard

When ORB finds fewer than 20 good matches (e.g. a textureless scene,
a dark overnight frame, or a lens-cap-on camera), the result is reported as
`drifted=False` with `reason="insufficient_features"`.  This prevents spurious
alerts from degraded frames.

---

## Human-in-the-loop policy

**The system never automatically recalibrates.**

When drift is detected:
1. The `cts_cameras.needs_recalibration` flag is set to `True` in the database.
2. A structured warning is logged (`camera_drift_detected`).
3. A "Drift" badge appears on the camera in the admin UI (Cameras list) and a
   warning banner is shown in the Calibration view.

The operator then chooses whether to recalibrate.  The UI provides a
**"Re-run Auto-Calibration"** CTA that goes through the same operator-reviewed
auto-calibration flow — the operator must review and commit the
draft before it becomes active.  No silent homography mutations occur.

---

## Reference frame capture

When an operator commits a homography (manual calibration), the system:
1. Fetches a live snapshot from the RTSP ingress.
2. Stores it to MinIO with a stable key:
   `calibration-refs/{camera_id}/{calibrated_at_iso}.jpg`
3. Records the key in `cts_cameras.calibration_ref_key`.

This reference frame is the "what the scene looked like when calibrated"
baseline used by all subsequent drift checks.  Cameras without a
`calibration_ref_key` (legacy calibrations or cameras calibrated while the
ingress was unavailable) are skipped by the drift poll until a new calibration
is committed.

---

## Drift check frequency

The drift poll runs hourly by default (configurable via
`cts.drift_poll_interval_s`, default `3600`).  Drift is rare; a 1-minute poll
cadence is unnecessary and would waste orchestrator resources.

---

## How to recalibrate after drift

1. In the admin UI, navigate to **CTS → Cameras** and note the "Drift" badge
   on the affected camera.
2. Navigate to **CTS → Calibration**, select the affected camera.  A warning
   banner with a **"Re-run Auto-Calibration"** button will be shown.
3. Click the button.  The system will fetch a fresh snapshot, run the
   depth-based floor-plane fit, and present draft calibration points for review.
4. Verify the suggested points are correct, adjust if needed, then click
   **Commit Calibration**.
5. On commit:
   - The new homography takes effect immediately in the orchestrator.
   - `needs_recalibration` is reset to `False`.
   - A new reference frame is captured for future drift checks.

Alternatively, use manual calibration if the auto-calibration draft is
inaccurate (e.g. a heavily cluttered scene or unusual camera angle).

---

## Thresholds and tuning

| Parameter | Default | Description |
|---|---|---|
| `min_inlier_ratio` | 0.35 | ORB inlier ratio below which drift is declared |
| `min_rotation_deg` | 1.5° | Rotation angle above which drift is declared |
| `min_translation_px` | 15 px | Translation magnitude above which drift is declared |
| `drift_poll_interval_s` | 3600 | Seconds between drift checks per camera |

**Tuning guidance**: false positives annoy operators; false negatives leave
stale calibration active.  Err on the side of lower sensitivity (higher
thresholds) for most deployments.  If a camera is in a location where people
frequently brush it (e.g. a doorframe), raise `min_rotation_deg` to reduce
noise.  The ORB inlier ratio threshold is the most reliable knob; the
rotation/translation thresholds provide a secondary safety net for
near-threshold inlier ratios.

---

## Implementation reference

| Component | Location |
|---|---|
| Drift scoring (pure) | `tracking-orchestrator/app/calibration/drift.py` |
| CTS drift endpoint | `POST /internal/calibration/drift/{camera_id}` |
| CC drift poller | `cognitive-companion/backend/services/cts/drift_poll.py` |
| CC model columns | `cts_cameras.needs_recalibration`, `.drift_checked_at`, `.drift_reason`, `.calibration_ref_key` |
| CC migration | `alembic/versions/0012_drift_detection.py` |
| Frontend badge | `CTSCamerasView.vue` (camera list), `CTSCalibrationView.vue` (banner + CTA) |
