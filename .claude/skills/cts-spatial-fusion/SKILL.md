---
name: cts-spatial-fusion
description: Use when changing CTS floor-plane localization, the homography Jacobian, per-observation measurement covariance, cross-camera fusion (dedup), the floor-plane Kalman filter, ZUPT, primary-camera selection, or geometry-aware posture weighting.
paths: ["**/tracking-orchestrator/app/tracking/**"]
---

# CTS Spatial Fusion

How CTS turns per-camera detections into one stable floor position and one posture per person.
All geometry/uncertainty math lives in ONE module: `app/tracking/world/observation_model.py`.
Consumers (FloorProjector, dedup, GlobalPostureTracker, WorldTracker) call it; they never
re-derive geometry. This single-owner rule is load-bearing: position and posture fusion
diverged historically because the geometry was computed in two places.

## Coordinate frames and units (non-negotiable)

The shared floor frame (origin, axes, scale check, the two calibration tiers) is defined in
`reference/floor-frame.md`. A metre is only a metre on a calibrated camera: code that compares
floor positions must check `floor_point.calibrated`.

| Quantity | Unit | Frame |
|---|---|---|
| `FloorPoint.x_mm/y_mm` | mm | shared floor plan |
| Kalman state mean `[x,y,vx,vy]` | m, m/s | shared floor plan |
| Measurement covariance `R` | **m²** | shared floor plan |
| Pixel covariance `Σ_px` | **px²** | image |
| Homography Jacobian `J` | **m/px** | image→floor |
| Calibration residual `floor_residual_m` | m | floor plan |

`R = J · Σ_px · Jᵀ` is m² because (m/px)·px²·(m/px) = m². The Kalman state is in metres, so all
covariance entering it is m². Never feed mm² into the Kalman.

## The measurement-uncertainty model

Each calibrated observation gets an anisotropic 2×2 covariance:

```
R_obs = J · Σ_px · Jᵀ + R_cal
J      = analytic Jacobian of the homography projection at the footpoint (m/px).
Σ_px   = base_px² · I, inflated when the footpoint is unreliable and scaled by
         detection confidence / crop quality.
R_cal  = (k_cal · floor_residual_m)² · I   (per-camera systematic-bias term).
```

The Jacobian makes the error ellipse stretch **along the viewing ray** for oblique/far cameras:
this is the whole point of the anisotropic model; a scalar σ throws it away.

**Footpoint reliability.** The footpoint (bbox bottom-centre) is only valid when the feet
contact the floor in view. Inflate `Σ_px` steeply when: ankle keypoints are not visible in the
pose result, OR the bbox touches the image edge (truncation). Do not silently drop; inflate.

## Cross-camera fusion: random vs systematic split (CRITICAL)

Information-form fusion `R*⁻¹ = Σ Rᵢ⁻¹` is identical to N sequential Kalman updates: it shrinks
covariance ~1/N. That is correct ONLY for the **random** part (`J·Σ_px·Jᵀ`, independent detector
noise). It is WRONG for the **systematic** part (`R_cal`: each camera has a fixed calibration
offset that does not average toward truth). If you fuse `R_cal` away, the filter goes
overconfident and the fused point JUMPS by the inter-camera bias whenever the visible camera set
changes, reproducing the bug.

Rule: fuse only the random term, then add a non-shrinking bias floor:

```
x*   = R* · Σ ( (J Σ_px Jᵀ)ᵢ⁻¹ · xᵢ )
R*   = ( Σ (J Σ_px Jᵀ)ᵢ⁻¹ )⁻¹  +  R_bias_floor
```

`R_bias_floor` is derived from the cluster's residuals (e.g. the representative/max per-camera
`R_cal`), NOT the inverse-sum. This belongs in `dedup.py` and is unit-tested with N cameras.

## Kalman filter rules

- The filter is constant-velocity on the floor plane (`kalman.py`). `update` and
  `mahalanobis2_position` take a **2×2 R matrix**, never a scalar.
- Backward-compatible default: `R = observation_noise_m² · I` reproduces the old scalar behavior;
  use it for uncalibrated/synthetic floor points (no homography → no Jacobian).
- **ZUPT (zero-velocity update).** A stationary person must not drift. When innovation magnitude
  AND estimated speed stay below thresholds for K consecutive frames, apply a zero-velocity
  pseudo-measurement to the velocity sub-state. Tune K and thresholds so a slow shuffle
  (0.2–0.4 m/s, clinically relevant for dementia gait) is NOT clamped.
- **Time model and coasting state.** Stored `PersonHypothesis` state (`state_mean`, `state_cov`)
  always represents the posterior at `last_seen_at` (capture time); in-round predictions to
  `now` (`max(ctx.capture_time)`) are transient for association and `coasting=True` snapshots,
  and matched updates predict from `last_seen_at` to `obs.captured_at` (see
  `reference/time-model.md`).
- **Wearable-to-PH association innovation covariance.** Trajectory-class wearable fixes
  (`sqrt(lambda_max(Sigma_f)) <= wearable.trajectory_max_sigma_m`, default `0.75 m`) score
  against calibrated PH floor history (`tracking-orchestrator/app/tracking/world/ph_floor_history.py`,
  appended only on calibrated rounds, never synthetic virtual tiles or coasting predictions)
  interpolated at `fix.measured_at` with innovation covariance
  `S = Sigma_f + Sigma_j + (v_j * sigma_t)^2 * I + sigma_bias^2 * I` (`sigma_t = 0.5 s`,
  `sigma_bias = 0.15 m`, chi-square gate `9.21`). Wide-covariance fixes reduce their 2D
  Gaussian over room polygons and use sole-occupant room rules.

## Primary-camera selection

Position is always inverse-covariance fused across cameras, never single-camera. But each PH
has ONE *primary camera* (for the displayed crop/keyframe and room fallback), chosen by the
best view-quality (footpoint reliability + crop/face quality) and **stabilized with hysteresis**
(switch only when another camera is clearly better for N frames). Lives in the tracker; uses the
shared descriptor.

## Geometry-aware posture weighting

`GlobalPostureTracker._fuse` weights cameras by `keypoint_confidence × posture_view_suitability`,
where suitability comes from the shared descriptor (penalize foreshortened head-on/overhead views
and occluded lower bodies) and the `orientation` bin (side views best separate sit/stand/lie).
`_resolve` requires a margin between the top-two fused scores before flipping class.

## Single-view standing height metrology

All height geometry lives in one pure module: `app/tracking/camera_pose.py`, called by
`HeightEstimationStage` (`app/pipeline/stages/height_estimation.py`) right after `InferenceStage`.

- **Pose recovery (`pose_from_homography`).** Recovers camera rotation `R` and centre `C` in the
  shared floor frame from the floor-to-image homography and `focal_length_px(width_px, horizontal_fov_deg)`
  (`cts_contracts.optics`). Handles both right-handed and left-handed (top-left-origin floor-plan)
  frames by picking the camera-above-floor sign (`C_z > 0`) without flipping `det(R)` during polar
  orthonormalization.
- **Startup and sync validation (`validate_camera_pose`).** Rejects cameras with missing profile
  fields (`horizontal_fov_deg`, `mounting_height_m`, `tilt_deg`), `|C_z - mounting_height_m| > 0.25 m`,
  or `|tilt_recovered - tilt_deg| > 10 deg`. `CameraPoseCache` caches per
  `(cal_version, profile_version, width_px, height_px)` and increments
  `cts_camera_height_pose_unreliable_total{reason}` once per epoch.
- **Metrology, uncertainty, and upright gate.** `is_upright_standing` requires a non-truncated box
  inside the central image band, aspect ratio `>= 1.3`, and visible shoulders above hips above
  ankles with torso tilt `<= 25 deg`. `standing_height_m` intersects the head ray with the vertical
  line through the calibrated foot point; `standing_height_with_sigma` propagates the 2x2 floor
  footpoint covariance, `sigma_head_px = 4.0`, and `sigma_model_m = 0.04 m` via central differences.
- **PH running state (`update_ph_height_estimate`).** World tracking fuses accepted measurements by
  inverse variance with a 3-sigma innovation gate (rejecting sitting outliers) and floors
  `PersonHypothesis.height_sigma_m` at `0.02 m`. Accepted observations also append to
  `ph_height_samples` (14-day TimescaleDB retention) for `HeightProfileLearner`.

## Where things live

| Concern | Module |
|---|---|
| Jacobian, Σ_px, R_obs, posture-view weight, primary-cam score | `app/tracking/world/observation_model.py` (pure) |
| Camera pose recovery, standing height, uncertainty, PH fusion | `app/tracking/camera_pose.py` (pure) |
| Single-view height stage and pose validation cache | `app/pipeline/stages/height_estimation.py` |
| `ObservationGeometry` descriptor (frozen) | `app/domain/__init__.py` |
| Footpoint→floor projection (+ covariance via the module) | `app/tracking/floor_projector.py` |
| Information-form cross-camera fusion | `app/tracking/world/dedup.py` |
| Matrix-R update, ZUPT | `app/tracking/world/kalman.py` |
| Primary-camera selection | `app/tracking/world/tracker.py` |
| Time model, coasting state, damped prediction, time-skew dedup | `reference/time-model.md` |
| Floor frame definition, calibration tiers, CC sync authority | `reference/floor-frame.md` |
| Posture weighting | `app/trajectory/posture.py` |

## Tests

All math is pure → table-driven unit tests, no Triton/DB. Mandatory fixtures: stationary-person
(variance must fall), slow-shuffle (must not be ZUPT-clamped), oblique-camera (R elongated along
the ray), N-camera fusion (R* does not shrink below the bias floor).
