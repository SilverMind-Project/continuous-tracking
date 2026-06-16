---
name: cts-spatial-fusion
description: Use when changing CTS floor-plane localization, the homography Jacobian, per-observation measurement covariance, cross-camera fusion (dedup), the floor-plane Kalman filter, ZUPT, primary-camera selection, or geometry-aware posture weighting.
---

# CTS Spatial Fusion

How CTS turns per-camera detections into one stable floor position and one posture per person.
All geometry/uncertainty math lives in ONE module: `app/tracking/world/observation_model.py`.
Consumers (FloorProjector, dedup, GlobalPostureTracker, WorldTracker) call it; they never
re-derive geometry. This single-owner rule is load-bearing: position and posture fusion
diverged historically because the geometry was computed in two places.

## Coordinate frames and units (non-negotiable)

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
R_cal  = (k_cal · floor_residual_m)² · I  — per-camera systematic-bias term.
```

The Jacobian makes the error ellipse stretch **along the viewing ray** for oblique/far cameras —
this is the whole point of the anisotropic model; a scalar σ throws it away.

**Footpoint reliability.** The footpoint (bbox bottom-centre) is only valid when the feet
contact the floor in view. Inflate `Σ_px` steeply when: ankle keypoints are not visible in the
pose result, OR the bbox touches the image edge (truncation). Do not silently drop — inflate.

## Cross-camera fusion: random vs systematic split (CRITICAL)

Information-form fusion `R*⁻¹ = Σ Rᵢ⁻¹` is identical to N sequential Kalman updates: it shrinks
covariance ~1/N. That is correct ONLY for the **random** part (`J·Σ_px·Jᵀ`, independent detector
noise). It is WRONG for the **systematic** part (`R_cal` — each camera has a fixed calibration
offset that does not average toward truth). If you fuse `R_cal` away, the filter goes
overconfident and the fused point JUMPS by the inter-camera bias whenever the visible camera set
changes — reproducing the bug.

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

## Primary-camera selection

Position is always inverse-covariance fused across cameras — never single-camera. But each PH
has ONE *primary camera* (for the displayed crop/keyframe and room fallback), chosen by the
best view-quality (footpoint reliability + crop/face quality) and **stabilized with hysteresis**
(switch only when another camera is clearly better for N frames). Lives in the tracker; uses the
shared descriptor.

## Geometry-aware posture weighting

`GlobalPostureTracker._fuse` weights cameras by `keypoint_confidence × posture_view_suitability`,
where suitability comes from the shared descriptor (penalize foreshortened head-on/overhead views
and occluded lower bodies) and the `orientation` bin (side views best separate sit/stand/lie).
`_resolve` requires a margin between the top-two fused scores before flipping class.

## Where things live

| Concern | Module |
|---|---|
| Jacobian, Σ_px, R_obs, posture-view weight, primary-cam score | `app/tracking/world/observation_model.py` (pure) |
| `ObservationGeometry` descriptor (frozen) | `app/domain/__init__.py` |
| Footpoint→floor projection (+ covariance via the module) | `app/tracking/floor_projector.py` |
| Information-form cross-camera fusion | `app/tracking/world/dedup.py` |
| Matrix-R update, ZUPT | `app/tracking/world/kalman.py` |
| Primary-camera selection | `app/tracking/world/tracker.py` |
| Posture weighting | `app/trajectory/posture.py` |

## Tests

All math is pure → table-driven unit tests, no Triton/DB. Mandatory fixtures: stationary-person
(variance must fall), slow-shuffle (must not be ZUPT-clamped), oblique-camera (R elongated along
the ray), N-camera fusion (R* does not shrink below the bias floor).
