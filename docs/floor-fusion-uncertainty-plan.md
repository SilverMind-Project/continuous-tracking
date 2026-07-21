# Floor-Plane Fusion, Per-Observation Uncertainty, and Visibility-Polygon Plan

Status: design agreed (interview-driven), 2026-06-16. Implementation not started.

This plan covers two related but separable workstreams:

1. **Multi-camera floor-position + posture fusion** (the live-map "one dot vibrating"
   problem and the "4 cameras disagree on posture" problem).
2. **Visibility-polygon / floor-region correction** (Track G) and **camera-drift
   detection**.

---

## 1. Corrected diagnosis (what is and isn't broken)

The original concern was that walls contaminate the homography projection. Tracing the
code shows the wall problem is real in **exactly one place**, and the runtime is clean:

| Mechanism | Location | Wall-contaminated? |
|---|---|---|
| Homography fit | `app/calibration/homography.py::compute_homography` | No — fits H from only the picked point correspondences; unpicked pixels (walls) never enter. |
| Person localization | `app/tracking/floor_projector.py::project` | No — projects the footpoint (feet on the floor plane). |
| Room/zone containment | `app/tracking/world/helpers.py::is_in_any_room_polygon` | No — floor-meter point-in-polygon; the camera image never enters. |
| Visibility/coverage polygon | CC `backend/services/cts_visibility.py` | **Yes** — projects the 4 image edges (mostly walls) through H. Used only by CC adjacency inference + UI; zero CTS runtime usage. |

So the **jitter and posture disagreement are NOT a wall/homography bug.** They are a
**multi-camera sensor-fusion gap**: the system never models per-observation measurement
uncertainty from view geometry.

### Root cause of the jitter (confirmed symptom: "one dot vibrating in place")

Two independent levers, both unaddressed today:

- **Measurement model.** Floor-position fusion is split across two places that both
  ignore view geometry:
  - `dedup.py::_build_representative` merges cross-camera observations with a
    **crop-quality-weighted mean** (crop sharpness, not geometric trust).
  - `tracker.py:399` Kalman `update(..., cfg.observation_noise_m)` uses a **single fixed
    isotropic** `observation_noise_m = 0.25 m` for every camera/angle/distance.
  `WorldObservation.floor_residual_m` exists but is used only to widen the dedup gate.
- **Process model.** The constant-velocity Kalman manufactures phantom velocity from
  measurement noise; `velocity_decay_s = 3 s` keeps it alive, so a stationary person's
  estimate drifts then corrects → visible tremble. Per-observation R does not fix this.

CC's live map plots the Kalman mean carried on the `TrackingEvent` snapshot
(`floor_x_m = ph.state_mean[0]`), so the tremble is in the tracker state, not rendering.

### Posture is the same gap in a second place

`GlobalPostureTracker._fuse` (`app/trajectory/posture.py`) already has the mature
*structure* position lacks (per-camera snapshots, 10 s staleness expiry, weighted fuse,
hysteresis) — but it weights only by `keypoint_confidence`, **not** view geometry. A
camera viewing a person head-on/overhead can have a confident skeleton yet a
foreshortened, wrong posture and drag the fused result.

**Unifying insight:** one per-observation view-geometry descriptor feeds position R,
posture weight, and primary-camera selection. One concept, three thin consumers — this
is the "reduce branching" backbone.

---

## 2. Agreed design decisions

| # | Decision | Choice |
|---|---|---|
| D1 | Fusion ambition | **Full anisotropic** 2×2 covariance R per observation. |
| D2 | Fusion structure | **Contained**: dedup still emits ONE representative/frame carrying a fused covariance; `kalman.update` takes a 2×2 R. **Rejected** per-camera sequential Kalman updates (no statistical gain; over-counts correlated cameras; more branching). |
| D3 | R inputs | Homography **Jacobian** (mandatory, gives anisotropy) + **footpoint occlusion/truncation** + **calibration-residual** inflation + **detection/crop quality**. No explicit depth term (Jacobian already inflates far-field). |
| D4 | Process lever | **Adaptive zero-velocity update (ZUPT)**: when innovation magnitude AND speed stay below thresholds for K frames, drive velocity toward zero. Rejected IMM (too heavy) and Q-only retune (blunt). |
| D5 | Occluded feet | **Inflate R steeply, keep the obs** (other cameras + ZUPT + prediction carry it). Head+height reconstruction documented as a future option for single-camera areas; not in scope. |
| D6 | Posture fusion | **Geometry-aware weighting**: `keypoint_confidence × posture-view-suitability` (penalize foreshortened head-on/overhead/occluded views; use the `orientation` bin since side views best separate sit/stand/lie) + a margin in `_resolve` to stop near-tie flips. Keep staleness + hysteresis. |
| D7 | Primary camera | **Stabilized best-view primary** (hysteresis) for snapshot `camera_id`, keyframe association, and room fallback. **Position stays fully inverse-covariance fused** (not single-camera). |
| D8 | Persistence | Add `position_sigma_m`, `primary_camera_id`, `contributing_camera_count`, `footpoint_reliable` to `person_trajectories`. Pre-release → fold into `0001_init` baseline (no migration chain). Enables post-hoc bug tracing and future signal confidence-gating. |
| D9 | Visibility polygon | Derive wall-excluded floor region from the **depth-inlier mask** CTS already computes (Depth Anything v2 + RANSAC). **No SAM.** Feed CC `compute_visibility_from_homography`. (Track G.) |
| D10 | Drift | **Detect + flag for operator review** (store reference keyframe at calibration; periodic SSIM/feature-match; below threshold → "needs recalibration" alert). No silent auto-recalibration. |
| D11 | Architecture | **One shared `ObservationGeometry` descriptor + pure-function module** (`tracking/world/observation_model.py`); `FloorProjector` returns `(point, covariance)`; three thin consumers (position R, posture weight, primary-cam score). |
| D12 | Rollout | **Single production path** (no permanent fixed-R fallback flag). Validate via replay fixtures + quantified CI acceptance gates. Test harness may run old-vs-new on the same fixture to prove the gain. |
| D13 | Docs | **Ordered progressive series** under `docs/features/continuous-tracking/` (silvermind-project.github.io), beginner→advanced; each milestone ships its matching page. |

### The R model (concrete)

```
R_obs = J · Σ_px · Jᵀ + R_cal

J      = ∂(floor_xy)/∂(pixel_xy) at the footpoint, from the homography H (anisotropy).
Σ_px   = footpoint pixel covariance, driven by:
           - base detector localization noise,
           - × inflation when footpoint unreliable (pose ankle keypoints not visible
             OR bbox clipped at image edge)  [D5: steeply inflated],
           - × scaling from detection confidence / crop quality.
R_cal  = isotropic term from floor_residual_m (per-camera systematic bias).
```

Cross-camera fusion in dedup becomes **information-form** — but only the *random* part
may shrink with more cameras. **Critical:** information-form fusion (`Σ Rᵢ⁻¹`) is
mathematically identical to N sequential Kalman updates, so it shrinks R* ~1/N. That is
correct for the **random** part (`J·Σ_px·Jᵀ`, independent detector noise) but **wrong for
the systematic part** (`R_cal` — each camera has a *fixed* calibration offset that does
not average toward truth). If `R_cal` is fused away, the filter becomes overconfident
(lags real motion) and the fused point jumps by the inter-camera bias difference whenever
the visible camera set changes — reproducing the very symptom we are fixing.

Therefore fuse only the random term, then add a non-shrinking bias floor:

```
R* = ( Σ (J · Σ_px · Jᵀ)ᵢ⁻¹ )⁻¹  +  R_bias_floor
```

where `R_bias_floor` is derived from the cluster's calibration residuals (e.g. the
max/representative per-camera `R_cal`, NOT the inverse-sum). The representative carries
`(x*, R*)`; the Kalman update consumes the matrix `R*`. The association/dedup grouping
gates should use the same uncertainty (Mahalanobis with R) where they currently use
residual-widened Euclidean distance — fold in, single path.

### Stabilized primary camera

Floor position is still the fused Kalman state, not a single-camera projection. The tracker
also maintains one per-PH primary camera for the downstream snapshot `camera_id`, keyframe
selection, and camera-based room fallback when no room polygon contains the fused point.

The primary is selected from the cameras that contributed to the PH update in the current
tracker frame. Each observation carries `primary_score`, computed only by
`observation_model.primary_camera_score(ObservationGeometry)` from footpoint reliability,
crop quality, and detection confidence. The tracker chooses the best-scoring camera for the
frame, then applies hysteresis: a different camera must be best for
`WorldTrackerConfig.primary_switch_frames` consecutive tracker frames before the PH primary
switches. This keeps displayed crops and room labels from flickering across overlapping
cameras while leaving fused position, covariance, and lifecycle logic unchanged.

---

## 3. Milestones (with dependencies)

Fusion track is independent of the visibility/drift track; they can
proceed in parallel. Each milestone ships its docs page.

- **M1 — Observation model foundation.** `ObservationGeometry` frozen descriptor +
  `observation_model.py` pure functions: footpoint Jacobian, Σ_px, `R_obs`,
  posture-view weight, primary-camera score. Unit tests (no Triton/DB). → docs pages 1, 3.
- **M2 — Wire it to observations.** `FloorProjector.project` returns `(FloorPoint, cov)`;
  `WorldObservation` gains `floor_cov` + footpoint-reliability fields; the
  detection/pose pipeline stage assembles the descriptor (needs ankle-keypoint
  visibility + bbox-truncation). Depends M1.
- **M3 — Matrix-R Kalman + information-form dedup.** `kalman.update` / gating take a 2×2
  R; `dedup` fuses via inverse-covariance mean carrying `R*`. **Must implement the
  random-vs-systematic split: fuse only `J·Σ_px·Jᵀ`, add the non-shrinking
  `R_bias_floor` after fusion** (see §2) — otherwise the dot gets worse (overconfident,
  jumps on camera-set changes), not better. Depends M1, M2. → docs 4.
- **M4 — ZUPT stationarity.** Adaptive zero-velocity treatment. Depends M3. → docs 4.
- **M5 — Geometry-aware posture fusion.** Suitability weight + orientation + resolve
  margin in `GlobalPostureTracker`. Depends M1. → docs 5.
- **M6 — Stabilized primary-camera selection.** Hysteresis-smoothed primary for snapshot
  `camera_id` / keyframe / room fallback. Depends M1.
- **M7 — Persistence.** `person_trajectories` confidence fields folded into `0001_init`;
  repo + `TrajectoryWriter` + snapshot plumbing. Depends M2, M3, M6.
- **M8 — Acceptance fixtures + CI gates.** Replay fixtures: stationary-person position
  variance ↓ (target threshold TBD from baseline), moving-person lag no-regression,
  posture cross-camera agreement ↑, and a **slow-shuffle-gait fixture (~0.2–0.4 m/s) with
  a "not clamped by ZUPT" assertion** — dementia patients shuffle slowly and that motion
  feeds pacing/wandering/stillness signals; ZUPT thresholds (D4) must be tuned against
  this dangerous middle case. Old-vs-new comparison harness. Depends M3–M6.
- **M9 — Floor-region polygon + visibility (Track G).** CTS: `floor_region_polygon(...)`
  from depth inliers; return it from `/internal/calibration/auto`. CC: store on
  `cts_cameras`, project the floor-region boundary in `compute_visibility_from_homography`,
  operator confirm/edit. Independent. → docs 6.
- **Drift detection.** Reference keyframe at calibration; periodic SSIM/feature
  comparison; operator "needs recalibration" flag + alert. Independent. → docs 7.
- **Docs series.** Tie pages 1–7 together (+ overview page 2 "why one dot
  jitters"); link from `person-tracking.md`. Cross-cutting.

### Recommended sequencing

Front-load the jitter fix (M1 → M2 → M3 → M4), then M5/M6 in parallel, then M7/M8.
Run visibility polygon (a quick, isolated win) in parallel from the start. Drift detection after
visibility polygon. Docs ride each milestone; finalizes.

### Open verification items (resolve during implementation, not blockers)

- Confirm whether any protobuf/wire field is needed. Expectation: **no** — `camera_id`
  already exists on the event (we only change its meaning to "primary"), `position_sigma_m`
  already on the snapshot; persistence fields are CTS-internal. Verify before assuming.
- Set concrete acceptance thresholds for M8 from a measured baseline run.
- Define R for **uncalibrated / synthetic floor points** (the `_synthetic_floor_point`
  path has no homography, so no Jacobian). They already bypass the geometric dedup pass
  (gated on `floor_point.calibrated`), but the matrix-R Kalman update still runs and needs
  some R — use a large fixed default so they contribute little. Specify in M3.

---

## 4. Literature basis (indoor camera ↔ floor plane)

- Ground-plane homography from ≥4 floor-point correspondences is the standard
  pixel↔world map; walls only matter when you project non-floor pixels (the
  visibility-polygon bug), not for the fit or footpoint localization.
- Multi-sensor fusion best practice: weight each measurement by its covariance
  (inverse-variance / information filter); do not pre-average then trust equally.
  Propagate pixel covariance through the projection Jacobian for an anisotropic R —
  oblique/far cameras have error elongated along the view ray. (This is exactly D3.)
- Floor/wall separation for the coverage polygon: depth-plane fitting (already in CTS)
  or 2D semantic segmentation (SAM/FloorSAM) back-projected. We reuse depth inliers (D9).
- Camera-drift detection: mean-SSIM (or feature-match) against a reference frame is the
  established CCTV/PTZ-drift method; flag above a rotation/translation threshold (D10).

Key sources:
- NVIDIA, "Simplifying Camera Calibration to Enhance AI-powered Multi-Camera Tracking" (2024).
- "Homography-based ground plane detection using a single on-board camera."
- FloorSAM: SAM-guided floorplan reconstruction (2025) — considered and rejected for CTS (D9).
- "Methodology for Automatically Detecting PTZ CCTV Camera Drift" (MDPI, 2024) — SSIM drift.
```
