# Spike S5a: Online Self-Calibration from Corresponded Tracks

**Status**: Evaluation memo only — not a committed feature.

## Objective

Evaluate whether accumulating corresponded foot-point trajectories from
identity-linked PHs across two cameras can be used to estimate a relative
homography via `cv2.findHomography` + RANSAC, and whether using that estimated
homography improves cross-camera association versus appearance-only matching.

## Background

Uncalibrated cameras receive synthetic floor points in disjoint 200 m virtual
tiles, making geometric cross-camera association impossible. However, once the
cross-camera revival (CTS_ROBUSTNESS_M5.2) links a person's PH across two
uncalibrated cameras, their foot-point pixel trajectories form putative
correspondences. Accumulating enough correspondences could enable automatic
homography estimation, promoting cameras from uncalibrated to calibrated
opportunistically.

## Acceptance Criteria

1. On a labelled two-camera replay, the estimated homography back-projects
   held-out correspondences to within a stated pixel or metric error.
2. Using the estimated homography improves cross-camera association versus
   appearance-only by a stated margin (lower association failure rate or
   higher identity-continuity rate).
3. The method uses only approved libraries (`cv2` from `opencv-python-headless`,
   `numpy`).

If the criteria are not met within the timebox, the recommendation is "do not
adopt now" and the zero-calibration appearance + topology design stands.

## Proposed Approach

1. Accumulate correspondences: when cross-camera revival or continuation links
   a PH from camera A to camera B, record the bbox centre pixel coordinate on
   each camera as a putative correspondence pair.
2. Filter correspondences: require face-anchored PHs (identity-confirmed) and
   a minimum quality threshold to reject calibration noise.
3. Once `N >= N_min` correspondences (suggest `N_min = 20`) accumulate:
   - Run `cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)`.
   - Validate: compute mean reprojection error on held-out pairs.
4. Store the estimated homography in `calibration_state` as an
   `estimated`-quality entry (distinct from operator-calibrated).
5. When available, use it in `SpatialProjectionStage` to produce calibrated
   floor points for that camera pair, enabling geometric dedup and association.

## Key Risks

- **Degenerate correspondences**: If the person walks along a line, the
  correspondences are colinear and `findHomography` produces a degenerate
  matrix. Mitigation: require correspondences to span at least 3 non-colinear
  positions, detectable via singular-value analysis of the point set.
- **Drift over time**: Camera mounts may shift. The estimated homography
  should expire and be recomputed periodically.
- **Accumulation latency**: It may take hours or days to accumulate enough
  correspondences for a rarely-used camera pair. The appearance + topology
  path must work without it.

## Timebox

Fixed small budget. If acceptance criteria are not met within it, output is a
short memo with a binary "do not adopt now" recommendation.

## Recommendation (placeholder)

To be filled after evaluation.
