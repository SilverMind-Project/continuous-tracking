# The Household Floor Frame

One metric frame is shared by cameras, rooms, transit zones, and (later) radio anchors and
wearable fixes. Every metre in CTS is a metre in this frame or it is not a metre at all.

## Definition

- **Origin:** the top-left corner of the household floor-plan image held by CC
  (`household_settings`, served at `GET /api/v1/household/floor-plan`).
- **Axes:** x to the right, y down, as in the image. The frame is left-handed when viewed from
  above; nothing in CTS depends on handedness, but code that computes a heading must use
  `atan2(dy, dx)` in these axes and never assume y up.
- **Units:** metres. A CC normalized coordinate `(u, v)` in `[0, 1]` maps to
  `x_m = u * floor_plan_width * floor_meters_per_pixel` and
  `y_m = v * floor_plan_height * floor_meters_per_pixel`. CTS `FloorPoint` stores the same
  value in millimetres (`x_mm`, `y_mm`); the Kalman state is in metres.
- **Height:** z is metres above the floor of the one storey the floor plan describes. A second
  storey is a separate floor plan and out of scope for the current deployment.
- **Scale check:** the scale (`floor_meters_per_pixel`) is valid only after two taped wall
  lengths, at right angles and each at least 3 m, agree with the plan within 2%. Without a
  scale, `cc_config_sync` clears room polygons and transit zones and logs
  `cc_config_sync_room_polygons_disabled`.

## Two tiers

| Tier | Holds | Enables |
| --- | --- | --- |
| 1 | Floor plan image, validated scale, a polygon for every room | Anchor placement, room-level wearable evidence, room attribution by polygon, transit zones |
| 2 | A homography per camera within the residual budget, overlap groups, a passed acceptance walk | Cross-camera dedup by floor distance, speed and gait signals, height cue, trajectory association |

Residual budget: `homography_residual_m` at or below 0.15 m for a camera whose floor view is
within 5 m, at or below 0.30 m otherwise. A camera that cannot meet it stays uncalibrated.

## Rules

- **A metre is only a metre on a calibrated camera.** An uncalibrated camera's detections get a
  synthetic floor point in a 4 m virtual room on a per-camera 200 m tile
  (`_synthetic_floor_point` in `app/pipeline/stages/world_tracking.py`). Code that compares
  floor positions, computes a speed, or tests a room polygon must check
  `floor_point.calibrated` (or the PH's calibrated provenance) first.
- **CC is authoritative.** `CCConfigSyncService` applies CC's homographies every poll and removes
  a runtime homography CC no longer holds (cleared, rejected by validation, camera disabled or
  deleted); the camera returns to its virtual room on the next frame. Overlap groups are applied
  on the same poll; a failed fetch keeps the last applied groups.
- **Overlap groups are the only cross-camera link for uncalibrated cameras.** Group appearance
  dedup merges uncalibrated observations from cameras of one group by appearance; without a
  group, two uncalibrated cameras never merge.

## Observing it

| Metric | Meaning |
| --- | --- |
| `cts_worldtracker_observations_by_calibration_total{camera_id, calibrated}` | Observations per camera by floor-point source |
| `cts_worldtracker_dedup_pair_distance_m` | Distance of every calibrated cross-camera pair merged by the floor-distance pass: the agreement measurement for the acceptance walk (median at or below 0.5 m, p90 at or below 1.0 m). Pairs beyond the gate are splits and do not appear |
| `cts_worldtracker_group_appearance_dedup_total` | Uncalibrated pairs merged by appearance inside an overlap group |
| `cts_overlap_groups_loaded` | Overlap groups currently applied |
| `cts_transit_zones_loaded` | Transit zones loaded (zero without a floor-plan scale) |
