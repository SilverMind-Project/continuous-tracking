## Floor Region & Visibility Polygon

### Problem

The visibility polygon for a camera is derived by projecting boundary points through the
homography matrix (pixel → floor-plan metres). Without a floor-region mask, the projection
samples the full image border — including wall pixels above the floor line. Walls are not on
the floor plane, so the homography produces geometrically incorrect floor-plan coordinates for
them. This contaminates the visibility polygon with large "shadow" regions that do not
correspond to walkable floor.

### Solution: floor-region polygon

The tracking orchestrator's depth auto-calibration (`FloorPlaneFitter`) already segments the
floor using Depth Anything v2 + RANSAC. The `floor_region_polygon()` function in
`app/calibration/floor_plane.py` converts that segmentation into a polygon:

1. Extract RANSAC inlier pixel coordinates (`FloorPlaneResult.sample_indices[inlier_mask]`).
2. Keep only points in the lower `region_fraction` (default 0.9) of the image height — this
   filters out near-ceiling inliers from very wide-angle or high-mounted cameras.
3. Build a concave hull via `shapely.concave_hull(ratio=0.3)`, which correctly handles
   L-shaped floors; falls back to `cv2.convexHull` on degenerate input.
4. Simplify with `cv2.approxPolyDP` (ε = 3 px) to reduce vertex count.
5. Normalise: output is `[[x_norm, y_norm], ...]` in **image space [0, 1]**.

The polygon is returned in the `/internal/calibration/auto` response field
`floor_region_polygon`. It is **not** persisted CTS-side and **not** encoded in protobuf —
only the CC BFF consumes it.

### Coordinate spaces (non-negotiable)

| Space | Units | Description |
|---|---|---|
| `floor_region_polygon` | normalised image [0, 1] | `[col/image_w, row/image_h]` — x is column, y is row |
| `visibility_polygon` | normalised floor-plan [0, 1] | `[x_m / fp_width_m, y_m / fp_height_m]` |
| Floor-plan metres | m | intermediate; passed through homography |

`floor_region_polygon` and `visibility_polygon` are in **different spaces**. They share the
[0, 1] normalisation convention but are not interchangeable. Do not feed one where the other
is expected.

### CC persistence and endpoint

`cognitive-companion` stores the polygon in `cts_cameras`:

| Column | Type | Set by |
|---|---|---|
| `floor_region_polygon` | JSONB | auto-calibrate endpoint or manual POST |
| `floor_region_source` | `"depth_auto"` \| `"manual"` | always set with polygon |
| `floor_region_set_at` | `timestamptz` | always set with polygon |

Added in Alembic migration `0011_floor_region`.

Endpoints:

- `POST /api/v1/cts/calibration/auto/{camera_id}` — auto-cal stores the polygon as a side
  effect (source `"depth_auto"`) without committing the homography draft.
- `POST /api/v1/cts/calibration/floor_region/{camera_id}` — operator endpoint that accepts a
  polygon in normalised image [0, 1] coordinates and an optional `source` (`"manual"` default,
  `"depth_auto"` accepted). If the camera already has a committed homography the visibility
  polygon is recomputed immediately.

### Visibility polygon derivation

`compute_visibility_from_homography` in `backend/services/cts_visibility.py`:

- **With** `floor_region_polygon`: densifies each polygon edge to one sample per 10 px
  (`_DENSIFY_STEP_PX = 10`) to capture lens distortion, then projects through `H`.
- **Without** `floor_region_polygon`: falls back to the original 80-point image-border
  sampling (4 edges × 20 points) and logs `visibility_polygon_no_floor_region`.

Edge densification is important: a straight line in image space curves in floor-plan space due
to radial lens distortion. Sampling only at polygon vertices would miss the curve.

### Calibration UI flow

1. Operator clicks **Auto-Calibrate**. The BFF forwards the request to CTS; CTS returns
   `floor_region_polygon` alongside the draft matrix.
2. CC stores the polygon (source `"depth_auto"`) and returns it in the response.
3. The calibration view shows a **green dashed polygon** overlaid on the camera snapshot.
   Vertices are draggable.
4. Operator reviews the polygon: if walls are enclosed, they drag vertices inward to exclude
   them.
5. **Save Region** calls `POST /floor_region/{camera_id}` (source `"manual"` when dragged,
   `"depth_auto"` when accepted as-is). If a committed homography exists the visibility
   polygon is recomputed immediately.
6. Hand-drawing is the complete no-model fallback: the operator can draw a polygon even
   without running auto-calibration.

### Tests

| Suite | What is tested |
|---|---|
| `tracking-orchestrator/tests/test_floor_region.py` | Concave hull, convex fallback, axis order, region_fraction filter, None on < 3 inliers |
| `cognitive-companion/backend/tests/unit/services/test_cts_visibility.py` | Wall exclusion, backward-compat fallback, edge densification, degenerate input |
| `cognitive-companion/backend/tests/routers/test_cts_calibration.py` — `TestFloorRegionEndpoint` | auto-cal stores polygon; manual save persists source; coord validation; `_refresh_visibility_polygon` uses stored polygon |
| `cognitive-companion/frontend/tests/views/CTSCalibrationView.spec.js` — floor-region suite | `floorRegionDraft` populated from auto-cal; `saveFloorRegion` calls service; `discardFloorRegion` clears draft; `onCameraChange` clears draft |
