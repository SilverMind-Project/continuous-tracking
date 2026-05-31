---
name: cts-pipeline
description: A map of the CTS frame pipeline stages (app/pipeline/stages/), the StageRunner contract, where to insert a stage, the dedup/quality concepts, and the rule that a stage is "live" only if wired in frame_pipeline.py.
---

# CTS Pipeline

This skill maps the 15-stage frame processing pipeline so you can reason about where a new feature fits, how stages interact, and what constraints apply when adding or modifying a stage.

## The golden rule

> A stage is "live" only if it is wired into the `stages` list built inside `FrameProcessingPipeline.initialize()` (`app/pipeline/frame_pipeline.py`). A file under `app/pipeline/stages/` that is not in that list does not run.

Always verify by reading `frame_pipeline.py` after adding a stage.

## Stage map (in execution order)

All stage classes live in `app/pipeline/stages/`. Each implements `FrameStage` from `app/pipeline/stages/base.py`.

| # | Class | File | Responsibility |
|---|-------|------|----------------|
| 1 | `FetchStage` | `fetch.py` | Fetch JPEG from MinIO by the key in `FrameReady`; stale frames (> 30 s) are XACK'd and skipped |
| 2 | `DetectStage` | `detect.py` | YOLO26L via Triton; IoU dedup of near-duplicate bounding boxes |
| 3 | `PrivacyStage` | `privacy.py` | Blur/mask pixels inside operator-drawn privacy polygons; drop detections whose foot-point falls in a drop zone |
| 4 | `SpatialProjectionStage` | `spatial_projection.py` | Apply per-camera homography matrix to compute calibrated `FloorPoint` for every detection; runs before inference so the dedup pass can use floor positions |
| 5 | `InferenceStage` | `inference.py` | SOLIDER-REID appearance embedding + RTMPose pose estimation; both run per-detection |
| 6 | `FaceIdentityStage` | `face_identity.py` | Rate-limited ArcFace calls to `person-identification-service`; produces `FaceAnchor` per detection |
| 7 | `WorldTrackingStage` | `world_tracking.py` | Builds `WorldObservation`s (synthetic virtual-tile floor point for uncalibrated cameras), then delegates to `WorldTracker.step`: (a) Kalman predict all open PHs, (b) pre-association cross-camera dedup, (c) Hungarian association over floor-plane cost matrix, (d) PH update/spawn/close, (e) Bayesian identity resolution |
| 8 | `DetectionBackfillStage` | `detection_backfill.py` | Writes `ph_id` onto each detection from the WorldTracker's assignment map; downstream stages read `detection.ph_id` not a separate map |
| 9 | `ClosePHStage` | (see frame_pipeline.py) | Closes PHs that were active in the previous frame but absent in this one; writes final trajectory/dwell segments |
| 10 | `PostureStage` | `posture_stage.py` | Keypoint geometry classifier per detection: `lying`, `sitting`, `standing`, `unknown` |
| 11 | `TrajectoryStage` | `trajectory.py` | Writes `person_trajectories` and `room_dwells` hypertable rows |
| 12 | `KeyframeStage` | `keyframes.py` | Periodic + identity-change keyframe sampling to `tagged_keyframes` and `scene.samples` stream |
| 13 | `RevisionsStage` | `revisions.py` | Publishes `IdentityRevision` protos and triggers retroactive cross-table rewriting |
| 14 | `TrailsStage` | `posture_trails.py` | Maintains per-PH foot-point ring buffer published in `TrackingEvent` |
| 15 | `PublishStage` | `publish.py` | Assembles and publishes `TrackingEvent` proto to `tracking.events` Redis stream |

## FrameContext: shared state between stages

`FrameContext` (`app/pipeline/frame_context.py`) is passed to every stage's `run(ctx)` method. Key fields:

| Field | Set by | Read by |
|-------|--------|---------|
| `frame` | `FetchStage` | All stages |
| `domain_detections` | `DetectStage` | `PrivacyStage`, `SpatialProjectionStage`, `InferenceStage`, `DetectionBackfillStage` |
| `world_tracker_result` | `WorldTrackingStage` | `DetectionBackfillStage`, `ClosePHStage`, `TrajectoryStage`, `PublishStage` |
| `face_anchors` | `FaceIdentityStage` | `WorldTrackingStage` |
| `event_time` | Initialized at context creation | All stages (do not use `datetime.now(UTC)` inside a stage unless you specifically need wall-clock time) |

**Context ownership rule.** Each field has one producing stage. No stage may overwrite a field produced by another stage. Shared logic between two stages becomes a small service injected into both.

## StageRunner contract

`StageRunner` (`app/pipeline/stages/base.py`) iterates the stage list and calls `await stage.run(ctx)` in order. Exceptions propagate to the `_consume_loop` handler, which logs and retries the frame. Individual stages must not catch broad exceptions; let them propagate.

A stage that is computationally expensive but optional (e.g., pose estimation with an unavailable model) should degrade gracefully and log at `warning` level, but must not silently skip work without a log.

## Where to insert a new stage

**Before `WorldTrackingStage`:** stages that produce evidence consumed by the tracker (floor points, face anchors, ReID embeddings). Current examples: `SpatialProjectionStage`, `InferenceStage`, `FaceIdentityStage`.

**After `WorldTrackingStage` and before `TrajectoryStage`:** stages that enrich detections with PH-level data. Current example: `DetectionBackfillStage`, `ClosePHStage`, `PostureStage`.

**After `TrajectoryStage`:** stages that sample or publish; they read final per-PH trajectory state. Current examples: `KeyframeStage`, `RevisionsStage`, `TrailsStage`, `PublishStage`.

Adding a stage to the wrong position causes the stage to read stale context. When in doubt, check which `FrameContext` fields your stage reads and confirm they are set by a prior stage.

## Cross-camera dedup: how it works and where not to touch it

The pre-association floor-point dedup runs inside `WorldTrackingStage` by calling `dedup_observations()` (`app/tracking/world/dedup.py`) before the `associate()` call. See the engineering-standards skill for the algorithm and the configuration knobs (`dedup_enabled`, `dedup_max_distance_m`, `dedup_require_no_face_conflict`).

**Do not** add dedup logic in `WorldTrackingStage.run()` or in `associate()`. All dedup changes go in `dedup.py`. The `dedup_observations()` function is pure (no I/O, no DB) and is tested independently.

The integration proof for the hallway/bathroom case is `tests/integration/test_world_tracker_e2e.py::test_hallway_bathroom_one_person` using `tests/fixtures/frame_replays/hallway_bathroom_door.bin`.

## Quality capture: where it happens and what travels downstream

`CropQuality` (`app/pipeline/crop_quality.py`) is the single scorer. It is called inside `WorldTracker.step()` to compute per-observation quality (used for dedup representative selection) and to update `PersonHypothesis.mean_quality` (EMA: `0.1 * obs.quality + 0.9 * prev`).

`mean_quality` travels via the `IdentitySnapshot` proto field to the CC side, where `TrackingEventSubscriber` stores it in the presence segment and `PersonLocationEnvelope` surfaces it as the `quality` field. Do not add a second scorer; do not compute quality on the CC side.

## Adding a stage: checklist

1. Create `app/pipeline/stages/my_stage.py` implementing `FrameStage`.
2. Add `class MyStage(FrameStage): ...` with `name = "my_stage"` and `async def run(self, ctx: FrameContext) -> None`.
3. Import `MyStage` in `app/pipeline/stages/__init__.py`.
4. Wire the stage into the `stages` list inside `FrameProcessingPipeline.initialize()` at the correct position.
5. Add a unit test in `tests/unit/pipeline/stages/test_my_stage.py` with an in-memory `FrameContext`.
6. If the stage has config knobs, add them to `PipelineConfig` (`app/pipeline/config.py`) with documented defaults.
7. Register a Prometheus counter or histogram if the stage can fail or has measurable latency.

## Verification commands

```bash
# Full quality gate including integration proofs (requires Docker)
cd continuous-tracking && make ci

# Fast unit-only gate
cd continuous-tracking && make check

# Check that your stage is wired in (must print your stage class name)
grep "MyStage" tracking-orchestrator/app/pipeline/frame_pipeline.py

# Run integration proofs only
tracking-orchestrator/.venv/bin/pytest tests/integration/ -m integration -v
```
