---
name: cts-pipeline
description: Use when changing CTS frame-pipeline stages, execution paths, tracker-round side effects, batching behavior, or replay fixture formats.
---

# CTS Pipeline

When a pipeline change affects PH identity evidence, identity resolution, gallery seeding, or
identity revisions, also load
`/home/sriram/code/nanai/continuous-tracking/.claude/skills/cts-identity-governance/SKILL.md`.

This skill maps the configured frame-processing pipeline so you can reason about where a new feature fits, how stages interact, and what constraints apply when adding or modifying a stage.

## The golden rule

> A stage is "live" only if it is wired into the `stages` list built inside `FrameProcessingPipeline.initialize()` (`app/pipeline/frame_pipeline.py`). A file under `app/pipeline/stages/` that is not in that list does not run.

Always verify by reading `frame_pipeline.py` after adding a stage.

## Execution paths

A frame reaches the stages through one of three routes. The route is selected during `FrameProcessingPipeline.initialize()` from batching configuration and detector availability.

| Path | Active when | Route |
|---|---|---|
| Direct | `batch_window_s == 0` | `_handle_frame` -> `_process_frame_direct` -> `_process_frame` -> full `_stage_runner` |
| Per-camera batch | `batch_window_s > 0`, with cross-camera detector batching off or no detector present | `FrameBatcher` -> `_handle_batch` -> `_process_frame` once per frame -> full `_stage_runner` |
| Cross-camera batch | `batch_window_s > 0` + `cross_camera_detector_batching` + detector present | `_handle_cross_camera_batch` -> shared fetch/detect -> `_process_cross_camera_post_detect_batch` round loop -> `_pre_world_runner` / `world_tracking_stage.run_many` / `_post_world_runner` |

The production defaults in `config/settings.yaml` are `batch_window_s: 0.5` and `cross_camera_detector_batching: true`, so production normally uses the cross-camera path. Testing only `_process_frame` tests a path production does not normally run.

**Execution-path rule.** Any per-frame or per-round side effect, including cache epochs, metrics, camera-row seeding, staleness checks, acknowledgements, and state pruning, must be wired into all three paths. A tracker round means one call to `WorldTrackingStage`: one context on the direct and per-camera paths, or the earliest remaining context from each camera on the cross-camera path.

Current orchestration side effects are placed as follows:

| Side effect | Direct | Per-camera batch | Cross-camera batch |
|---|---|---|---|
| Gallery-cache epoch | `_process_frame` calls `_begin_tracker_round` once per frame | Same, because `_handle_batch` delegates each frame to `_process_frame` | The round loop calls `_begin_tracker_round` once before pre-world stages and `run_many` |
| Staleness check | `_process_frame` calls `_is_stale` | Same through `_process_frame` | `_handle_cross_camera_batch` calls `_is_stale` before context creation, fetch, and round assembly |
| Camera-row seeding | `_process_frame_direct` calls `_ensure_camera_row` before `_process_frame` | `_handle_batch` calls `_ensure_camera_row` before each `_process_frame` | The round loop calls `_ensure_camera_row` before each context enters `_pre_world_runner` |
| Success ACK and latency log | `_process_frame_direct` after `_process_frame` | `_handle_batch` after each `_process_frame` | `_process_post_world_context` after `_post_world_runner` |
| Failure metrics | `_process_frame_direct` increments `frames_failed_total` | `_handle_batch` increments `frames_failed_total` | Fetch, detector, pre-world, world, and post-world failure branches increment `frames_failed_total` |

`_begin_tracker_round` is the required chokepoint for work that must happen exactly once before world tracking, such as invalidating the per-round `GalleryCache`. Any new route that can reach `WorldTrackingStage` must call it. Do not add a cache invalidation directly to an individual stage.

Path coverage is split across these test modules:

| Path or contract | Test coverage |
|---|---|
| Direct pipeline | `tests/test_pipeline.py`; gallery epoch specifically in `tests/pipeline/stages/test_pipeline_fixes.py::TestGalleryCacheLifecycle::test_non_batched_path_invalidates_cache_each_frame` |
| Per-camera batching | `tests/test_batcher.py` covers grouping, ordering, timing, and dispatch to the per-camera handler; direct pipeline behavior delegated through `_process_frame` is covered by `tests/test_pipeline.py` |
| Cross-camera batching | `tests/test_batcher.py` covers mixed-camera dispatch; `tests/pipeline/stages/test_pipeline_fixes.py::TestCrossCameraPostDetectBatch` covers round assembly, post-world ACKs, and gallery invalidation |
| Stage-runner segment shapes | `tests/unit/pipeline/stages/test_stage_runner_segments.py` |

## Stage map (in execution order)

All stage classes live in `app/pipeline/stages/`. Each implements `FrameStage` from `app/pipeline/stages/base.py`.

The list is configuration-dependent. A conditional stage must name its enabling condition in this table, and tests must cover both the wired and unwired pipeline shapes.

| # | Class | File | Enabled when | Responsibility |
|---|-------|------|--------------|----------------|
| 1 | `FetchStage` | `fetch.py` | Always | Fetch JPEG from MinIO by the key in `FrameReady`; stale frames (> 30 s) are ACKed and skipped by pipeline orchestration before this stage |
| 2 | `DetectStage` | `detect.py` | Always wired; skeleton mode bypasses the stage runner when no detector exists | YOLO26L via Triton; IoU dedup of near-duplicate bounding boxes |
| 3 | `PrivacyStage` | `privacy.py` | Always | Blur/mask pixels inside operator-drawn privacy polygons; drop detections whose foot-point falls in a drop zone |
| 4 | `SpatialProjectionStage` | `spatial_projection.py` | Always | Apply per-camera homography matrix to compute calibrated `FloorPoint` for every detection; runs before inference so the dedup pass can use floor positions |
| 5 | `InferenceStage` | `inference.py` | Always wired; ReID and pose work depend on their configured clients and flags | SOLIDER-REID appearance embedding + RTMPose pose estimation; both run per-detection |
| 6 | `FaceIdentityStage` | `face_identity.py` | Always wired; remote calls require `face_id.enabled`, a non-empty `face_id.url`, and per-camera face ID enabled | Rate-limited ArcFace calls to `person-identification-service`; produces `FaceAnchor` per detection. With no client, the stage returns without work |
| 7 | `WorldTrackingStage` | `world_tracking.py` | Always | Builds `WorldObservation`s (`ObservationGeometry` and per-observation covariance for calibrated cameras; synthetic virtual-tile floor point for uncalibrated cameras), then delegates to `WorldTracker.step`: (a) Kalman predict all open PHs, (b) pre-association cross-camera dedup, (c) Hungarian association over floor-plane cost matrix, (d) PH update/spawn/close, (e) Bayesian identity resolution |
| 8 | `DetectionBackfillStage` | `detection_backfill.py` | Always | Writes `ph_id` onto each detection from the WorldTracker's assignment map; downstream stages read `detection.ph_id` not a separate map |
| 9 | `ClosePHStage` | `trajectory.py` | Always | Closes PHs that were active in the previous frame but absent in this one (`terminated_ph_ids = prev_active_ph_ids - current_ph_ids`); calls `TrajectoryWriter.close_track` to finalize dwell segments. Defined alongside `TrajectoryStage` in `trajectory.py`, not in `frame_pipeline.py` |
| 10 | `PostureStage` | `posture_stage.py` | Always | Keypoint geometry classifier per detection: `lying`, `sitting`, `standing`, `unknown` |
| 11 | `FallDetectionStage` | `fall_detection.py` | `fall_detection.enabled` and signal publishing infrastructure is available through `signals_enabled` | Extracts fall features, evaluates impact/escalation state, and persists/publishes fall signals |
| 12 | `TrajectoryStage` | `trajectory.py` | Always | Writes `person_trajectories` and `room_dwells` hypertable rows |
| 13 | `KeyframeStage` | `keyframes.py` | Always | Periodic + identity-change keyframe sampling to `tagged_keyframes` and `scene.samples` stream |
| 14 | `RevisionsStage` | `revisions.py` | Always | Publishes `IdentityRevision` protos and triggers retroactive cross-table rewriting |
| 15 | `TrailsStage` | `posture_trails.py` | Always | Maintains per-PH foot-point ring buffer published in `TrackingEvent` |
| 16 | `PublishStage` | `publish.py` | Always | Assembles and publishes `TrackingEvent` proto to `tracking.events` Redis stream |

## FrameContext: shared state between stages

`FrameContext` (`app/pipeline/frame_context.py`) is passed to every stage's `run(ctx)` method. Key fields:

| Field | Set by | Read by |
|-------|--------|---------|
| `frame` | `FetchStage` | All stages |
| `domain_detections` | `DetectStage` | `PrivacyStage`, `SpatialProjectionStage`, `InferenceStage`, `DetectionBackfillStage` |
| `geometry_by_detection` | `WorldTrackingStage._build_observations` | Geometry-aware posture weighting and observation persistence |
| `world_snapshots`, `outcome_decisions`, `committed_ids`, `det_to_ph`, `active_ph_ids`, `new_revisions`, `ph_born_at_by_id` | `WorldTrackingStage._populate_context` | `DetectionBackfillStage`, `ClosePHStage`, `TrajectoryStage`, `RevisionsStage`, `PublishStage` (there is no single `world_tracker_result` field; the result is fanned out into these named fields) |
| `face_anchors`, `_face_evidence` | `FaceIdentityStage` | `WorldTrackingStage` |
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

Cross-camera batches must stay together through `WorldTrackingStage`. Detector batching followed by per-camera world-tracking calls prevents `dedup_observations()` from seeing overlapping-camera observations and causes duplicate unknown PH creation. Split camera-local persistence/publication only after the world step has assigned PH IDs.

## Replay fixture ownership

There are two fixture families at different replay altitudes:

| Family | Format and location | Loader and replay target | Generator |
|---|---|---|---|
| World-tracker replay | Length-prefixed JSON `WorldObservation` steps in `tests/fixtures/frame_replays/*.bin`, with `*.truth.json` sidecars | `tests/integration/_replay.py::load_fixture` and `load_truth`; replayed directly through `WorldTracker.step` by `test_world_tracker_e2e.py`, `test_m5_cross_camera.py`, `test_m2_ph_continuity.py`, and related integration tests | `scripts/synthesize_replay_fixture.py`; `scripts/record_replay_fixture.py` captures approved replay inputs |
| Fall sequence | Header plus `FallFrameInput` rows as JSON Lines in `tests/fixtures/fall_sequences/*.jsonl` | `tests/integration/test_fall_sequences.py::_load_fixture`; replayed through `FallFeatureExtractor` and `FallDetector`. `scripts/calibrate_fall_thresholds.py` has a matching calibration loader | `scripts/synthesize_fall_sequence.py` |

World-tracker fixture observations carry `calibrated`, optional `face_anchor`, `orientation`,
`orientation_confidence`, `floor_cov_random`, `footpoint_reliable`, and (from M09) `floor_residual_m`.
Their truth sidecars carry person labels per detection plus expected events; M09 fixtures also carry
scenario-specific extra fields (e.g., `truth_position_mm`, `cam_b_dropout_at_step`, `truth_trajectory_mm`,
`posture_truth`, `per_camera_posture_scores`). These proofs bypass the full frame pipeline, so they do
not exercise MinIO, Triton, detection, face/pose inference, stage ordering, or the three execution routes.

Fall fixtures carry bbox, keypoints, posture scores, floor speed, motion energy, timing, room metadata, and a required expectation (`detect`, `no-detect`, or `warning-max`). They exist separately because `WorldObservation` replay cannot exercise `FallDetectionStage` inputs.

**M09 acceptance fixtures (five scenarios, committed 2026-06-18):**

| Fixture | Scenario | Key truth sidecar extras |
|---|---|---|
| `stationary_two_camera` | Person still; cam-A (+0.3 m offset) and cam-B (-0.2 m offset); cam-B drops at step 30 | `truth_position_mm`, `cam_b_dropout_at_step` |
| `slow_shuffle` | 0.30 m/s walk along x; single camera | `truth_trajectory_mm`, `truth_speed_m_s` |
| `oblique_single_camera` | Fixed point; anisotropic R `[0.16, 0, 0, 0.01]` (sigma_x=0.4 m, sigma_y=0.1 m) | `truth_position_mm` |
| `posture_disagreement_four_camera` | 4 cameras (LEFT/RIGHT/BACK/FRONT); 3 sitting, 1 standing; geometry-aware fusion should output sitting | `posture_truth`, `per_camera_posture_scores` |
| `moving_then_stop` | Walk 20 frames at 0.30 m/s then stop for 40 frames; ZUPT engages at stop | `truth_trajectory_mm`, `walk_end_step`, `truth_stop_pos_mm` |

**M09 measurement harness:** `tests/integration/_fusion_metrics.py` provides `run_replay()` and `compute_eigen_ratio()`.
`run_replay(steps, legacy_mode=True)` strips `floor_cov_random` (forces isotropic) and sets `zupt_consecutive_frames=99999`
(disables ZUPT) — TEST-ONLY toggle. Gate tests live in `tests/integration/test_fusion_acceptance.py` (6 tests,
`@pytest.mark.integration`, use `InMemoryPHRepository`, no Postgres required).

**Fixture ownership rule.** A fixture-format change must update the generator, every loader or calibration reader for that family, round-trip or compatibility tests, regenerated committed fixtures, and this skill in one PR. Do not hand-edit generated fixture payloads.

## Quality capture: where it happens and what travels downstream

`CropQuality` (`app/pipeline/crop_quality.py`) is the single scorer. It is called inside `WorldTracker.step()` to compute per-observation quality and to update `PersonHypothesis.mean_quality` (EMA: `0.1 * obs.quality + 0.9 * prev`). Crop quality now scales `Σ_px` in the observation model rather than directly weighting dedup fusion.

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
