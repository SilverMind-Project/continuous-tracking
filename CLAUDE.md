# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a design-specification repository for a **continuous tracking system for monitoring seniors with early dementia**. It contains 6 phase documents (7,387 lines total) that define the architecture, data model, and implementation plan for a system that:

- Tracks motion and activities of seniors via RTSP camera streams
- Performs person re-identification across cameras using face + body embeddings
- Detects dementia-relevant patterns (pacing, sundowning, bathroom anomalies, stillness)
- Integrates with an existing `cognitive-companion` system

## Document Order (Read in This Sequence)

1. **phase-0-design-review.md** (1,686 lines) — The load-bearing foundation. Critiques and revises phases 1-5. Contains the revised identity model, runtime partitioning, storage abstraction, message transport, dementia activity layer, privacy/audit, schema, UI deliverables, validation matrix, and 16-week implementation playbook (M1-M10). **Read this first.**
2. **phase-1-architecture.md** (1,013 lines) — System architecture, identity model, database schema, layering rules.
3. **phase-2-rtsp-ingestion.md** (1,014 lines) — Go service for RTSP ingest, frame decode, motion gating, MinIO upload.
4. **phase-3-tracking-reid.md** (1,237 lines) — Tracking orchestrator (Python), BoT-SORT, cross-camera association, identity resolution with Bayesian posterior, retroactive revision.
5. **phase-4-scene-semantic.md** (735 lines) — Scene analysis, semantic memory integration, VLM pipeline.
6. **phase-5-backend-integration.md** (1,702 lines) — Cognitive companion integration, backend routers, Vue views, gateway contract, validation.

## Key Architecture Decisions (from phase-0)

- **Runtime**: Go service (`rtsp-ingress`) for RTSP/frame handling + Python service (`tracking-orchestrator`) for ML logic + NVIDIA Triton for GPU inference batching.
- **Identity model**: Bayesian posterior over identities, not single-assignment. Gallery of embeddings per person (not one embedding). Retroactive revision protocol.
- **Transport**: Redis Streams (not PubSub) with consumer groups, XACK, replay capability.
- **Storage**: Postgres/TimescaleDB with pgvector (HNSW index). Repository pattern — no raw SQL in services.
- **Models**: YOLO11m (detection), SOLIDER-REID (768-dim body ReID), RTMPose (pose), ArcFace (face ID).
- **Dementia signals**: Pacing, sundowning, bathroom anomaly, stillness, nighttime movement, absence — computed from trajectory data against per-person baselines.
- **Integration**: All external traffic routes through `cognitive-companion` as a BFF gateway. No direct UI access to CTS services.

## Implementation Playbook

The design defines 10 milestones (M1-M10) spanning 16+ weeks. Never implement out of order:

| Milestone | Weeks | Scope |
|-----------|-------|-------|
| M1 | 1-2 | Protobuf contracts, repository interfaces, docker-compose, CI — **COMPLETE** |
| M2 | 3-4 | rtsp-ingress (Go) — **COMPLETE** |
| M3 | 5-6 | Triton models + benchmark harness — **COMPLETE** |
| M4 | 7-8 | Tracking orchestrator skeleton (per-camera tracking, tracklets) — **COMPLETE** |
| M5 | 9-10 | Identity resolution, retroactive revision — **COMPLETE** |
| M6 | 11 | Trajectories, dwells, keyframes — **COMPLETE** |
| M7 | 12-13 | Admin UI (cameras, calibration, privacy, adjacency) — **COMPLETE** |
| M8 | 14-15 | Dementia signals, dashboard, keyframes — **COMPLETE** |
| M9 | 16 | Live view, identity corrections, runtime integration — **NEXT** |
| M10 | 16+ | K8s, observability, docs, performance hardening |

## Coding Rules (Derived from M4 Code Review)

These rules were extracted from bugs found during the M4 implementation review. Apply them to all future milestone work.

### Python / asyncpg

1. **Use `$N` placeholders in all asyncpg SQL.** asyncpg uses `$1, $2, ...` positional parameters. Never use `%s` (psycopg2 syntax) or `?` (sqlite3). `executemany` also requires `$N` style — one row's worth of placeholders per statement.

2. **`datetime.now(UTC)` everywhere.** Never call `datetime.now()` without a timezone argument. Bare `datetime.now()` produces a timezone-naive object that conflicts with TimescaleDB's timestamptz columns and the domain model's aware datetimes. Always `from datetime import UTC` and use `datetime.now(UTC)`.

3. **Declare all instance attributes in `__init__`.** Every attribute a class owns must be declared in `__init__` with its correct `Optional[T] | None` type, even if its real value is set later (e.g., in an `initialize()` method). Attributes first assigned outside `__init__` cause `AttributeError` on any access before that method runs, and mypy cannot track them.

4. **Never mutate frozen dataclasses.** `@dataclass(frozen=True)` raises `FrozenInstanceError` at runtime on any `instance.attr = value` assignment — even if mypy is silenced with `# type: ignore`. For transport-layer metadata that must travel alongside a frozen message (e.g., Redis message IDs), maintain a side-channel `dict[int, T]` keyed by `id(obj)` on the owning transport object. Clean up entries after use.

### SQL correctness

1. **PostgreSQL array concatenation uses `||` directly.** In `ON CONFLICT DO UPDATE SET`, to merge two existing array columns write `col = EXCLUDED.col || table.col`. The `array[...]` constructor syntax is for array literals only; `array[EXCLUDED.col || table.col]` is a syntax error.

2. **Conditional SQL injection: replace a unique anchor, not a suffix keyword.** When adding a `WHERE ... AND extra_clause` branch at runtime via `str.replace`, replace a unique substring that includes the insertion point (e.g., `"WHERE global_track_id = $1"` → `"WHERE global_track_id = $1\n    AND extra_clause"`). Never replace a trailing keyword like `"LIMIT 100"` and prepend `AND ...` — this inserts a predicate after `ORDER BY`, producing invalid SQL.

### Tracking / data-structure correctness

1. **Build `active_tracks` from `.items()`, not `enumerate(.values())`.** When you need to map a position index back to a dict key (e.g., after Hungarian assignment), you must capture the key at construction time. `enumerate(dict.values())` gives positions 0..N that diverge from key names after any deletion. Use `[(key, value) for key, value in dict.items() if ...]` and return keys from assignment functions, not positions.

2. **Use explicit reverse maps for cross-namespace ID lookups.** When two modules use different ID namespaces for the same logical entity (e.g., `local_track_id` strings from the tracker vs. UUID `tracklet_id`s in the tracklet manager), maintain an explicit `dict[local_id, uuid_id]` reverse map on the owning object. Document which namespace each dict is keyed by.

3. **Increment shared counters exactly once per code path.** A pattern like:

   ```python
   state.count += 1          # unconditional increment
   if state.count >= threshold:
       ...
   else:
       state.count += 1      # BUG: second increment in the else branch
   ```

   silently doubles the increment on the non-threshold path. Use a single increment before the branch.

4. **Prefer method parameters over object fields when they carry the same semantic.** If a helper method receives `embedding` as a parameter and also has access to `detection.embedding` (a domain placeholder field), use the parameter — it carries the live value from Triton. Domain fields like `Detection.embedding` are zero-initialized placeholders until M5 identity resolution fills them in.

## Engineering Standards (from phase-0, section 0.18)

- **Python**: 3.12, `from __future__ import annotations`, strict mypy on domain/services/storage/transport, Pydantic v2 at boundaries, frozen dataclasses internally.
- **Go**: 1.24+, `-race` mandatory, `go vet`, `staticcheck`, `golangci-lint`.
- **Proto**: `buf lint`, `buf breaking` against last merged commit.
- **Layering**: core -> domain -> storage -> services -> transport -> routers (upward dependencies forbidden). Enforced by `import-linter`.
- **Error taxonomy**: Standardized `ErrorCode` enum, no stack traces over the wire.
- **CI gates**: ruff, mypy, import-linter, pytest (unit + testcontainers), go test -race, buf lint, docker build, helm template, replay suite, Playwright E2E.

## Quality Gate (Automated Checks)

All Python code must pass the full quality gate before committing. Run locally before every PR:

```bash
make check        # Python: ruff check + ruff format + mypy + pytest
make all-check    # Python + Go + proto (full repo gate)
```

Pre-commit hooks mirror the CI checks (ruff, mypy, golangci-lint, buf). Install with:
```bash
pre-commit install
```

**Rule: never write code that fails the quality gate.** Every new feature, fix, or test must be verified with `make check` (Python-only) or `make all-check` (full repo) before committing. The CI pipeline runs the same checks — local runs are the fast feedback loop.

**Always use the Python venv inside the project folder** (`tracking-orchestrator/.venv/`). Never rely on global system Python for running code, tests, linting, or type-checking. The Makefile targets (`make check`, `make lint`, `make test`, `make mypy`) and CI all use the project-local venv.

For the tracking-orchestrator specifically:
- All repository methods are async — tests must be `async def` with `await`
- Domain types are frozen dataclasses; Pydantic models only at boundaries

## Current Progress

**M1 (Protobuf contracts, repository interfaces, docker-compose, CI) — COMPLETE.**
Implemented files:
- `proto/continuoustracking/v1/{tracking,frame}.proto` — message contracts
- `proto/buf.yaml` — buf build config
- `tracking-orchestrator/app/domain/__init__.py` — frozen dataclasses (Detection, Tracklet, GlobalTrack, IdentityRevision, GalleryEntry, CameraConfig, StreamConfig, PersonActivity)
- `tracking-orchestrator/app/storage/base.py` — 5 repository protocols + 5 InMemory implementations
- `tracking-orchestrator/migrations/0001_init.sql` — full schema with TimescaleDB hypertables + pgvector HNSW
- `tracking-orchestrator/app/main.py` — FastAPI app factory
- `tracking-orchestrator/pyproject.toml` — dependencies + tool config
- `tracking-orchestrator/tests/test_in_memory_repos.py` — repo tests
- `rtsp-ingress/cmd/server/main.go` — health endpoint (M3+ stubs)
- `rtsp-ingress/internal/{rtsp,motion,media,streams,reconciler}/` — stub packages
- `docker-compose.yml` — TimescaleDB, Redis, MinIO, Triton, orchestrator, ingress
- `.github/workflows/ci.yml` — Python/Go/proto CI
- `.pre-commit-config.yaml` — ruff, mypy, golangci-lint, buf
- `Makefile` — lint, format, test, docker, proto commands
- `cognitive-companion/config/settings.yaml` — `cts.enabled: false` feature flag

**M2 — Implemented.** `rtsp-ingress/` Go service with full source code and 26 unit tests across 7 packages (config, motion, media, reconciler, rtsp, streams, cmd/server). All tests pass. See `rtsp-ingress/` for implementation details.

**M3 — Implemented.** Triton model repository + inference client module. Implemented files:

- `triton-models/person-detector/config.pbtxt` — YOLO11m TensorRT config (input: `images` [3,640,640], output: `output0` [84,8400], dynamic batching, `max_batch_size: 16`)
- `triton-models/reid-solider/config.pbtxt` — SOLIDER-REID ONNX config (input: `input` [3,256,128], output: `output` [768])
- `triton-models/pose-rtmpose/config.pbtxt` — RTMPose-m ONNX config (input: `input` [3,256,192], outputs: `simcc_x` [17,384], `simcc_y` [17,512])
- `triton-models/{person-detector,reid-solider,pose-rtmpose}/1/.gitkeep` — placeholders for model binaries (generated with export scripts)
- `triton-models/scripts/{export_yolo,export_reid,export_pose}.py` — model export scripts
- `triton-models/README.md` — materialisation and verification instructions
- `tracking-orchestrator/app/inference/{__init__,schemas,triton_client,detector,reid_embedder,pose}.py` — async Triton gRPC client + typed wrappers; `TritonClientProtocol` allows mock injection in tests
- `tracking-orchestrator/scripts/benchmark_triton.py` — p50/p99 sweep at batch [1,4,8,16] with DoD gate check (person-detector p99 ≤ 12ms at batch 8)
- `tracking-orchestrator/notebooks/model_demo.ipynb` — end-to-end demo (JPEG → DetectionBox / Embedding / PoseResult) per model
- `tracking-orchestrator/tests/test_inference.py` — 27 unit tests (mocked Triton, no GPU required); all pass under `make check`

**Outstanding DoD gate**: `tritonserver` loading all three models (`curl :8000/v2/models/ready`) and benchmark p99 ≤ 12ms at batch 8 require materialised `.plan`/`.onnx` files (run export scripts on the target GPU). The code scaffolding is complete and verified.

**M4 — Implemented.** Tracking orchestrator skeleton with per-camera tracking and tracklet lifecycle management. Implemented files:

- `tracking-orchestrator/app/tracking/{__init__,tracker,tracklet_manager}.py` — BoT-SORT-like tracker (Kalman filter + IoU + embedding Hungarian assignment), tracklet-to-track bridging, ID lifecycle management
- `tracking-orchestrator/app/pipeline/{__init__,frame_pipeline}.py` — `FrameProcessingPipeline` wiring transport → inference → tracking → tracklet management → persistence
- `tracking-orchestrator/app/transport/redis_streams.py` — Redis Streams transport with consumer groups, XACK for at-least-once delivery
- `tracking-orchestrator/app/storage/postgres/{__init__,tracking_repo,gallery_repo}.py` — Postgres/pgvector repository implementations
- `tracking-orchestrator/app/main.py` — FastAPI app factory with asyncpg lifespan (Triton + Postgres + Redis)
- `tracking-orchestrator/tests/test_tracker.py` — 36 unit tests (mocked Triton, no GPU required)
- `tracking-orchestrator/tests/test_tracklet_manager.py` — 20 unit tests (tracklet lifecycle, gallery management)
- `tracking-orchestrator/tests/test_pipeline.py` — 10 unit tests (pipeline wiring, event propagation)
- `tracking-orchestrator/tests/test_transport.py` — 14 unit tests (Redis Streams, gallery repo helpers)
- `tracking-orchestrator/pyproject.toml` — added `scipy>=1.14.0` dependency

**DoD verified**: `make check` passes cleanly — ruff check, ruff format, mypy (21 files, 0 errors), import-lint, pytest (90/90 tests across 6 files).

**Post-review fixes applied** (all latent runtime/logic bugs, no regressions):

- `tracker.py`: `_associate` redesigned to return track ID strings instead of position indices; dead-code double-loop removed; `_track_id()` helper eliminated.
- `tracklet_manager.py`: added `_local_to_tracklet` reverse map to resolve the UUID/local-ID key mismatch; fixed double `lost_count` increment; `_append_gallery` now uses the passed Triton embedding, not the placeholder `detection.embedding`.
- `redis_streams.py`: replaced `frame._message_id = ...` mutation (crashes on frozen dataclass) with a `_pending_acks: dict[int, Any]` side-channel keyed by `id(frame)`.
- `tracking_repo.py`: `_SQL_SAVE_DETECTIONS` changed from `VALUES %s` to asyncpg `VALUES ($1...$11)`; `_SQL_SAVE_GLOBAL_TRACK` array concat fixed (`EXCLUDED.col || table.col`, not `array[...]`); `list_identity_revisions` SQL replacement anchored to `WHERE` clause (not `LIMIT`); `datetime.now()` → `datetime.now(UTC)`.
- `frame_pipeline.py`: `PerCameraTrackers` imported at module level; `self._tracker: PerCameraTrackers | None = None` declared in `__init__`.

**M5 — Implemented.** Identity resolution with Bayesian posterior and retroactive revision. Implemented files:

- `tracking-orchestrator/app/tracking/identity_resolver.py` — Bayesian identity resolver: posterior over {identities, UNKNOWN}, evidence from face anchors + ReID gallery + temporal prior, commit rule with prob+margin+evidence-gate. Retroactive revision protocol with rate limiting.
- `tracking-orchestrator/app/tracking/camera_adjacency.py` — Camera adjacency graph with time-bounded reachability for cross-camera association.
- `tracking-orchestrator/app/tracking/cross_camera.py` — Cross-camera associator: merges tracklets from adjacent cameras into GlobalTracks.
- `tracking-orchestrator/app/transport/revision_publisher.py` — Publishes IdentityRevision messages to Redis Streams.
- `tracking-orchestrator/app/storage/postgres/tracking_repo.py` — Updated to persist all IdentityRevision fields (tracklet_ids, previous/new identity, reason, evidence).
- `tracking-orchestrator/app/storage/postgres/gallery_repo.py` — pgvector HNSW search for ReID.
- `tracking-orchestrator/app/pipeline/frame_pipeline.py` — Full pipeline integration: detection → tracking → tracklet management → cross-camera association → identity resolution → persistence → event emission.
- `tracking-orchestrator/app/domain/__init__.py` — M5 domain types: IdentityRevision, PosteriorDist, ResolveOutcome, IdentityDecision, IdentityCandidate, FaceAnchor, GlobalTrack, Tracklet.
- `tracking-orchestrator/app/storage/base.py` — Repository protocols + in-memory implementations (InMemoryTrackingRepository, InMemoryGalleryRepository, InMemoryGlobalTrackRepository).
- `tracking-orchestrator/tests/test_identity_resolver.py` — 25 unit tests (PosteriorDist + IdentityResolver, all pass).
- `tracking-orchestrator/tests/test_camera_adjacency.py` — 12 unit tests (all pass).
- `tracking-orchestrator/tests/test_cross_camera.py` — 7 unit tests (all pass).
- `tracking-orchestrator/tests/test_revision_publisher.py` — 6 unit tests (RevisionPublisher connect, publish, integration).
- `tracking-orchestrator/tests/test_in_memory_repos.py` — 15 unit tests (all repo protocols).
- `tracking-orchestrator/tests/test_pipeline.py` — 5 unit tests (skeleton mode with mocked deps).
- `tracking-orchestrator/migrations/0001_init.sql` — Updated identity_revisions table with tracklet_ids, previous/new identity, reason, evidence columns.

**M5 parameter tuning**: `commit_prob` lowered from 0.85 to 0.65 and evidence-gate added to `_commit()`. The temporal prior (weight=0.6) alone can push the posterior above 0.85 for a single identity (~0.81) or two identities (~0.71), causing false commits. The evidence-gate requires the top identity to appear in the face or ReID likelihood distribution before the commit rule applies. This ensures the prior maintains existing assignments but cannot create new ones without sensory evidence.

**M5 bug fix**: Fixed `_from_face_anchors` bug where the remainder-smoothing loop overwrote the best person_id's likelihood with the per_id value. This caused face evidence to be diluted to uniform when multiple identities existed.

**DoD verified**: `make check` passes cleanly — ruff check, ruff format, mypy (25 files, 0 errors), import-lint, pytest (139/139 tests across 15 files).

**Post-M5 code review fixes** (all 10 issues from `phase-M1-M5-code-review.md` — all FIXED):

- `tracklet_manager.py` (#1 Critical): `_find_embedding_index` replaced placeholder `return 0` with linear search over `detections` by detection ID.
- `tracking_repo.py` (#2 Medium): `save_tracking_event` serializes `capture_time` into `frame_data` JSONB; `get_tracking_event` parses it back.
- `tracking_repo.py` (#3 Medium): `_SQL_LIST_IDENTITY_REVISIONS` SELECT includes `evidence`; row loop parses it.
- `tracker.py` (#4 Medium): Mixed embedding history bias fixed — no-history tracks use `np.full(768, 0.5, dtype=np.float32)`.
- `tracker.py` (#5 Low): `_embedding_distance` normalized to `[0, 1]` (`return (1.0 - cosine_sim) / 2.0`).
- `tracker.py` (#6 Low): Embedding arrays use `dtype=np.float32`.
- `redis_streams.py` (#7 Medium): `_pending_acks` stores `(message_id, time.monotonic())`; `_cleanup_stale_acks()` evicts entries older than 300s.
- `tracklet_manager.py` (#8 Low): `_compute_quality` accepts `*, max_area: int = 1920 * 1080` parameter.
- `gallery_repo.py` (#9 Medium): `search_similar` supports `camera_id` and `max_age_seconds` filters in both in-memory and Postgres impls.
- `camera_adjacency.py` (#10 Low): `within_s` is correctly wired in both call sites in `cross_camera.py`.

**DoD verified (post-review)**: pytest (163/163 tests pass), ruff + mypy clean.

**M6 — Implemented.** Trajectories, room dwells, and keyframe sampling. Implemented files:

- `tracking-orchestrator/migrations/0002_m6_trajectory_keyframes.sql` — `person_trajectories` (TimescaleDB hypertable), `room_dwells`, `tagged_keyframes` tables.
- `tracking-orchestrator/app/domain/__init__.py` — Added M6 types: `PersonTrajectoryPoint`, `RoomDwell`, `TaggedKeyframe`, `PostureType`, `TagReason`.
- `tracking-orchestrator/app/storage/base.py` — Added `TrajectoryRepository`, `KeyframeRepository` protocols and `InMemoryTrajectoryRepository`, `InMemoryKeyframeRepository` implementations.
- `tracking-orchestrator/app/trajectory/trajectory_writer.py` — `TrajectoryWriter`: writes one `person_trajectories` row per committed identity decision; tracks current room per GlobalTrack; opens/closes `room_dwell` intervals on room transitions.
- `tracking-orchestrator/app/sampling/keyframe_sampler.py` — `KeyframeSampler`: periodic sampling enforcing `keyframe_min_interval_s` per tracklet; `trigger_sample()` forces a sample on identity_changed/hazard/dwell_start without resetting the periodic timer.
- `tracking-orchestrator/app/storage/postgres/trajectory_repo.py` — Postgres impl of `TrajectoryRepository`.
- `tracking-orchestrator/app/storage/postgres/keyframe_repo.py` — Postgres impl of `KeyframeRepository`.
- `tracking-orchestrator/app/transport/scene_publisher.py` — `SceneSamplesPublisher`: publishes `TaggedKeyframe` messages to the `scene.samples` Redis Stream (consumer group `scene-worker`).
- `tracking-orchestrator/app/pipeline/frame_pipeline.py` — Updated to wire steps 7 (trajectory writer) and 8 (keyframe sampler) after identity resolution; `PipelineConfig` gains `sampler` and `camera_room_map`; `SceneSamplesPublisher` disconnected on stop.
- `tracking-orchestrator/tests/test_trajectory_writer.py` — 10 unit tests (trajectory points, dwell lifecycle, room transitions, multi-track isolation).
- `tracking-orchestrator/tests/test_keyframe_sampler.py` — 10 unit tests (interval enforcement, trigger override, timer independence, expiry duration, persistence).

**DoD verified**: ruff check, ruff format, mypy (32 files, 0 errors), pytest (181/181 tests across 17 files).

**Notes on M6 scope**: Floor-plane coordinates (`ground_x/y`) default to 0.0 — homography-based projection from pixel coords is M9. `posture` defaults to `"unknown"` — pose integration is M8.

**M7 — Implemented.** Admin UI for cameras, calibration, privacy zones, and camera adjacency. All work lives in `cognitive-companion/` (the BFF gateway), not in `tracking-orchestrator/`. Implemented files:

Backend (cognitive-companion):

- `backend/core/upstream_errors.py` — `UpstreamError` exception with HTTP status forwarding
- `backend/core/service_jwt.py` — EdDSA JWT generation for service-to-service auth (CTS upstream calls)
- `backend/integrations/cts_ingress.py` — `IngressAdminClient`: RTSP test, snapshot proxy, health check, stream reload
- `backend/integrations/cts_orchestrator.py` — `OrchestratorClient`: homography push/get, privacy zones push/get, adjacency push/get, calibration status
- `backend/models/cts_camera.py` — `CtsCamera` SQLAlchemy model (id, name, rtsp_url, location, enabled, homography_json, privacy_zones_json, created_at, updated_at)
- `backend/schemas/cts.py` — Pydantic schemas: `CtsCameraCreate`, `CtsCameraUpdate`, `CtsCameraOut`, `HomographyRequest`, `HomographyOut`, `PrivacyZoneRequest`, `PrivacyZoneOut`, `AdjacencyEdge`, `AdjacencyOut`
- `backend/routers/cts.py` — Feature-flag status/features endpoints; `_cts_enabled()` guard returning 404 with `{"code": "cts.disabled"}` when `cts.enabled` is falsy
- `backend/routers/cts_cameras.py` — Camera CRUD (9 endpoints): list, create, get, patch, delete, test-connect, snapshot, health, reload
- `backend/routers/cts_calibration.py` — Calibration endpoints (6 endpoints): homography fit via `compute_homography()` (OpenCV RANSAC, min 4 points), privacy zones replace, adjacency replace
- `backend/tests/routers/test_cts.py` — 8 tests (feature flag on/off, status, features)
- `backend/tests/routers/test_cts_cameras.py` — 16 tests (CRUD, test-connect, snapshot, health, reload)
- `backend/tests/routers/test_cts_calibration.py` — 5 tests (homography fit, privacy zones, adjacency)

Frontend (cognitive-companion):

- `frontend/src/services/cts.js` — CTS API client: cameras CRUD, snapshot (returns blob URL with lifecycle management), calibration, privacy zones, adjacency
- `frontend/src/views/admin/CTSCamerasView.vue` — Camera roster table (enabled chip, calibrated icon, privacy zone count); Add/Edit/Delete dialogs; RTSP test dialog; snapshot preview dialog
- `frontend/src/views/admin/CTSCalibrationView.vue` — Click-to-place homography calibration: snapshot with crosshair cursor, SVG point overlay, floor coordinate inputs, RANSAC fit, residual table, status chip
- `frontend/src/views/admin/CTSPrivacyView.vue` — Per-camera privacy zone editor: zone cards with inline SVG polygon preview (normalized coords); Add/Edit/Delete dialog with vertex list editor
- `frontend/src/views/admin/CTSAdjacencyView.vue` — Camera adjacency graph editor: from/to pairs with min/max transit window; inline validation; save pushes full graph to orchestrator
- `frontend/src/router/index.js` — Added four CTS child routes under `/admin`: `cts/cameras`, `cts/calibration`, `cts/privacy`, `cts/adjacency`
- `frontend/src/views/AdminView.vue` — Added "Tracking (CTS)" nav subheader with four list items

Key implementation notes:

- All CTS router handlers check `_cts_enabled()` first; returns 404 + `{"code": "cts.disabled"}` when off
- `compute_homography()` in `cts_calibration.py` is a pure module-level function; raises `ValueError` for fewer than 4 point pairs
- Snapshot endpoint proxies raw JPEG bytes from ingress; frontend creates a blob URL and revokes it on close
- Router tests override `get_auth_context` (not `require_permission`); use `StaticPool` for in-memory SQLite; call `register_exception_handlers(app)` on every test app instance

**DoD verified**: All 29 new router tests pass. `make check` passes cleanly.

## When Working with This Repo

- **M1–M7 are implemented (committed).** M8 is partially complete — see status below. Remaining milestones build on the scaffolding in `tracking-orchestrator/`, `rtsp-ingress/`, `proto/`, `triton-models/`, and `cognitive-companion/`.
- **Always reference phase-0 first.** It supersedes phases 1-5 where they conflict.
- **The `cognitive-companion` project** is a dependent system. Its CLAUDE.md and README.md are required reading before starting implementation (per phase-0 section 0.27).
- **Validation gates** (phase-0 section 0.31) define binary pass/fail criteria for each milestone. Each PR that adds code must satisfy the relevant gates.

## M8 — Implemented. Dementia signals, dashboard, and keyframes.

### tracking-orchestrator

- `tracking-orchestrator/migrations/0003_m8_signals.sql` — `dementia_signals` TimescaleDB hypertable (identity_id, signal_kind, severity, value, baseline, z_score, window_start/end, context_json, emitted_at). Retention policy (365 days). Continuous aggregate `dementia_signals_daily` for baseline computation.
- `tracking-orchestrator/app/domain/__init__.py` — Added M8 types: `DementiaSignal`, `DementiaSignalKind`, `DementiaSignalSeverity`.
- `tracking-orchestrator/app/storage/base.py` — Added `DementiaSignalRepository` protocol and `InMemoryDementiaSignalRepository` implementation.
- `tracking-orchestrator/app/storage/postgres/signal_repo.py` — `PostgresDementiaSignalRepository`: asyncpg impl with `$N` placeholders, upsert on conflict, filtered list query.
- `tracking-orchestrator/app/trajectory/dementia_signals.py` — `DementiaSignalWorker` with 6 signal detectors: pacing, sundowning_index, bathroom_dwell_anomaly, nighttime_movement, stillness_anomaly, absence. `SignalConfig` for threshold tuning. All detectors sort trajectory windows ascending before comparing consecutive points.
- `tracking-orchestrator/app/transport/signal_publisher.py` — `SignalPublisher`: publishes `DementiaSignal` messages to the `tracking.signals` Redis Stream.
- `tracking-orchestrator/app/routers/dashboard.py` — 6 internal endpoints: `GET /internal/dashboard/signals`, `GET /internal/dashboard/trajectory`, `GET /internal/dashboard/dwell_summary`, `GET /internal/keyframes`, `GET /internal/keyframes/{sample_id}`, `POST /internal/keyframes/{sample_id}/retain`.
- `tracking-orchestrator/app/main.py` — Dashboard router wired in.
- `tracking-orchestrator/pyproject.toml` — Added `httpx>=0.27` (test dep), `B008` added to ruff ignore list (FastAPI Depends pattern).
- `tracking-orchestrator/tests/test_dementia_signals.py` — 18 unit tests covering all 6 detectors with fixture trajectories/dwells.

**DoD verified (tracking-orchestrator)**: ruff clean, mypy (40 files, 0 errors), import-lint, pytest (211/211 tests across 15 files).

**Post-M8 verification fixes** (found during M1–M8 correctness review — all fixed):

- `migrations/0003_m8_signals.sql`: Added `CREATE SCHEMA IF NOT EXISTS continuous_tracking;` before `CREATE TABLE` — schema must exist before schema-qualified table creation. Also fixed `uuid_generate_v4()` → `gen_random_uuid()` (consistent with 0001/0002; avoids requiring uuid-ossp extension).
- `app/transport/signal_publisher.py`: Fixed mypy return-type errors — `xadd` returns `Any`; explicit `str()` casts added; `publish_batch` return changed from `# type: ignore` to `[str(mid) for mid in message_ids]`; `_serialize` annotated as `dict[str, object]`.
- `app/routers/calibration.py`: Fixed RUF001 ambiguous Unicode `×` in Field description and validator message; deprecated `HTTP_422_UNPROCESSABLE_ENTITY` → `HTTP_422_UNPROCESSABLE_CONTENT` (Starlette deprecation becomes a test failure with `filterwarnings = ["error::DeprecationWarning"]`); added missing `Any` import.
- `app/calibration/state.py`: Fixed RUF003 ambiguous `×` in comment; C416 unnecessary dict comprehension removed.
- `tests/test_calibration_router.py`: Replaced `create_app()` fixture (triggered full FastAPI lifespan → Redis connection attempt → `ConnectionRefusedError`) with a minimal `FastAPI()` app wrapping only the calibration router; added `monkeypatch` to inject a fresh `CalibrationState` per test for isolation.
- `app/storage/postgres/gallery_repo.py`: **Critical runtime bug** — `search_similar()` passed 6 arguments to `_SQL_SEARCH_SIMILAR` which has 7 placeholders (`$5` = IS NULL guard, `$6` = interval seconds, `$7` = LIMIT). asyncpg would raise `ValueError: bind parameter $7 not found` at runtime. Fixed by passing `max_age_seconds` twice and `limit` as `$7`.
- `app/trajectory/dementia_signals.py` + `app/storage/postgres/signal_repo.py`: **Idempotency design flaw** — `signal_id=str(uuid.uuid4())` generated a fresh UUID every emission, so `ON CONFLICT (signal_id, emitted_at)` could never trigger (upsert was always a plain INSERT). Fixed by adding `_stable_signal_id()` helper using `uuid.uuid5(NAMESPACE_URL, "{identity_id}\x00{signal_kind}\x00{window_start}\x00{window_end}")` — the same window always hashes to the same UUID — and setting `emitted_at=window_end` explicitly at all 6 construction sites so both conflict-key columns are deterministic on retry.

### cognitive-companion (BFF gateway)

- `backend/models/cts_signal.py` — `DementiaSignal` SQLAlchemy ORM model.
- `backend/services/cts/signal_store.py` — `SignalStore`: async persistence and read API. Bug fixed: `with self._db_factory() as db:` → `db = self._db_factory(); try/finally db.close()`. Bug fixed: `func.case()` → standalone `case()` (SQLAlchemy 2.x).
- `backend/services/cts/subscriber.py` — `DementiaSignalSubscriber`: Redis Streams consumer for `tracking.signals`.
- `backend/services/cts/stream_consumer.py` — `StreamConsumer` base class.
- `backend/filters/builtin/dementia_signal.py` — `DementiaSignalFilter`: rule-engine context filter.
- `backend/routers/cts_signals.py` — 5 endpoints. Bug fixed: `get_db` → `get_session` in `_get_signal_store`.
- `backend/routers/cts_keyframes.py` — 3 endpoints. Bug fixed: catches `UpstreamError` not just `HTTPException`.
- `backend/routers/cts_dashboard.py` — 3 proxy endpoints: `GET /cts/dashboard/signals`, `GET /cts/dashboard/trajectory`, `GET /cts/dashboard/dwell_summary`.
- `backend/integrations/tracking_orchestrator_client.py` — Added `list_keyframes`, `get_keyframe`, `retain_keyframe`, `get_dashboard_signals`, `get_dashboard_trajectory`, `get_dashboard_dwell_summary`.
- `backend/main.py` — `DementiaSignalSubscriber` wired into CTS startup/shutdown. `cts_dashboard` router included.
- `backend/tests/services/test_signal_store.py` — 20 unit tests.
- `backend/tests/services/test_dementia_signal_subscriber.py` — 10 unit tests.
- `backend/tests/filters/test_dementia_signal_filter.py` — 20 unit tests.
- `backend/tests/routers/test_cts_signals.py` — 14 router tests.
- `backend/tests/routers/test_cts_keyframes.py` — 11 router tests.
- `frontend/src/views/admin/CTSDashboardView.vue` — Per-person signal timeline, floor-plan trajectory SVG overlay, room dwell bar chart. Wired to `/cts/dashboard/*` proxy endpoints.
- `frontend/src/services/cts.js` — Added `getDashboardSignals`, `getDashboardTrajectory`, `getDashboardDwellSummary`.
- `backend/pyproject.toml` — Added `redis[hiredis]>=5.0`.

**DoD verified (cognitive-companion)**: ruff clean, pytest (852/852 tests).

## When Working with This Repo

- **M1–M8 are implemented and fully verified.** All correctness bugs identified during the M1–M8 review pass have been fixed. The Python quality gate (`make check`) passes cleanly at 211/211 tests with zero ruff, mypy, or import-linter errors. Remaining milestones build on the scaffolding in `tracking-orchestrator/`, `rtsp-ingress/`, `proto/`, `triton-models/`, and `cognitive-companion/`.
- **Always reference phase-0 first.** It supersedes phases 1-5 where they conflict.
- **The `cognitive-companion` project** is a dependent system. Its CLAUDE.md and README.md are required reading before starting implementation (per phase-0 section 0.27).
- **Validation gates** (phase-0 section 0.31) define binary pass/fail criteria for each milestone. Each PR that adds code must satisfy the relevant gates.

## Next Milestone: M9 (Live View, Identity Corrections, Runtime Integration)

M9 ties the full system together under a single runtime and gives caregivers direct control over identity. It depends on M8 being complete.

**Scope:**

- `CTSRuntime` lifecycle manager in CC backend — owns all three subscribers (tracking events, identity revisions, dementia signals).
- `TrackingEventSubscriber` and `IdentityRevisionSubscriber` — consume `tracking.events` and `tracking.revisions` streams; write to `PersonLocationState`/`PersonLocationHistory`; apply identity rewrites.
- `LocationWriter`, `SourceAuthority`, `IdentityRewriter` services in CC.
- Pipeline step `tracking_query` — inject CTS identity and location context into rule pipelines.
- MCP tools: `get_tracking_status`, `get_person_location`, `get_recent_dementia_signals`.
- CC routers: `cts_live.py` (WebSocket `/ws/cts`), `cts_identity.py` (corrections, merges).
- Vue views: `CTSLiveView.vue`, `CTSIdentityCorrectionsView.vue`; extensions to `PersonsView.vue` and `PersonTimelineView.vue`.
- Manual identity override end-to-end (UI click → `OrchestratorClient.manual_identity_override` → `IdentityRevision` back to CC → DB rewrite → WS toast).

**DoD gate (from phase-0 Appendix F):**

- Toggling `cts.enabled=true` starts the runtime and all three subscribers without any other config change.
- `tests/e2e/test_cts_pipeline.py` passes in CI nightly: feed a capture, assert rule fires, assert UI reflects the identity correction.
- A caregiver can open the Live view, watch bboxes labeled with identities, click one, issue a manual override, and see the dashboard update within 2 seconds.
