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
| M1 | 1-2 | Protobuf contracts, repository interfaces, docker-compose, CI |
| M2 | 3-4 | rtsp-ingress (Go) |
| M3 | 5-6 | Triton models + benchmark harness |
| M4 | 7-8 | Tracking orchestrator skeleton (per-camera tracking, tracklets) |
| M5 | 9-10 | Identity resolution, retroactive revision |
| M6 | 11 | Trajectories, dwells, keyframes |
| M7 | 12-13 | Admin UI (cameras, calibration, privacy, adjacency) |
| M8 | 14-15 | Dementia signals, dashboard, keyframes |
| M9 | 16 | Live view, identity corrections, runtime integration |
| M10 | 16+ | K8s, observability, docs, performance hardening |

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

## When Working with This Repo

- **M1, M2, M3, and M4 are implemented.** Remaining milestones build on the scaffolding in `tracking-orchestrator/`, `rtsp-ingress/`, `proto/`, and `triton-models/`.
- **Next milestone: M5** (weeks 9-10) — Identity resolution, retroactive revision. See `phase-3-tracking-reid.md` and `phase-5-backend-integration.md` for scope.
- **Always reference phase-0 first.** It supersedes phases 1-5 where they conflict.
- **The `cognitive-companion` project** is a dependent system. Its CLAUDE.md and README.md are required reading before starting implementation (per phase-0 section 0.27).
- **Validation gates** (phase-0 section 0.31) define binary pass/fail criteria for each milestone. Each PR that adds code must satisfy the relevant gates.
