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

**M2–M10 — Not yet started.** See Implementation Playbook above for milestone scope.

## When Working with This Repo

- **M1 is implemented.** Remaining milestones build on the scaffolding in `tracking-orchestrator/`, `rtsp-ingress/`, and `proto/`.
- **Always reference phase-0 first.** It supersedes phases 1-5 where they conflict.
- **The `cognitive-companion` project** is a dependent system. Its CLAUDE.md and README.md are required reading before starting implementation (per phase-0 section 0.27).
- **Validation gates** (phase-0 section 0.31) define binary pass/fail criteria for each milestone. Each PR that adds code must satisfy the relevant gates.
