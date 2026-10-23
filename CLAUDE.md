# CLAUDE.md

Guidance for Claude Code agents working in this repository.

## Skills to load

| Skill | Use when |
| --- | --- |
| `.claude/skills/engineering-standards/SKILL.md` | CTS architecture, repositories, configuration, tests, wire formats, units, caches, security boundaries, logging, and naming |
| `.claude/skills/cts-pipeline/SKILL.md` | Frame stages, execution paths, tracker-round side effects, batching, StageRunner/FrameContext, and replay fixture formats |
| `.claude/skills/cts-signals/SKILL.md` | Dementia signals, detectors, baselines, robust z-scores, hysteresis, severity, protobuf/CC kind plumbing, and signal tests |
| `.claude/skills/cts-spatial-fusion/SKILL.md` | Floor-plane localization, homography Jacobian, measurement covariance, cross-camera fusion, Kalman/ZUPT, primary camera selection, and posture geometry weighting |
| `.claude/skills/cts-identity-governance/SKILL.md` | PH identity evidence, the identity resolver, authority order, ReID gallery lifecycle, revisions, operator corrections, external identity evidence, and identity provenance |
| `/home/sriram/code/nanai/.claude/skills/nanai-workspace/SKILL.md` | Any change spanning more than one repository: repo map, gates, proto two-repo rule, milestone workflow |

`engineering-standards` lists its rules as one-liners and keeps the detail in `reference/` beside
it (approved libraries, SQL rules, migrations, Go, domain principles). Cross-cutting rules
(markdown, comments, time, the Python environment, migrations, SQL) are in `.claude/rules/` and
load for matching files.

## What This System Does

The **Continuous Tracking System (CTS)** monitors seniors with early dementia via RTSP cameras. It:

- Pulls camera streams via a go2rtc sidecar and uploads JPEG keyframes to MinIO
- Tracks individuals using a single floor-plane Kalman world tracker (PersonHypothesis entities, Hungarian association over calibrated floor positions)
- Re-identifies people across cameras using a Bayesian posterior over identity candidates (ArcFace face recognition + SOLIDER-REID body appearance)
- Detects dementia-relevant behavioural patterns: pacing, sundowning, bathroom anomaly, prolonged stillness, nighttime movement, unexplained absence
- Streams results via Redis Streams (protobuf wire format) to `cognitive-companion`, a BFF gateway that serves the Vue admin UI

**Infrastructure**: TimescaleDB + pgvectorscale (StreamingDiskANN) · Redis Streams (AOF) · MinIO · Triton Inference Server. **GPU support**: NVIDIA only (TensorRT/CUDA execution provider via ONNX Runtime). **Shared libraries**: `triton-shared/` (sibling) provides the Triton gRPC client (`TritonClientProtocol`, `TritonGrpcClient`) and inference pre/post-processing used by both CTS and `scene-analysis-service`.

## Design Documents

The canonical reference for architecture decisions is this file and the code. The architecture
reference is [docs/systems-architecture.md](docs/systems-architecture.md): the service diagram,
the PersonHypothesis world-tracker model and its 10-step loop, the frame pipeline, the identity
resolver and how priors update, home-camera nuances, rtsp-ingress and camera configuration, the
settings coupling rule, and the CTS-to-CC wire contract. The CC-side companion doc is
[`../cognitive-companion/docs/systems-architecture.md`](../cognitive-companion/docs/systems-architecture.md).

## Repository Layout

```text
.
├── rtsp-ingress/                  Go RTSP ingest service (cmd/server, internal/{config,go2rtc,imageops,
│                                  media,metrics,motion,poll,reconciler,supervisor})
├── tracking-orchestrator/         Python ML orchestration service
│   ├── app/domain/                Frozen dataclasses (Detection, WorldObservation, PersonHypothesis, …)
│   ├── app/inference/             Triton gRPC client (delegates to triton_shared) + ReID/Pose wrappers
│   ├── app/tracking/              world tracker (world/: Kalman, association, cost matrix, dedup), identity resolver
│   ├── app/trajectory/            Trajectory writer, dementia signal detectors, posture classifier, motion energy, robust stats
│   ├── app/transport/             Redis Streams codec (protobuf), publishers
│   ├── app/services/              Coordinating services (config sync, identity rewriter, ...)
│   ├── app/storage/               Repository protocols, InMemory impls, Postgres impls, MigrationRunner
│   ├── app/routers/               Internal FastAPI endpoints (dashboard, live, calibration, gallery, corrections, trajectory)
│   ├── app/pipeline/              Frame processing pipeline (detection, tracking, pose, ReID, privacy enforcement)
│   ├── app/calibration/, app/sampling/, app/observability/, app/config/
│   ├── app/cli.py                 `cts-db` migration CLI (migrate / rollback / status)
│   ├── app/proto/                 Generated protobuf Python bindings (committed)
│   └── migrations/                SQL migrations (0001_init baseline; numbered .up.sql/.down.sql pairs)
├── triton-models/                 Triton model configs + export/download scripts (triton-models-jetson/ for Jetson)
│                                  one directory per model (person-detector YOLO26L, reid-solider, pose-rtmpose,
│                                  face-*, depth-anything-v2, clip-vision, florence-2*); see its README.md
├── proto/                         Protobuf contracts (frame, tracking, signals, scene)
├── calibration-data/, scripts/, tools/
└── docs/                          Architecture, runbook, wire-format spec
```

Siblings: `../cognitive-companion/` (BFF, its own CLAUDE.md), `../triton-shared/`,
`../kubernetes/continuous-tracking/` (K8s manifests).

## Non-negotiable invariants

- Layering `core → domain → storage → tracking.world → services → transport → routers`, enforced by import-linter.
  Nothing above `storage` imports a concrete `Postgres*` class; services depend on `Protocol`s.
- Every persistent resource is a `Protocol` + `InMemory*` + `Postgres*` triplet with identical
  filters, wired explicitly in `app/main.py`.
- Domain objects are frozen dataclasses; change them with `dataclasses.replace`.
- SQL uses asyncpg `$N` params and schema-qualifies every table (`continuous_tracking.<table>`).
- Schema changes are new numbered migration pairs on top of `0001_init`; never edit an applied
  migration or drop and recreate the database.
- Redis Streams carry raw protobuf bytes through `app/transport/codec.py`; clients set
  `decode_responses=False`. Proto changes are two-repo changes (see Proto Codegen).
- Timestamps are timezone-aware (`datetime.now(UTC)`, Go `time.Now().UTC()`).
- An identity commit needs sensory (face or ReID) evidence; the temporal prior alone never
  commits. Overturned identities publish an `IdentityRevision` on `tracking.revisions`.
- Use the project venv at `tracking-orchestrator/.venv/`, never system Python. Markdown has no
  em-dashes.

## Claude Code guardrails

Start sessions in this directory (or `/cd` here): `.claude/settings.json` applies only from the
session's working directory. It sets `UV_FROZEN=1`, skips the vendored `CLAUDE.md` files under
`tools/go-mod-cache/`, and runs four hooks from `.claude/hooks/`:

- **Guard:** blocks `pytest` outside `tracking-orchestrator/.venv`.
- **Post-edit:** an em-dash in an edited `.md` file, a `ruff check` failure in
  `tracking-orchestrator/`, or a `gofmt` finding in `rtsp-ingress/` comes back as feedback.
- **Stop:** flags added em-dashes and `uv.lock` churn. It blocks once and never loops.
- **Session start:** branch, changed files, gates.

Machine-wide guards live in `~/.claude/`. Run any blocked command yourself with `!` if it is
intended. `make hooks-test` (part of `make check` and CI) runs the hook test table.

## Quality Gate

Run before every PR. CI runs the same checks.

```bash
make check        # Python: ruff check + ruff format --check + mypy + import-linter + pytest (fast local gate)
make check-all    # Python + Go (golangci-lint + go test -race + go build) + buf lint
make ci           # Authoritative gate: check-all + every integration-marked test (requires Docker)
make install-hooks  # Install the git pre-commit hooks (once per clone; includes the em-dash check)
make check-md       # Markdown has no em-dashes (also in make check and CI)
make docs-check     # Instruction files name only files that exist (also in make check)
```

**`make ci` is the authoritative gate.** `make check` is the fast local gate; `make check-all` adds Go and proto validation. `make ci` additionally runs `make test-integration` (`pytest -m integration tests`), which proves C1/C2 (WorldTracker end-to-end) and T5 (PH repository parity) against a testcontainer Postgres. These proofs must pass before merging any change to the tracker or storage layer. CI's `python-test` job runs the same two targets, `make test` and `make test-integration`, and `tests/contracts/test_ci_workflow_paths.py` fails if the workflow names a file that does not exist.

The pre-commit hooks (`make install-hooks`: a pinned `uv tool install pre-commit`, then `pre-commit install`) run ruff, ruff format, import-linter, and mypy from the project venv, golangci-lint through `make go-lint`, and buf through the pinned toolchain, so every hook uses the same tool versions as `make check` and CI. No test reads `DATABASE_URL`; integration tests use the session testcontainer.

**Always use the project venv** at `tracking-orchestrator/.venv/`. The Makefile uses it automatically. Sync dependencies with:

```bash
cd tracking-orchestrator && uv sync --frozen --extra dev
```

## Proto Codegen

Generated bindings are **committed to the repo** -- do not gitignore them. Regenerate after changing `.proto` files:

```bash
make proto          # Go (buf generate) + Python (protoc --python_out --pyi_out)
make proto-lint     # buf lint only
```

Requires `protoc >= 25` (`apt install protobuf-compiler`) and the project Go toolchain (`make go-install`). The `proto-py` target writes to both `tracking-orchestrator/app/proto/` and `../cognitive-companion/backend/integrations/proto/`.

## Known Tech Debt

| ID | Severity | Description |
| --- | --- | --- |
| TD-003 | Medium | Revision stream consumer group pre-created by publisher instead of admin tooling |
| TD-007 | Low | Validated by [`docs/identity-path-validation-2026-06.md`](docs/identity-path-validation-2026-06.md): CC face-based location conclusions are redundant when `cts.enabled=true`, but remain the non-CTS reCamera path and the owner of raw sightings/activity capture. Deprecate its identity/location writes behind an explicit CTS-aware boundary; keep non-CTS and non-location responsibilities until their consumers migrate. |
| TD-008 | Medium | DepthEstimator (`app/inference/depth.py`) is implemented and importable but not wired into the live frame pipeline. It is used only during homography auto-calibration. M7 (Occluded Posture Detection) will wire it into the posture slow-path. Do not connect it to `FrameProcessingPipeline` before M7 is complete. |
| TD-009 | Low | ReID crop invariant: `_crop_detection()` in `frame_pipeline.py` must always be called before `ReidEmbedder.embed_batch()`. Any refactor that moves or merges these calls must preserve this ordering or the gallery distances become meaningless. |
| TD-010 | Low | Floor-region visibility fix (Track G): harvest a floor-region polygon from `FloorPlaneFitter` inliers and return it on `/internal/calibration/auto`; CC then projects that boundary (not the image border) for the visibility polygon. Reuses Depth Anything v2 (no new model); non-blocking. Plan: `/home/sriram/code/nanai/notes/cc/guided-companion/guided-companion-trackG-floor-geometry.md`. |

The resolved 2026-07 hardening findings (F1 to F14) and their evidence live in
`/home/sriram/code/nanai/notes/guided-companion-and-identity-hardening/`.

## Working in This Repository

- **cognitive-companion** is a dependent system in a sibling directory (`../cognitive-companion`). Its own CLAUDE.md is required reading before touching anything under `cognitive-companion/`.
- **scene-analysis-service** shares the Triton instance and the `triton-shared/` client library. Changes to `triton-shared/` affect both CTS and SAS.
- **Model binaries are not in git.** Run the export/download scripts and place outputs in `triton-models/<model>/1/`. See `triton-models/README.md`. Models run as FP32 ONNX; no quantization.
- **Triton inference code** in `app/inference/` delegates to `triton_shared.client` and `triton_shared.inference`. CTS-specific wrappers (`PersonDetector`, `ReidEmbedder`, `PoseEstimator`) remain in `app/inference/`; shared logic lives in `triton-shared/`.
- **Never implement a feature that depends on another in-flight PR.** The layering is load-bearing.
- **Validation gates** (section 0.31, "Validation Matrix", of `/home/sriram/code/nanai/notes/continuous-tracking/phase-0-design-review.md`) define binary pass/fail criteria for each subsystem. Any PR touching a subsystem must satisfy the relevant gates.
