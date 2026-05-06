# CLAUDE.md

Guidance for Claude Code agents working in this repository.

---

## What This System Does

The **Continuous Tracking System (CTS)** monitors seniors with early dementia via RTSP cameras. It:

- Pulls camera streams via a go2rtc sidecar and uploads JPEG keyframes to MinIO
- Tracks individuals frame-to-frame with BoT-SORT (Kalman filter + appearance embeddings)
- Re-identifies people across cameras using a Bayesian posterior over identity candidates (ArcFace face recognition + SOLIDER-REID body appearance)
- Detects dementia-relevant behavioural patterns: pacing, sundowning, bathroom anomaly, prolonged stillness, nighttime movement, unexplained absence
- Streams results via Redis Streams (protobuf wire format) to `cognitive-companion`, a BFF gateway that serves the Vue admin UI

---

## Design Documents

Read the phase documents before making architectural decisions. Always reference phase-0 first — it supersedes phases 1–5 where they conflict.

| File | What it covers |
| --- | --- |
| `phase-0-design-review.md` | **Authoritative foundation.** Identity model, runtime partitioning, storage abstraction, message transport, dementia activity layer, schema. |
| `phase-1-architecture.md` | System architecture, identity model, database schema, layering rules. |
| `phase-2-rtsp-ingestion.md` | Go service for RTSP ingest, frame decode, motion gating, MinIO upload. |
| `phase-3-tracking-reid.md` | Tracking orchestrator, BoT-SORT, cross-camera association, Bayesian identity resolution. |
| `phase-4-scene-semantic.md` | Scene analysis, semantic memory integration, VLM pipeline. |
| `phase-5-backend-integration.md` | Cognitive companion integration, backend routers, Vue views, gateway contract. |

---

## System Architecture

```text
IP Cameras (RTSP)
       │
       ▼
 go2rtc (sidecar, port 1984)
  ┌──────────────┐    HTTP /api/frame.jpeg
  │  RTSP proxy  │◀──────────────────────────  rtsp-ingress (Go, port 8090)
  │  multiplexer │                             ┌────────────────────────┐
  └──────────────┘                             │  Motion gating         │
                                               │  MinIO JPEG upload     │
                                               │  frames.ready publish  │
                                               └────────────────────────┘
                                                         │ frames.ready (protobuf)
                                                         ▼
                                           tracking-orchestrator (Python, port 8000)
                                           ┌──────────────────────────────┐
                                           │ YOLO26L  person detection    │
                                           │ SOLIDER-REID body embeds     │──▶ Triton (gRPC 8001)
                                           │ RTMPose  pose estimation     │
                                           │ BoT-SORT tracker             │
                                           │ Bayesian identity resolver   │
                                           │ Dementia signal worker       │
                                           └──────────────────────────────┘
                                                         │ tracking.events
                                                         │ tracking.revisions
                                                         │ tracking.signals
                                                         ▼
                                            cognitive-companion (Python/FastAPI, port 8080)
                                            BFF gateway · WebSocket live view
                                            Vue 3 admin UI · MCP tools
```

**Infrastructure**: TimescaleDB + pgvector · Redis Streams (AOF) · MinIO · Triton Inference Server

**GPU support**: NVIDIA (TensorRT execution provider) and Intel Arc (OpenVINO execution provider). Switch with `python triton-models/scripts/configure_gpu.py --vendor nvidia|intel`.

---

## Repository Layout

```text
.
├── rtsp-ingress/                  Go RTSP ingest service
│   ├── cmd/server/                Entry point
│   ├── internal/config/           Config loading (service settings from YAML + env vars)
│   ├── internal/go2rtc/           go2rtc HTTP API client
│   ├── internal/motion/           Motion gating (pixel diff)
│   ├── internal/media/            MinIO upload + Redis Streams publish
│   ├── internal/metrics/          Prometheus metrics
│   ├── internal/poll/             Camera config polling worker
│   ├── internal/reconciler/       Camera state reconciliation
│   ├── internal/streams/          Stream lifecycle management
│   └── internal/supervisor/       Runtime supervisor
├── tracking-orchestrator/         Python ML orchestration service
│   ├── app/domain/                Frozen dataclasses (Detection, Tracklet, GlobalTrack, …)
│   ├── app/inference/             Triton gRPC client + YOLO26L/ReID/Pose wrappers
│   ├── app/tracking/              BoT-SORT, identity resolver, cross-camera association
│   ├── app/trajectory/            Trajectory writer + dementia signal detectors
│   ├── app/transport/             Redis Streams codec (protobuf), publishers
│   ├── app/storage/               Repository protocols, InMemory impls, Postgres impls
│   ├── app/routers/               Internal FastAPI endpoints
│   ├── app/observability/         Prometheus metrics
│   ├── app/calibration/           Homography calibration state
│   ├── app/sampling/              Keyframe sampler
│   ├── app/pipeline/              Frame processing pipeline wiring
│   ├── app/proto/                 Generated protobuf Python bindings (committed)
│   └── migrations/                SQL migrations (0001–0005)
├── proto/                         Protobuf contracts (frame, tracking, signals, scene)
├── triton-models/                 Triton model configs + export scripts
├── cognitive-companion/           BFF gateway (sibling repo — see its own CLAUDE.md)
├── k8s/                           Kubernetes manifests
└── docs/                          Runbook, wire-format spec
```

---

## Code Architecture Patterns

### Layering (strictly enforced by import-linter)

```text
core → domain → storage → services → transport → routers
```

Nothing above `storage` may import a concrete `Postgres*` class. Services depend only on `Protocol` interfaces. Import-linter runs at CI and in pre-commit hooks.

### Storage: Protocol + InMemory + Postgres triplet

Every persistent resource in `tracking-orchestrator` has three artifacts:

1. **`Protocol`** in `storage/base.py` — the contract services depend on
2. **`InMemory*`** in the same file — zero-dependency, used in all unit tests
3. **`Postgres*`** in `storage/postgres/` — production asyncpg implementation

```python
# storage/base.py
class TrackletRepository(Protocol):
    async def save_tracklet(self, tracklet: Tracklet) -> None: ...
    async def get_tracklet(self, tracklet_id: str) -> Tracklet | None: ...
```

Rules:

- All repository methods are `async`.
- Return domain types, never raw DB rows.
- `Protocol` carries no state.
- `InMemory` uses plain `dict`/`list`; never touches a DB.
- `Postgres*` receives only an `asyncpg.Pool`; holds no other state.

### Domain objects: frozen dataclasses

Internal domain objects are `@dataclass(frozen=True)`. They carry no validation logic — validation happens at service boundaries via Pydantic v2. Never mutate a frozen dataclass; use a side-channel dict keyed by `id(obj)` for transport metadata (e.g., Redis message IDs).

### Go: interface contracts

Each subsystem in `rtsp-ingress/internal/` exports an interface as its public API. Tests inject a fake; production wires the real implementation. Define interfaces in the **consumer** package, not the provider.

### RTSP ingest: go2rtc sidecar

`rtsp-ingress` does **not** manage RTSP sessions directly. go2rtc owns all RTSP sessions. The flow per camera:

1. `reconciler` fetches the enabled camera list from `GET /api/v1/cts/cameras` on cognitive-companion (polled every 60 s)
2. `Supervisor.Reconcile()` calls `go2rtc.RegisterStream(ctx, cameraID, rtspURL)` — idempotent HTTP PUT to go2rtc `/api/streams`; heals go2rtc restarts
3. `poll.Worker` ticks every `frame_interval_ms` ms and calls `go2rtc.FetchJPEG(ctx, cameraID)` — HTTP GET `/api/frame.jpeg`
4. Frame passes motion gate, uploads to MinIO, publishes to `frames.ready`

`rtsp-ingress/config/go2rtc.yaml` has **no `streams:` section** — all registrations are dynamic. Do not add `gortsplib` or `pion/rtp` to this codebase.

### Identity resolution: Bayesian posterior

The identity resolver maintains a posterior probability distribution over `{known_identities ∪ UNKNOWN}` for each tracklet. Evidence sources:

- **Face anchors** (ArcFace similarity → likelihood update)
- **ReID gallery** (SOLIDER-REID embedding → pgvector HNSW nearest-neighbour search)
- **Temporal prior** (continuity from prior frame's assignment)

A commit fires only when `prob ≥ threshold AND margin ≥ min_margin AND sensory_evidence_present`. The temporal prior alone cannot trigger a commit — face or ReID evidence must be present. Parameter: `commit_prob = 0.65`.

Retroactive revision: when a committed identity is later overturned, an `IdentityRevision` is published to `tracking.revisions`, the `PersonLocationHistory` row in cognitive-companion is soft-deleted (`superseded_by_revision_id` stamped), and a replacement row is inserted.

---

## Approved Libraries

### Python (`tracking-orchestrator/pyproject.toml`)

| Library | Role |
| --- | --- |
| `fastapi` | HTTP router and dependency injection |
| `pydantic v2` | Validation at all external boundaries (HTTP, config, Redis payloads) |
| `asyncpg` | Async Postgres driver — `$1..$N` positional params always |
| `redis[hiredis]` | Async Redis Streams client — consumer groups + XACK |
| `aiobotocore` | Async S3-compatible client for MinIO |
| `structlog` | Structured JSON logging — never `logging.getLogger` or `print` |
| `numpy` | Frame data and embedding arithmetic |
| `opencv-python-headless` | Image preprocessing — `cv2.resize(INTER_LINEAR)` for all letterbox/crop resizing |
| `scipy` | Hungarian assignment (`linear_sum_assignment`) for tracking |
| `protobuf` | Message contracts; generated bindings committed in `app/proto/` |
| `prometheus-client` | Metrics exposition at `GET /metrics` |
| `pytest` + `pytest-asyncio` | Test runner; all repo tests are `async def` |
| `mypy` (strict) | Type checking on `domain/`, `storage/`, `services/`, `transport/` |
| `ruff` | Lint + format |
| `import-linter` | Layering enforcement |

Do not introduce `aiohttp`, `celery`, or `psycopg2`. `httpx` is permitted as a **dev-only** dependency.

### Go (`rtsp-ingress/go.mod`)

| Library | Role |
| --- | --- |
| `minio/minio-go/v7` | S3-compatible object storage client |
| `redis/go-redis/v9` | Redis Streams producer |
| `prometheus/client_golang` | Metrics exposition |
| `go.uber.org/zap` | Structured logging — `zap.L()` for global logger |
| `google.golang.org/protobuf` | Protobuf runtime |
| `golang.org/x/image` | Image format decoding in media processing |
| `gopkg.in/yaml.v3` | Config file parsing |

Do not add `bluenviron/gortsplib` or `pion/rtp` — RTSP is delegated to go2rtc.

---

## Redis Stream Wire Format

All streams carry raw protobuf bytes — no JSON, no base64. Use `transport/codec.py`:

```python
from app.transport.codec import encode, decode

# publish
await redis.xadd("tracking.events", encode(event_msg, field="event"))

# consume
event = decode(fields, TrackingEvent, field="event")
```

| Stream | Field | Message type |
| --- | --- | --- |
| `frames.ready` | `"frame"` | `FrameReady` |
| `tracking.events` | `"event"` | `TrackingEvent` |
| `tracking.revisions` | `"revision"` | `IdentityRevision` |
| `tracking.signals` | `"signal"` | `DementiaSignal` |
| `scene.samples` | `"sample"` | `SceneSample` |

Redis clients must set `decode_responses=False` so binary payloads round-trip unchanged.

---

## Database Schema

Migrations in `tracking-orchestrator/migrations/`:

| File | Contents |
| --- | --- |
| `0001_init.sql` | `tracklets`, `detections` (hypertable), `global_tracks`, `identity_revisions`, `gallery_entries` (pgvector HNSW), `tracking_events` |
| `0002_m6_trajectory_keyframes.sql` | `person_trajectories` (hypertable), `room_dwells`, `tagged_keyframes` |
| `0003_m8_signals.sql` | `dementia_signals` hypertable + continuous aggregate `dementia_signals_daily` |
| `0004_nullable_embedding.sql` | Embedding column nullable |
| `0005_fix_signal_identity_id_type.sql` | Type correction on `dementia_signals.identity_id` |

---

## Coding Rules

These rules come from bugs caught during implementation and are non-negotiable.

### Python / asyncpg

**`$N` placeholders always.** asyncpg uses `$1, $2, …` positional params. Never use `%s` or `?`. `executemany` also requires `$N` style — one row's worth of placeholders per statement.

**`datetime.now(UTC)` always.** `from datetime import UTC`. Bare `datetime.now()` produces timezone-naive objects that break TimescaleDB `timestamptz` columns.

**Declare all attributes in `__init__`.** Every attribute must be present in `__init__` with its correct `Optional[T] | None` type, even if set later. Attributes first assigned outside `__init__` cause `AttributeError` and mypy cannot track them.

**Never mutate frozen dataclasses.** `@dataclass(frozen=True)` raises `FrozenInstanceError` on `instance.attr = value`. For transport metadata (e.g., Redis message IDs) alongside frozen domain objects, maintain a side-channel `dict[int, T]` keyed by `id(obj)`.

**Stable signal IDs.** Dementia signals use `uuid.uuid5(NAMESPACE_URL, "{identity_id}\x00{signal_kind}\x00{window_start}\x00{window_end}")` so the same detection window always maps to the same UUID. This makes the `ON CONFLICT` upsert idempotent on retry.

### SQL correctness

**PostgreSQL array concatenation is `||`.** In `ON CONFLICT DO UPDATE SET`, write `col = EXCLUDED.col || table.col`. The `array[...]` constructor is for array literals only — wrapping a `||` expression in it is a syntax error.

**Anchor conditional SQL replacements.** When adding `AND extra_clause` to a query via `str.replace`, replace a unique substring that includes the insertion point. Never replace a trailing keyword like `LIMIT 100` and prepend `AND …` — that produces invalid SQL (predicate after `ORDER BY`).

**Match bind parameter count exactly.** Count `$1..$N` placeholders and verify the argument tuple has the same arity. Off-by-one raises `ValueError: bind parameter $N not found` at runtime.

### Tracking / data structures

**Use `.items()` when you need both key and value.** Build `active_tracks` with `[(key, val) for key, val in d.items() if …]` and return keys from assignment functions. `enumerate(dict.values())` gives positions that diverge from key names after any deletion.

**Explicit reverse maps for cross-namespace lookups.** When two modules use different ID namespaces for the same entity (e.g., `local_track_id` strings vs. UUID `tracklet_id`s), maintain an explicit `dict[local_id, uuid_id]` reverse map on the owning object.

**One increment per code path.** An unconditional increment followed by a conditional second increment silently doubles the count on the non-threshold branch.

**Prefer method parameters over placeholder fields.** If a helper receives `embedding` as a parameter and the detection also has a placeholder `detection.embedding`, use the parameter — it carries the live Triton value.

**Sort trajectory windows ascending.** Dementia signal detectors need time-ordered points to compute distances and direction changes correctly.

### Go

**go2rtc owns all RTSP sessions.** Use `go2rtc.Client` — `RegisterStream`, `DeregisterStream`, `FetchJPEG`. Never open raw RTSP connections in Go.

**Wrap every error with context.** `fmt.Errorf("streams: register %q: %w", cfg.ID, err)`.

**Check every error.** `go vet` and `golangci-lint` flag unchecked returns — treat them as compile errors.

**`context.Context` first.** Every function that does I/O takes `ctx context.Context` as its first parameter and respects cancellation via `ctx.Err()`.

---

## Engineering Standards

| Area | Standard |
| --- | --- |
| Python version | 3.12; `from __future__ import annotations` in every file |
| Type checking | `mypy --strict` on `domain/`, `storage/`, `services/`, `transport/` |
| Go version | 1.24+ (toolchain pinned in `.tool-versions`; install with `make go-install`) |
| Go safety | `-race` mandatory in all test runs; `go vet`; `golangci-lint` |
| Protobuf | `buf lint`; `buf breaking` against last merged commit; generated bindings committed |
| Logging | `structlog` in Python (`get_logger(__name__)`); `zap.L()` in Go. Include `camera_id`, `tracklet_id`, `global_track_id` in every relevant log line. |
| Timestamps | Always timezone-aware. Python: `datetime.now(UTC)`. Go: `time.Now().UTC()`. |
| Image resize | `opencv-python-headless` `cv2.resize(INTER_LINEAR)`. Never hand-roll bilinear in NumPy. |
| Secrets | In environment variables or `.env` files (never committed). YAML holds defaults; env overrides. |
| SQL injection | asyncpg `$N` params only. Never build SQL with f-strings or `%`. |
| Error taxonomy | Domain errors use the project's `ErrorCode` enum. No stack traces in API responses — log locally, return only code + message. |
| Metrics | Prometheus via `prometheus-client` (Python) and `prometheus/client_golang` (Go). Register in the respective `metrics.py` / `metrics.go` files. |

---

## Testing Strategy

### Python

- Unit tests inject `InMemory*` implementations directly — never mock `asyncpg.Pool` or Redis.
- All tests touching a repository are `async def` (configured with `asyncio_mode = "auto"`).
- Triton calls are mocked via `TritonClientProtocol` — all inference tests run without a GPU.
- Use `pytest.fixture` for shared domain objects; do not construct the same frozen dataclass inline in multiple tests.
- Test only through the public API of the class under test — never assert on private `_attributes`.
- Integration tests (real Postgres + Redis) are in `tests/integration/` and skipped by default.

### Go (rtsp-ingress tests)

- Table-driven tests (`t.Run`) for functions with multiple input/output cases.
- Inject interfaces in constructors — tests replace the real implementation.
- Always run with `-race`: `go test -race ./...`. Hard CI gate.
- HTTP handler tests use `httptest.NewRecorder()`.
- go2rtc calls are tested with `httptest.Server` serving mock responses.

---

## Quality Gate

Run before every PR. CI runs the same checks.

```bash
make check        # Python: ruff check + ruff format --check + mypy + import-linter + pytest
make all-check    # Python + Go (golangci-lint + go test -race + go build) + buf lint
```

Install pre-commit hooks (ruff, mypy, golangci-lint, buf):

```bash
pre-commit install
```

**Always use the project venv** at `tracking-orchestrator/.venv/`. The Makefile uses it automatically. Sync dependencies with:

```bash
cd tracking-orchestrator && uv sync --frozen --extra dev
```

---

## Proto Codegen

Generated bindings are **committed to the repo** — do not gitignore them. Regenerate after changing `.proto` files:

```bash
make proto          # Go (buf generate) + Python (protoc --python_out --pyi_out)
make proto-lint     # buf lint only
```

Requires `protoc >= 25` (`apt install protobuf-compiler`) and the project Go toolchain (`make go-install`). The `proto-py` target writes to both `tracking-orchestrator/app/proto/` and `../cognitive-companion/backend/integrations/proto/`.

---

## Camera Configuration

All cameras are managed through the cognitive-companion Admin UI (`/admin/cts/cameras`). There is no static camera list in `settings.yaml`.

rtsp-ingress polls `GET /api/v1/cts/cameras` on cognitive-companion every 60 s (configurable via `cognitive_companion.reconcile_interval_s`). The reconciler maps each camera record to a `CameraConfig` and calls `Supervisor.Reconcile()`, which registers/deregisters streams with go2rtc idempotently.

`CameraConfig` fields populated from the API response:

| Field | Source |
| --- | --- |
| `ID` | `cts_cameras.id` |
| `RTSPURL` | `cts_cameras.rtsp_url` — full RTSP URL stored in the database |
| `RoomName` | `cts_cameras.location` |
| `Enabled` | only enabled cameras are included (filtered Go-side) |
| `FrameIntervalMs` | falls back to `defaults.frame_interval_ms` from settings.yaml |
| `MotionThreshold` | falls back to `defaults.motion_threshold` |
| `ReconnectBackoffSeconds` | falls back to `defaults.reconnect_backoff_s` |

rtsp-ingress authenticates to cognitive-companion with the `COGNITIVE_API_KEY` environment variable (sent as `X-API-Key`). The key must have the `cts_ingress` permission defined in `cognitive-companion/config/auth.yaml`. The `CC_INGRESS_API_KEY` env var should hold the same secret on both services.

RTSP credentials are stored in the cognitive-companion database as part of the full RTSP URL. They are never written to any config file on the rtsp-ingress side.

The CLAUDE.md for the `rtsp-ingress` architecture note:

```text
RTSP ingest: go2rtc sidecar
  1. reconciler fetches CameraConfig list from cognitive-companion API
  2. Supervisor.Reconcile() calls go2rtc.RegisterStream() — idempotent PUT, heals restarts
  3. poll.Worker ticks every frame_interval_ms and calls go2rtc.FetchJPEG()
  4. Frame passes motion gate, uploads to MinIO, publishes to frames.ready
```

---

## Known Tech Debt

| ID | Severity | Description |
| --- | --- | --- |
| TD-003 | Medium | Revision stream consumer group pre-created by publisher instead of admin tooling |
| TD-004 | Medium | `tracking.revisions`, `tracking.signals`, `scene.samples` not yet migrated to protobuf wire format |

---

## Working in This Repository

- **cognitive-companion** is a dependent system in a sibling directory (`../cognitive-companion`). Its own CLAUDE.md is required reading before touching anything under `cognitive-companion/`.
- **Model binaries are not in git.** Run the export scripts on the target GPU and place outputs in `triton-models/<model>/1/`. See `triton-models/README.md`.
- **Never implement a feature that depends on another in-flight PR.** The layering is load-bearing.
- **Validation gates** (phase-0 §0.31) define binary pass/fail criteria for each subsystem. Any PR touching a subsystem must satisfy the relevant gates.
