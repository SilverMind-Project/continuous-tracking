---
name: engineering-standards
description: This skill covers **how** to build - library choices, data abstraction patterns, testing strategy, wire format, and security at boundaries. For **what** to build (milestones, domain rules, quality gate, linting configuration), see [CLAUDE.md](CLAUDE.md).
---

# Engineering Standards


## Data Abstraction Layer

The goal is backend portability: swap Postgres for a different store by replacing one class, not
touching services.

### Python: Protocol + InMemory + Postgres triplet

Every persistent resource gets three artifacts:

1. A `Protocol` in `storage/base.py` — the contract services depend on.
2. An `InMemory*` class in the same file — fast, zero-dependency, used in unit tests.
3. A `Postgres*` class in `storage/postgres/` — the production implementation.

Services import only the `Protocol`. Import-linter enforces the layering contract
(`core → domain → storage → services → transport → routers`). Nothing above `storage` may
import a concrete `Postgres*` class.

```python
# storage/base.py
class TrackletRepository(Protocol):
    async def save_tracklet(self, tracklet: Tracklet) -> None: ...
    async def get_tracklet(self, tracklet_id: str) -> Tracklet | None: ...
```

Rules:

1. All repository methods are `async`.
2. Return domain types, never raw DB rows.
3. `Protocol` carries no state — it is a pure structural interface.
4. `InMemory` implementations use plain `dict` / `list`; they never touch a DB.
5. `Postgres*` classes receive only an `asyncpg.Pool`; they hold no other state.

### Go: interface contracts in `internal/`

Each subsystem in `rtsp-ingress/internal/` exports an interface, not a concrete struct, as its
public API. Tests inject a fake via the interface; production wires the real implementation.

```go
// streams/streams.go
type Manager interface {
    Register(ctx context.Context, cfg StreamConfig) error
    Stop(streamID string) error
}
```

Rules:

1. Accept interfaces, return concrete types (Go convention).
2. Never pass raw Redis clients across package boundaries — wrap them.
3. Define interfaces in the **consumer** package, not the provider package.

---

## Python Libraries

All entries are pinned in `tracking-orchestrator/pyproject.toml`. Do not add alternatives.

| Library | Role |
| --- | --- |
| `fastapi` | HTTP router and dependency injection for service entrypoints |
| `pydantic v2` | Validation and parsing at all external boundaries (HTTP, config, Redis payloads) |
| `asyncpg` | Async Postgres driver; use `$1..$N` positional params — never `%s` or `?` |
| `redis[hiredis]` | Async Redis Streams client; use consumer groups and XACK |
| `aiobotocore` | Async S3-compatible client for MinIO frame storage |
| `structlog` | Structured, JSON-renderable logging — no `logging.getLogger` |
| `numpy` | Frame data and embedding arithmetic |
| `opencv-python-headless` | Image preprocessing — `cv2.resize(INTER_LINEAR)` for letterbox and crop resizing in all inference modules |
| `scipy` | Hungarian assignment (`linear_sum_assignment`) for tracking |
| `protobuf` | Message contracts for cross-service transport; generated bindings committed in `app/proto/` |
| `prometheus-client` | Metrics exposition (`/metrics`) — use the global registry |
| `pytest` + `pytest-asyncio` | Test runner; all repository tests are `async def` |
| `mypy` (strict) | Type checking on `domain/`, `storage/`, `services/`, `transport/` |
| `ruff` | Lint + format (replaces flake8, isort, black) |
| `import-linter` | Enforces layering contract at CI time |

Do not introduce `aiohttp`, `celery`, or `psycopg2` — the approved stack covers all current
requirements without them.

`httpx` is permitted as a **test-only** dependency (included in `[project.optional-dependencies] dev`).

---

## Go Libraries

All entries are declared in `rtsp-ingress/go.mod`.

| Library | Role |
| --- | --- |
| `minio/minio-go/v7` | S3-compatible object storage client |
| `redis/go-redis/v9` | Redis Streams producer |
| `prometheus/client_golang` | Metrics exposition (`/metrics`) |
| `go.uber.org/zap` | Structured logging (use `zap.L()` for global logger) |
| `google.golang.org/protobuf` | Protobuf runtime for cross-service messages |
| `golang.org/x/image` | Image format decoding used in media processing |
| `gopkg.in/yaml.v3` | Config file parsing |

**RTSP ingest uses go2rtc as a sidecar — not gortsplib.** rtsp-ingress registers/deregisters
streams via go2rtc's HTTP API (`internal/go2rtc/client.go`) and polls `/api/frame.jpeg` for
frames. Do not add `bluenviron/gortsplib` or `pion/rtp` to the Go module.

---

## Redis Stream Wire Format

**All Redis Streams carry raw protobuf bytes — no JSON, no base64.**

Use `transport/codec.py` (`encode` / `decode`) for all publish and consume operations. Each stream
carries one named field whose value is `Message.SerializeToString()` output:

| Stream | Field | Message type |
| --- | --- | --- |
| `frames.ready` | `"frame"` | `FrameReady` |
| `tracking.events` | `"event"` | `TrackingEvent` |
| `tracking.revisions` | `"revision"` | `IdentityRevision` |
| `tracking.signals` | `"signal"` | `DementiaSignal` |
| `scene.samples` | `"sample"` | `SceneSample` |

Rules:

1. Redis clients must set `decode_responses=False` so binary payloads round-trip unchanged.
2. No codec discriminator field — consumers know the message type from the stream they subscribe to.
3. Do not introduce a dual-codec shim; the protobuf-only path is the sole wire format.

---

## Testing Strategy

### Python

1. Unit tests mock at the `Protocol` boundary — inject `InMemoryTrackingRepository` directly.
   Never mock `asyncpg.Pool` or Redis connections in unit tests.
2. All test functions that touch a repository are `async def` and decorated with
   `@pytest.mark.asyncio` (or `asyncio_mode = "auto"` in `pyproject.toml`).
3. Integration tests (testcontainers scope) are isolated in `tests/integration/` and skipped in
   the default `make check` target. They require a live Postgres + Redis container.
4. Use `pytest.fixture` for shared domain objects (`CameraConfig`, `Detection`). Avoid
   constructing the same frozen dataclass inline across many test functions.
5. Tests verify behavior through the public API of the class under test. Never access private
   `_attributes` directly to assert state.
6. Triton calls are mocked via `TritonClientProtocol` — all inference tests run without a GPU.

### Go

1. Prefer table-driven tests (`t.Run`) for functions with multiple input/output cases.
2. Inject interfaces in constructors so tests can replace the real implementation.
3. Always run with `-race`: `go test -race ./...`. This is a hard CI gate.
4. HTTP handler tests use `httptest.NewRecorder()` — no live server required.
5. go2rtc calls are tested with an `httptest.Server` serving mock responses — do not test against
   a live go2rtc instance in unit tests.

---

## Security at Boundaries

1. **Validate at entry points only.** Use Pydantic v2 models for HTTP request bodies, Redis
   message payloads, and config files. Internal domain objects are frozen dataclasses — they
   carry no validation overhead and must already be valid when constructed.

2. **Parameterized SQL, always.** asyncpg requires `$1..$N`; never build SQL strings with
   f-strings or `%` formatting. This is the only defense against injection at the storage layer.

3. **No secrets in code.** Database DSNs, Redis URLs, MinIO credentials, and API keys are
   environment variables resolved at startup. Config files (`settings.yaml`) hold defaults;
   secrets override via env or `.env` files (never committed). The `.env` file is loaded from
   the config directory before YAML parsing; `${VAR}` placeholders in YAML are expanded.

4. **Protobuf for cross-service messages.** Bytes on the wire must parse against a typed schema.
   Do not pass raw JSON blobs without a Pydantic model validating them.

5. **`go vet` + `staticcheck` + `golangci-lint` for Go.** These catch common security
   anti-patterns (unchecked errors, unguarded goroutine leaks) before runtime.

---

## Error Handling

### Python

1. Use the project's `ErrorCode` enum for all domain errors. Never return bare `Exception`.
2. Do not serialize stack traces in API responses. Log the trace locally (`structlog.exception`);
   return only the `ErrorCode` and a human-readable message to the caller.
3. For async operations, let exceptions propagate up to the `_consume_loop` handler, which logs
   and retries. Do not swallow exceptions silently inside `_process_frame`.

### Go

1. Wrap errors with context: `fmt.Errorf("streams: register %q: %w", cfg.ID, err)`.
2. Check every error — `go vet` will flag unchecked error returns.
3. Use `context.Context` as the first parameter of every function that does I/O; respect
   cancellation via `ctx.Err()`.

---

## Datetime and Timezones

All timestamps are timezone-aware. In Python:

```python
from datetime import UTC, datetime

now = datetime.now(UTC)          # correct
now = datetime.now()             # WRONG — naive, breaks TimescaleDB timestamptz
```

In Go, use `time.Now().UTC()` for all timestamps written to Redis or Postgres.

---

## Structured Logging

Python — use `structlog`, never `print` or `logging.getLogger`:

```python
from structlog import get_logger
logger = get_logger(__name__)

logger.info("tracklet created", tracklet_id=tid, camera_id=cam)
logger.exception("frame processing failed", camera_id=frame.camera_id)
```

Go — use `zap.L()` (global) or inject `*zap.Logger` via constructor:

```go
zap.L().Info("stream registered", zap.String("stream_id", cfg.ID))
```

Key fields (`camera_id`, `tracklet_id`, `global_track_id`, `event_id`) must appear in every
log line where they are known. This enables log-based correlation across services.

---

## Observability

- **Python**: use `prometheus-client` against the global registry. Register counters, histograms,
  and gauges in `app/observability/metrics.py`. Expose at `GET /metrics` via the FastAPI app.
- **Go**: use `prometheus/client_golang`. Register metrics in `internal/metrics/metrics.go`.
  Expose at `GET /metrics` on the health-check server.
- Key metrics to instrument: frames consumed, events/revisions/signals published, posterior
  entropy histogram, identity commits, inference latency histograms, gallery size gauge,
  active tracklet gauge.
- OpenTelemetry tracing across Redis Streams boundaries is deferred until a collector target
  exists — do not add OTel spans prematurely.

---

## Image Preprocessing

Use `opencv-python-headless` (`cv2`) for **all** letterbox and crop resizing in inference
modules. Do not re-implement bilinear resize in NumPy.

```python
import cv2
import numpy as np

def letterbox(img: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)
```

The `opencv-python-headless` package (no GUI dependencies) is the canonical choice. Never
import `opencv-python` (includes GUI libs that conflict with headless containers).
