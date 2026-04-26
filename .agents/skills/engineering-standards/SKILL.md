---
name: engineering-standards
description: This skill covers **how** to build - library choices, data abstraction patterns, testing strategy, and security at boundaries. For **what** to build (milestones, domain rules, quality gate, linting configuration), see [CLAUDE.md](CLAUDE.md).
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
2. Never pass `*sql.DB` or raw Redis clients across package boundaries — wrap them.
3. Define interfaces in the **consumer** package, not the provider package.

---

## Python Libraries

All entries are pinned in `tracking-orchestrator/pyproject.toml`. Do not add alternatives.

| Library | Role |
|---|---|
| `fastapi` | HTTP router and dependency injection for service entrypoints |
| `pydantic v2` | Validation and parsing at all external boundaries (HTTP, config, Redis payloads) |
| `asyncpg` | Async Postgres driver; use `$1..$N` positional params — never `%s` or `?` |
| `redis[hiredis]` | Async Redis Streams client; use consumer groups and XACK |
| `aiobotocore` | Async S3-compatible client for MinIO frame storage |
| `structlog` | Structured, JSON-renderable logging — no `logging.getLogger` |
| `numpy` | Frame data and embedding arithmetic |
| `scipy` | Hungarian assignment (`linear_sum_assignment`) for tracking |
| `protobuf` | Message contracts for cross-service transport |
| `pytest` + `pytest-asyncio` | Test runner; all repository tests are `async def` |
| `mypy` (strict) | Type checking on `domain/`, `storage/`, `services/`, `transport/` |
| `ruff` | Lint + format (replaces flake8, isort, black) |
| `import-linter` | Enforces layering contract at CI time |

Do not introduce `sqlalchemy`, `aiohttp`, `httpx`, `celery`, or `psycopg2` — the approved stack
covers all current requirements without them.

---

## Go Libraries

All entries are declared in `rtsp-ingress/go.mod`.

| Library | Role |
|---|---|
| `bluenviron/gortsplib/v4` | RTSP client and server |
| `minio/minio-go/v7` | S3-compatible object storage client |
| `redis/go-redis/v9` | Redis Streams producer |
| `pion/rtp` | RTP packet parsing |
| `prometheus/client_golang` | Metrics exposition (`/metrics`) |
| `go.uber.org/zap` | Structured logging (use `zap.L()` for global logger) |
| `google.golang.org/protobuf` | Protobuf runtime for cross-service messages |
| `gopkg.in/yaml.v3` | Config file parsing |

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

### Go

1. Prefer table-driven tests (`t.Run`) for functions with multiple input/output cases.
2. Inject interfaces in constructors so tests can replace the real implementation.
3. Always run with `-race`: `go test -race ./...`. This is a hard CI gate.
4. HTTP handler tests use `httptest.NewRecorder()` — no live server required.

---

## Security at Boundaries

1. **Validate at entry points only.** Use Pydantic v2 models for HTTP request bodies, Redis
   message payloads, and config files. Internal domain objects are frozen dataclasses — they
   carry no validation overhead and must already be valid when constructed.

2. **Parameterized SQL, always.** asyncpg requires `$1..$N`; never build SQL strings with
   f-strings or `%` formatting. This is the only defense against injection at the storage layer.

3. **No secrets in code.** Database DSNs, Redis URLs, MinIO credentials, and API keys are
   environment variables resolved at startup. Config files (`settings.yaml`) hold defaults;
   secrets override via env. Never commit `.env` files.

4. **Protobuf for cross-service messages.** Bytes on the wire must parse against a typed schema.
   Do not pass raw JSON blobs without a Pydantic model validating them.

5. **`go vet` + `staticcheck` + `golangci-lint` for Go.** These catch common security anti-patterns
   (unchecked errors, unguarded goroutine leaks) before runtime.

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
