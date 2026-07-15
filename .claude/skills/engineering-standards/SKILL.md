---
name: engineering-standards
description: Use when changing CTS architecture, repositories, configuration, tests, wire formats, units, caches, or security boundaries. For domain requirements and quality gates, see CLAUDE.md.
---

# Engineering Standards

This skill documents the current codebase. Update it in the same PR that changes any behavior
or contract it describes; task files own future deltas until their implementation lands.

## Architecture and SOLID Principles

### Single Responsibility

Every module, class, and function has exactly one reason to change.

- **Pipeline stages** do one thing: fetch, detect, redact, project, infer, track, resolve, write, sample, revise, or publish. A stage that does two unrelated things is a bug.
- **Services** coordinate; they do not own algorithms. `DementiaSignalWorker` schedules and
  coordinates; individual detectors own the algorithms. See
  [CTS Signals](../cts-signals/SKILL.md) for the detector contract.
- **Domain objects** are pure data -- frozen dataclasses with no behavior beyond property accessors.

### Dependency Inversion

High-level policy never depends on low-level implementation details. Both depend on abstractions.

- Services depend on `Protocol` interfaces, never on `Postgres*` or `Redis*` concrete classes.
- Constructors accept abstractions (`TrackingRepository`, not `PostgresTrackingRepository`).
- No service locator pattern. No `global` or module-level singletons used as implicit dependencies. The calibration state singleton is a known exception tracked as tech debt.
- Every dependency is explicit in the constructor signature. A reader can see what a class needs without reading its implementation.

### Open-Closed

Modules are open for extension, closed for modification.

- New signal detectors are added as new classes implementing a detector protocol -- never by adding `elif` branches inside `_process_identity`.
- New pipeline stages are added by creating a new `FrameStage` and inserting it into the stage list -- never by adding a private method and a call site inside `_process_frame`.

### Composition over inheritance

Prefer small, composable objects injected via constructors over deep inheritance hierarchies.

- `SpatialProjectionService` is injected into `SpatialProjectionStage`, not subclassed.
- `IdentityResolver._commit` is a private method; callers pass the resolved posterior, not a shared state object.

---

## Data Abstraction Layer

The goal is backend portability: swap Postgres for a different store by replacing one class, not touching services.

### Python: Protocol + InMemory + Postgres triplet

Every persistent resource gets three artifacts:

1. A `Protocol` in `storage/base.py` -- the contract services depend on.
2. An `InMemory*` class in the same file -- fast, zero-dependency, used in all unit tests.
3. A `Postgres*` class in `storage/postgres/` -- the production implementation.

Services import only the `Protocol`. Import-linter enforces the layering contract (`core → domain → storage → services → transport → routers`). Nothing above `storage` may import a concrete `Postgres*` class.

```python
# storage/base.py
class TrackletRepository(Protocol):
    async def save_tracklet(self, tracklet: Tracklet) -> None: ...
    async def get_tracklet(self, tracklet_id: str) -> Tracklet | None: ...
```

Rules:

1. All repository methods are `async`.
2. Return domain types, never raw DB rows or dicts.
3. `Protocol` carries no state -- it is a pure structural interface.
4. `InMemory` uses plain `dict` / `list`; never touches a database or network.
5. `Postgres*` receives only an `asyncpg.Pool`; holds no other state.

### Wiring completeness for repository Protocols

A repository abstraction is done only when all three artifacts exist and production wiring is
proved: a Postgres implementation, explicit injection through `app/main.py`, and an integration
test exercising a production-shaped path. The Protocol plus InMemory implementation is not done.

```python
# RIGHT: app/main.py; proof: tests/integration/test_baseline_repo_postgres.py
baseline_repo = PostgresBehaviorBaselineRepository(_pool)
deps = PipelineDependencies(baseline_repo=baseline_repo)

# WRONG: production silently reaches an InMemory default
deps = PipelineDependencies()  # initialize() substitutes InMemoryBehaviorBaselineRepository
```
An optional dependency may have an InMemory default only for tests. The production constructor
site must name the concrete class. For every repository Protocol or ABC in `app/storage/`, run:

```bash
grep -nE "class .*(Protocol|ABC)" tracking-orchestrator/app/storage/*.py
grep -n "PostgresXRepository" tracking-orchestrator/app/main.py  # repeat for each X
```

Any missing counterpart requires a written justification comment at the wiring site.

### Repository state/version filters must match across the triplet

Any filter baked into a Postgres query (state, active, version) must appear identically in
the InMemory peer and in the Protocol signature as an explicit parameter with a shared
default constant. A filter that exists in only one implementation is a parity defect even if
all tests pass. New filtered reads require a parity matrix test.

**RIGHT:** `app/storage/gallery.py`'s `VERIFIED_ONLY` constant is the shared default for
`list_gallery_entries`, `search_similar`, `list_gallery_entries_for_tracklets`, and
`gallery_similarity` across the Protocol, `InMemoryGalleryRepository`, and
`PostgresGalleryRepository`; `tests/storage/test_gallery_state_parity.py` (InMemory, fast gate)
and `tests/integration/test_gallery_state_parity_postgres.py` (Postgres, `make ci`) prove both
implementations return identical entry-ID sets for every state-filter value (M03).

**WRONG:** a Postgres SQL constant hard-codes `state = 'operator_verified'` while the InMemory
peer applies no state filter at all -- unit tests seeded with the default lifecycle state then
"prove" behavior (a pending/rejected row voting) that production, filtered by the SQL, can never
produce.

### Go: interface contracts in `internal/`

Each subsystem in `rtsp-ingress/internal/` exports an interface, not a concrete struct. Tests inject a fake; production wires the real implementation.

```go
// streams/streams.go
type Manager interface {
    Register(ctx context.Context, cfg StreamConfig) error
    Stop(streamID string) error
}
```

1. Accept interfaces, return concrete types.
2. Never pass raw Redis clients across package boundaries -- wrap them.
3. Define interfaces in the **consumer** package, not the provider.

---

## Dependency Injection

### Constructor injection only

Every dependency is a required constructor parameter. No optional service locators, no `if self._foo is not None` guards that silently skip work, no `getattr(self, "_foo", None)` fallbacks.

```python
# RIGHT -- explicit, testable
class TrajectoryWriter:
    def __init__(self, repo: TrajectoryRepository) -> None:
        self._repo = repo

# WRONG -- hidden dependency
class TrajectoryWriter:
    def __init__(self) -> None:
        self._repo = _create_default_repo()  # can't inject a mock
```

### Optional dependencies with sensible defaults

When a dependency is genuinely optional (e.g., `pose_estimator` may be None when pose is disabled), make it a keyword argument defaulting to `None`. The caller explicitly opts into omission. Never silently degrade without logging.

```python
class InferenceStage(FrameStage):
    def __init__(
        self,
        reid_embedder: ReidEmbedderProtocol | None = None,
        pose_estimator: PoseEstimator | None = None,
    ) -> None:
        ...
```

### No module-level singletons as hidden state

Module-level mutable state makes tests order-dependent and prevents parallel execution. The `calibration_state` singleton is a known exception (mutated only via async-locked methods, read by the hot-reload path). All other state lives in constructor-injected objects.

---

## Immutability and Pure Functions

### Frozen dataclasses by default

Every domain object is `@dataclass(frozen=True)` unless it has a documented reason to be mutable. Immutability prevents accidental mutation, makes hash-based dedup safe, and enables `dataclasses.replace()` for explicit state transitions.

```python
@dataclass(frozen=True)
class Detection:
    detection_id: str
    camera_id: str
    bbox: BoundingBox
    ...

# State change: explicit, traceable.
detection = replace(detection, tracklet_id="tl-1", global_track_id="gt-1")
```

### Never mutate frozen instances

`FrozenInstanceError` is a correctness guard. Use `dataclasses.replace()` to create new instances with updated fields. Never use `object.__setattr__` to bypass the freeze -- if you need mutability, the type should not be frozen.

### Class-level constants outside the dataclass

`set`, `dict`, and `list` defaults on frozen dataclass fields raise `ValueError: mutable default` at class definition time. Move class-level constants to module scope.

```python
# WRONG
@dataclass(frozen=True)
class IdentityEvidence:
    CAN_CREATE_IDENTITY: set[str] = {"direct_face", "reid"}  # ValueError

# RIGHT
CAN_CREATE_IDENTITY: set[str] = {"direct_face", "reid"}

@dataclass(frozen=True)
class IdentityEvidence:
    ...
```

### Pure functions for business logic

Algorithms live in pure functions that take inputs and return outputs. They never read global state, call I/O, or mutate arguments. This makes them trivially testable with table-driven tests.

```python
# RIGHT -- pure, testable without any fixture
def combine_evidence(
    evidence_list: list[IdentityEvidence],
    known_identities: set[str],
) -> EvidencePosterior:
    ...

# WRONG -- reads self._config, self._identities, calls self._gallery_repo
def _from_gallery(self, entity: IdentityResolvableEntity) -> PosteriorDist:
    ...
```

---

## Import Discipline

Rules based on [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and Microsoft pyright best practices, adapted to this codebase.

### All imports at the top of the file

Imports go at the top of the file, after the module docstring, before any executable code. Every import statement is executed at module load time -- no exception.

```python
# RIGHT
from __future__ import annotations
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.domain import PersonHypothesis

# THEN executable code
def process(ph: PersonHypothesis) -> None: ...

# WRONG -- conditional import on runtime data
if adjacency_edges_raw:
    from .calibration.state import calibration_state  # may never execute
```

### No conditional imports on runtime data

Never guard an import behind a condition that depends on runtime values (API responses, config flags, database contents). This produces `UnboundLocalError` when the condition is false and the name is referenced later.

The only legitimate conditional imports are:

| Pattern | Example | Why permitted |
|---|---|---|
| Optional dependency | `try: import numpy` / `except ImportError: ...` | Library may not be installed |
| Platform-specific | `if sys.platform == "win32": import msvcrt` | Module only exists on that platform |
| TYPE_CHECKING guard | `if TYPE_CHECKING: from ..pipeline import ...` | Avoids circular imports at runtime |
| Version fallback | `if sys.version_info >= (3, 12): ...` | Polyfill for older Python |

Any other conditional import is a bug. If a module-level import would cause a circular dependency, use `TYPE_CHECKING` or restructure the code -- don't bury the import in a function body.

### No import side-effects

Importing a module must not mutate global state, start threads, open connections, register atexit handlers, or modify `sys.path`. If a module needs initialization, expose a factory function or class.

```python
# WRONG -- side-effect at import time
# calibration/state.py
calibration_state = CalibrationState()  # singleton created on import

# RIGHT -- explicit initialization controlled by the caller
# calibration/state.py
def create_calibration_state() -> CalibrationState:
    return CalibrationState()
```

The existing `calibration_state` module-level singleton is a known exception (tracked as tech debt). No new singletons created at import time.

### Import ordering

Use three groups separated by a blank line, enforced by ruff `I001`:

1. Standard library (`from __future__`, `from datetime`, `import asyncio`, ...)
2. Third-party (`import asyncpg`, `from pydantic import BaseModel`, ...)
3. First-party (`from app.domain import ...`, `from .config import ...`)

Never use relative imports across package boundaries. `from ..pipeline.stages import ...` from inside `app/tracking/` is a layering violation that also creates cycles.

### Wildcard imports are forbidden

`from module import *` pollutes the namespace, defeats static analysis, and hides the origin of names. Import each symbol explicitly.

### Circular Import Prevention

The `tracking <-> pipeline` boundary is the most fragile import path in this codebase. When a tracking submodule imports from `..pipeline.*`, and the pipeline package init re-enters tracking, the module fails with `ImportError: cannot import name ... from partially initialized module`.

1. **Use `TYPE_CHECKING` for cross-boundary type annotations.** When a tracking module needs a pipeline type only for annotations, put the import inside `if TYPE_CHECKING:`. With `from __future__ import annotations`, the annotation is a string at runtime so the import is never executed.

    ```python
    from __future__ import annotations
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ..pipeline.gallery_cache import GalleryCache
    ```

2. **Do not eagerly re-export types in `__init__.py` that create cycles.** Package init files should be minimal. Any re-export that triggers a chain back to the importing module is a cycle.

3. **Shared types between packages live in a neutral module.** Types needed by both `frame_pipeline` and `stages/` live in `pipeline/types.py` -- imported by both sides without creating a cycle.

---

## Type Safety

### `from __future__ import annotations` in every file

All annotations are strings at runtime (PEP 563). This enables forward references and reduces import-time overhead. Every Python file in this codebase starts with this import.

### Exhaustive type annotations

All function signatures have parameter and return types. No bare `def foo(x):` -- always `def foo(x: int) -> str:`. `mypy --strict` runs on `domain/`, `storage/`, `services/`, and `transport/`.

### Strict Optional

Implicit `None` is a leading source of production bugs. `mypy` strict-optional is enabled. Every value that can be `None` must be annotated as `T | None`.

```python
# RIGHT -- explicit about nullability
def lookup(key: str) -> PersonHypothesis | None: ...

# WRONG -- no annotation, caller doesn't know None is possible
def lookup(key: str): ...
```

When a variable starts as `None` but is always assigned before use, declare it upfront with its full type:

```python
result: PersonHypothesis | None = None
if match:
    result = match
# mypy understands the narrowing
```

### Use modern union syntax

Python 3.10+ union syntax (`T | None`, `int | str`) throughout. No `Optional[T]`, no `Union[T, U]`. This is enforced by ruff `UP007` / `UP045`.

### No escape hatches without justification

- `# type: ignore` must use the specific mypy error code: `# type: ignore[arg-type]`.
- Bare `# type: ignore` is deprecated and fails CI.
- Remove `type: ignore` comments when the underlying issue is resolved -- mypy flags unused suppressions with `[unused-ignore]`.
- `Any` is prohibited in constructor signatures and return types. It is tolerated only in `**kwargs` passthrough to third-party libraries with `cast()` at the boundary.

### TypedDict for structured dicts, NamedTuple for simple carriers

When a dict has a known set of keys, use `TypedDict` so mypy can verify key access. Prefer `NamedTuple` for small, immutable value objects that don't need the overhead of a frozen dataclass.

```python
from typing import NamedTuple, TypedDict

class CameraEdge(TypedDict):
    from_camera: str
    to_camera: str
    min_transit_s: float
    max_transit_s: float

class PixelCoord(NamedTuple):
    x: float
    y: float
```

### `Protocol` for structural subtyping, `ABC` only when state is shared

Use `typing.Protocol` for dependency inversion (repository interfaces, strategy contracts). Reserve `abc.ABC` / `abc.abstractmethod` for base classes that share concrete state or template methods.

### NumPy type boundaries

Comparisons on numpy scalars return `numpy.bool_`, not Python `bool`. Functions annotated `-> bool` that perform numpy comparisons must wrap the result:

```python
def is_degenerate(crop: npt.NDArray[np.uint8]) -> bool:
    return bool(w < _MIN_CROP_WIDTH or h < _MIN_CROP_HEIGHT)
```

### `callable` is not a type

The builtin `callable` is a function, not a valid type annotation. Use `Callable[[ArgType], ReturnType]` from `collections.abc` when the signature matters, or `list[object]` for heterogeneous callable containers.

### Pydantic at boundaries, frozen dataclasses internally

Use Pydantic v2 models for HTTP request/response bodies, Redis payloads, config files, and any data crossing a service boundary. Use frozen dataclasses for internal domain objects. Never leak Pydantic model instances into service or repository layers -- convert to domain types at the boundary.

---

## Async/Concurrency

### Task management

- Every background task is created with `asyncio.create_task()` and stored in a list. `stop()` cancels all tasks and awaits their completion.
- Never fire-and-forget a coroutine with `asyncio.ensure_future()` without storing the handle.
- Tasks that run indefinitely must catch `asyncio.CancelledError` and clean up.

### Concurrency gates

- `asyncio.Semaphore` limits concurrent frame processing. The default is 4 -- tuned to GPU batch size and memory.
- Per-camera `asyncio.Lock` preserves frame ordering within a single camera. Two frames from the same camera never process concurrently.

### Cancellation and shutdown

- `stop()` sets a `_stop_event` (via `asyncio.Event`) and cancels all tasks.
- Loops check `self._stop_event.is_set()` or `self._running` at the top of each iteration.
- Shutdown is bounded by `shutdown_timeout` (default 5s). Tasks that don't complete within the timeout are abandoned with a warning log.

### No blocking calls in async context

- CPU-bound work (numpy array ops, image decoding) is acceptable if it runs in under 1ms. Longer operations use `asyncio.to_thread()` or a process pool.
- Never call `time.sleep()` in an async function. Use `asyncio.sleep()`.

---

## Error Handling

### Python

1. Domain errors use the project's `ErrorCode` enum. Never return or raise bare `Exception` for domain failures.
2. Stack traces never appear in API responses. Log the trace locally (`structlog.exception`); return only the error code and a human-readable message.
3. Async frame processing lets exceptions propagate to the `_consume_loop` handler, which logs and retries. Individual frames are never silently dropped -- failed frames produce a `FrameResponse` with `success=False` and an error code.
4. Model failures degrade per model: detector failure fails the frame; ReID/pose/face-ID failure degrades to missing evidence without dropping the frame.

### Go

1. Wrap errors with context: `fmt.Errorf("streams: register %q: %w", cfg.ID, err)`.
2. Check every error return -- `go vet` and `golangci-lint` flag unchecked returns as compile errors.
3. `context.Context` is the first parameter of every I/O function. Respect cancellation via `ctx.Err()`.

---

## Configuration Management

### Typed, frozen config

All configuration is defined as frozen dataclasses with sensible defaults. No raw `dict` access for config values -- every setting is a typed field.

```python
@dataclass(frozen=True)
class PipelineConfig:
    detector_confidence: float = 0.25
    max_concurrent_frames: int = 4
    signal_enabled: bool = True
    ...
```

### Environment overrides

Secrets (database DSNs, Redis URLs, MinIO credentials, API keys) come from environment variables. Config files hold defaults; env vars override. `.env` files are never committed. The startup path loads `.env` before YAML, then expands `${VAR}` placeholders.

### Config is validated at startup

Use Pydantic v2 models to validate configuration at application startup. Fail fast with a clear error message if required settings are missing or invalid. Never silently fall back to a default that produces incorrect behavior.

### Flag lifecycle

New behavior ships behind a `settings.yaml` flag defaulting to off. Its YAML comment states why
the flag exists and what evidence is required before enablement.

1. Add shadow mode when the behavior can run metrics-only without changing output.
2. Add a fixture or replay proof and a named guardrail test for the worst forbidden failure.
3. Append a dated enable note and the gating proofs to the CTS robustness rollout notes.
4. Remove the flag one release after enablement proves stable; removal is its own small PR.

**RIGHT:** `pipeline.adaptive_reid` combines an off-by-default flag, shadow mode, replay metrics,
and identity-contamination guardrails in `app/pipeline/reid_policy.py` and
`tests/integration/test_adaptive_reid_replay_equivalence.py`.

**WRONG:** merge a default-on behavior flag with no replay proof, then leave it permanently true.

---

## API Design (FastAPI)

### Request/response models

- Every endpoint has explicit Pydantic request and response models. No `dict[str, Any]` in endpoint signatures.
- Response models exclude internal fields -- the API surface is deliberate and documented.
- Use `response_model=` on `@router` decorators, not `response_model_by_alias` or manual dict construction.

### Error responses

- Use `HTTPException` with a structured `detail` dict: `{"code": "calibration.invalid_points", "message": "..."}`.
- Error codes are `snake_case` strings, namespaced by domain: `calibration.*`, `homography.*`.
- Never expose internal state, stack traces, or file paths in error responses.

### Internal vs external endpoints

- `/internal/calibration/*` endpoints are not publicly exposed. They are called by the Cognitive Companion BFF, authenticated via service JWT.
- Public endpoints follow REST conventions: `GET /resource`, `POST /resource`, `GET /resource/{id}`.

---

## Testing Strategy

### Python

1. **Test at the Protocol boundary.** Inject `InMemory*` implementations directly. Never mock `asyncpg.Pool`, Redis connections, or HTTP clients in unit tests.
2. **All repository tests are `async def`.** Configured via `asyncio_mode = "auto"` in `pyproject.toml`.
3. **Integration tests in `tests/integration/`.** Skipped by default in `make check`. Require live Postgres + Redis containers.
4. **Shared fixtures via `pytest.fixture`.** Build `CameraConfig`, `Detection`, `Tracklet` once and reuse. Avoid reconstructing the same frozen dataclass inline across tests.
5. **Tests assert behavior through public APIs.** Never assert on private `_attributes`. Tests that reach into private state are brittle and test implementation details, not contracts.
6. **Triton calls mocked via `TritonClientProtocol`.** All inference tests run without a GPU.
7. **Falsifiable assertions.** For signal detectors: "7-day TV-watching fixture produces zero stillness_anomaly", not just "some signal was produced". Assert counts, severity levels, and non-emissions.
8. **Hysteresis tests must be explicit.** One-run tests set `onset_consecutive_windows=1`. Full hysteresis tests use the default of 2 and verify onset debounce, cooldown, and escalation independently.

### Go

1. Table-driven tests (`t.Run`) for multi-case functions.
2. Interfaces injected in constructors; tests replace with fakes.
3. `-race` flag mandatory: `go test -race ./...`.
4. HTTP handlers tested with `httptest.NewRecorder()`.
5. go2rtc calls tested with `httptest.Server` serving mock responses.

### What NOT to test

- Do not test Python dataclass constructors (the language tests field assignment).
- Do not test third-party library behavior (Redis xadd, protobuf serialization).
- Do not test trivial property accessors or getters.
- Do not write tests that exist only to hit a coverage percentage -- every test must document a real failure mode.

---

## Security at Boundaries

1. **Validate at entry points only.** Pydantic v2 for HTTP bodies, Redis payloads, and config files. Internal domain objects must already be valid when constructed.
2. **Parameterized SQL, always.** asyncpg `$1..$N` -- never f-strings or `%` formatting.
3. **No secrets in code.** Environment variables only. `.env` never committed.
4. **Protobuf for cross-service messages.** Bytes on the wire parse against a typed schema. No raw JSON blobs without Pydantic validation.
5. **`go vet` + `staticcheck` + `golangci-lint` for Go.** Hard CI gates.

---

## Datetime and Timezones

All timestamps are timezone-aware:

```python
from datetime import UTC, datetime

now = datetime.now(UTC)          # correct
now = datetime.now()             # WRONG -- naive, breaks TimescaleDB timestamptz
```

Pipeline stages must use the correct time for the domain concept:
- `ctx.capture_time` -- physical observation timestamp (from the camera).
- `ctx.event_time` -- pipeline processing timestamp (set once at context init).
- `datetime.now(UTC)` -- only for true wall-clock side effects (e.g., `emitted_at` on a signal).

In Go, use `time.Now().UTC()` for all timestamps written to Redis or Postgres.

---

## Structured Logging

Python -- `structlog`, never `print` or `logging.getLogger`:

```python
from structlog import get_logger
logger = get_logger(__name__)

logger.info("tracklet created", tracklet_id=tid, camera_id=cam)
logger.exception("frame processing failed", camera_id=frame.camera_id)
```

Go -- `zap.L()` or inject `*zap.Logger`:

```go
zap.L().Info("stream registered", zap.String("stream_id", cfg.ID))
```

Key fields (`camera_id`, `tracklet_id`, `global_track_id`, `event_id`) appear in every log line where they are known.

### Log levels

- `debug`: per-frame/per-detection detail, model outputs, rejection reasons. Verbose, sampled in production.
- `info`: lifecycle events (startup, shutdown, config), signal emissions, identity commits, revision publications.
- `warning`: degraded operation (model timeout, service unavailable, stale frame), recoverable errors.
- `exception`: unrecoverable per-frame failures with full traceback. Always includes `camera_id` and `frame_index`.

### What NOT to log

- Never log embedding vectors, face crops, or full image data.
- Never log secrets, tokens, or full connection strings.
- Never log PII (person names, face images) without a documented anonymization policy.

---

## Observability

- **Python**: `prometheus-client` against the global registry. Register counters, histograms, and gauges in `app/observability/metrics.py`. Expose at `GET /metrics`.
- **Go**: `prometheus/client_golang`. Register in `internal/metrics/metrics.go`.

Key metrics: frames consumed, events/revisions/signals published, posterior entropy histogram, identity commits by source, demotions by reason, stage latency histogram (per stage, per camera), inference latency (per model), gallery size, active tracklet gauge, signal worker run duration.

OpenTelemetry tracing across Redis Streams is deferred until a collector target exists.

### Shadow computations must be sample-gated

Shadow computations (running an alternative configuration purely to compare outcomes) must
be gated by an explicit sample-rate config defaulting to 0.0 (off), written as
`rate > 0.0 and random.random() < rate`. An always-on shadow that issues I/O (gallery, DB)
is a performance defect. Route all shadow comparisons through one helper so the gate cannot
be forgotten.

---

## Image Preprocessing

Use `opencv-python-headless` for all letterbox and crop resizing:

```python
import cv2

resized = cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)
```

Never import `opencv-python` (GUI dependencies conflict with headless containers). Never hand-roll bilinear resize in NumPy -- it's slower and less correct.

---

## Python Virtual Environment

**Always use the project venv at `tracking-orchestrator/.venv/`; never invoke the system Python.** The Makefile targets activate it automatically. For direct invocation:

```bash
# Activate for interactive use
source tracking-orchestrator/.venv/bin/activate

# Run directly without activating (preferred for one-off commands)
tracking-orchestrator/.venv/bin/python -m pytest tests/...
tracking-orchestrator/.venv/bin/python -c "from app.domain import PersonHypothesis; print('ok')"

# Sync after pyproject.toml changes
cd tracking-orchestrator && uv sync --frozen --extra dev
```

Running bare `python` or `pip` installs into the system interpreter and silently breaks the locked dependency graph. Every command in this skill that shows `python ...` means the venv Python above.

---

## Database Migrations

### Tool split (intentional, not to be unified)

| Service | Tool | Why |
|---|---|---|
| `tracking-orchestrator` | Custom `MigrationRunner` + raw SQL | asyncpg + TimescaleDB non-transactional DDL requires the `-- migrate:no-transaction` split path; no SQLAlchemy ORM |
| `cognitive-companion` | Alembic | SQLAlchemy ORM with `target_metadata` autogenerate support |

### Lifecycle: pre-release vs. post-release

**Pre-release (current state):** A single squashed baseline `0001_NNNN` contains the complete
final schema. There are no incremental migration files. All changes go directly into the
baseline. Existing dev databases must be **dropped and recreated**: the `_schema_version`
table and Alembic `alembic_version` table hold stale entries from any prior chain.
Shape constraints belong in the baseline too: JSONB columns that the domain treats as objects
must have `CHECK (jsonb_typeof(column) = 'object')`, and repository mappers must reject invalid
legacy rows instead of silently coercing them.

**Post-release:** Each atomic schema change gets its own numbered file:
`NNNN_description.up.sql` / `NNNN_description.down.sql` (CTS) or `NNNN_description.py`
(CC). Rollbacks are supported and the `downgrade()` / `.down.sql` must be correct.

### CTS migration conventions

- Files: `migrations/NNNN_description.up.sql` and `migrations/NNNN_description.down.sql`.
- Zero-pad to 4 digits: `0001`, `0002`, …
- Every DDL object lives in `continuous_tracking` schema. Open every migration with:
  `SET search_path = continuous_tracking, public;`
- Non-transactional DDL (hypertables, continuous aggregates, policies) must declare:
  `-- migrate:no-transaction` as the **first line** of the file.
- The baseline down file is a no-op comment directing the operator to drop and recreate.
- Applied state is tracked in `continuous_tracking._schema_version` (managed by `MigrationRunner`).
- Advisory lock key `0x4354535F4D4947` ("CTS_MIG") ensures only one replica runs migrations.

### CC migration conventions

- Files: `alembic/versions/NNNN_description.py`.
- The baseline `downgrade()` is intentionally a no-op (`pass`). Document this in the docstring.
- Post-release migrations must have a correct `downgrade()` that exactly reverses `upgrade()`.
- Run `alembic check` against the live models to verify no schema drift before committing.

### Verification (no live DB required)

Grep the new baseline to confirm the fold is complete.

```bash
# CTS: zero hits -- no surviving FK to the removed legacy tables
grep -nE "REFERENCES (global_tracks|tracklets)" migrations/0001_init.up.sql

# CTS: zero hits -- legacy tables must not be created
grep -n "CREATE TABLE.*global_tracks\|CREATE TABLE.*tracklets\b" migrations/0001_init.up.sql

# CC: zero hits for superseded index names
grep -n "ix_person_location_history_global_track_id\|ix_cts_identity_revision_log_gt_applied" alembic/versions/0001_baseline.py

# CC: single head = 0001_baseline, no dangling revision references
cd cognitive-companion/backend && .venv/bin/alembic heads && .venv/bin/alembic history
```

Note: `detections.global_track_id` and its index are intentionally retained as plain columns
(no FK); they carry historical frame-level data and are exempt from the "zero hits" rule.

---

## Quality Gate

```bash
make check   # Python: ruff check + ruff format --check + mypy + import-linter + pytest
make check-all   # Python + Go (golangci-lint + go test -race + go build) + buf lint
```

### ruff rules (non-negotiable)

- `I001`: imports sorted within groups (stdlib → third-party → first-party).
- `E501`: 100-character line limit. Break long calls, wrap signatures, extract intermediate variables for long f-strings.
- `N806`: `SCREAMING_CASE` at module level; `lowercase` in function scope.
- `F401`, `F841`: no unused imports or variables. Unused unpacked names → prefix with `_`.
- `B007`: unused loop variables → `_i`, `_k`, `_v`.
- `SIM102`: combine nested `if` with `and` when the body is a single statement.
- `SIM108`: ternary for simple conditional assignments.
- `B017`: no `pytest.raises(Exception)` -- use the specific exception type.
- `C401`: use set/dict comprehensions, not generator-wrapped constructors.

### What `make check` does NOT cover

- Integration tests (opt-in, require containers).
- Performance regression (no benchmark suite yet).
- Behavioral correctness of ML model outputs (requires labeled test data).
- End-to-end cross-service contract compliance (requires CC in the loop).

---

## Code Review Checklist

For every PR affecting `tracking-orchestrator/`:

1. **Does it add a dependency?** Must be in the approved list or have explicit justification.
2. **Does it introduce a circular import?** Check `tracking <-> pipeline` boundaries. New imports from `pipeline` into `tracking` must use `TYPE_CHECKING`.
3. **Does it mutate a frozen dataclass?** All state changes must use `replace()`.
4. **Are all new dataclasses frozen by default?** If mutable, why?
5. **Do stages have a single responsibility?** No multi-purpose stages.
6. **Are constructor dependencies explicit?** No `self._pipeline` passthrough.
7. **Are errors handled per model?** Detector failure ≠ ReID failure ≠ face-ID failure.
8. **Do timestamps use the correct domain time?** `capture_time` for observation, `event_time` for processing, `datetime.now(UTC)` only for wall-clock side effects.
9. **Are new metrics registered?** Stage latency, signal emissions, identity commits.
10. **Do tests assert falsifiable outcomes?** Counts, severity levels, non-emissions, not just truthiness.
11. **Is `ruff format` clean?** Run it before committing.
12. **Does the proto change have a CC counterpart?** Proto changes are two-repo changes.

---

## Python Libraries

| Library | Role |
| --- | --- |
| `fastapi` | HTTP router and dependency injection |
| `pydantic v2` | Validation at all external boundaries |
| `asyncpg` | Async Postgres driver -- `$1..$N` positional params |
| `redis[hiredis]` | Async Redis Streams -- consumer groups + XACK |
| `aiobotocore` | Async S3-compatible client for MinIO |
| `structlog` | Structured JSON logging -- no `logging.getLogger` |
| `numpy` | Frame data and embedding arithmetic |
| `opencv-python-headless` | Image preprocessing -- `cv2.resize(INTER_LINEAR)` |
| `scipy` | Hungarian assignment, robust statistics (MAD, EWMA) |
| `shapely>=2.0` | Polygon containment for privacy zones |
| `protobuf` | Message contracts -- generated bindings committed in `app/proto/` |
| `prometheus-client` | Metrics exposition at `/metrics` |
| `pytest` + `pytest-asyncio` | Test runner -- all repo tests are `async def` |
| `mypy` (strict) | Type checking on `domain/`, `storage/`, `services/`, `transport/` |
| `ruff` | Lint + format |
| `import-linter` | Layering enforcement at CI time |

Do not add `aiohttp`, `celery`, or `psycopg2`. `httpx` is permitted as a **test-only** dependency.

---

## Go Libraries

| Library | Role |
| --- | --- |
| `minio/minio-go/v7` | S3-compatible object storage |
| `redis/go-redis/v9` | Redis Streams producer |
| `prometheus/client_golang` | Metrics exposition |
| `go.uber.org/zap` | Structured logging |
| `google.golang.org/protobuf` | Protobuf runtime |
| `golang.org/x/image` | Image format decoding |
| `gopkg.in/yaml.v3` | Config file parsing |

RTSP ingest uses go2rtc as a sidecar -- do not add `gortsplib` or `pion/rtp`.

---

## Redis Stream Wire Format

**All Redis Streams carry raw protobuf bytes -- no JSON, no base64.**

| Stream | Field | Message type |
| --- | --- | --- |
| `frames.ready` | `"frame"` | `FrameReady` |
| `tracking.events` | `"event"` | `TrackingEvent` |
| `tracking.revisions` | `"revision"` | `IdentityRevision` |
| `tracking.signals` | `"signal"` | `DementiaSignal` |
| `scene.samples` | `"sample"` | `SceneSample` |

1. Redis clients use `decode_responses=False`.
2. No codec discriminator field -- consumers know the message type from the stream.
3. No dual-codec shim; protobuf-only is the sole wire format.

---

## Domain-Specific Design Principles

### Units are part of the interface

Every dimensional numeric field names its unit: `_m`, `_m_s`, `_nu_s`, `_deg`, or `_px`. Its
docstring also names the reference frame, such as image-normalized, crop-normalized, or
floor-plane meters.

Threshold constants sit beside a comment stating the unit and calibration source. Link the
calibration script or fixture. Multiplying or dividing quantities from incompatible frames is a
review-blocking defect even when typical-scale output looks plausible.

```python
# WRONG: crop-normalized displacement / bbox_diagonal_px / dt_s
# RIGHT: app/trajectory/motion_energy.py converts to image px, then divides by bbox px and dt_s.
frame_vel = float(np.mean(disp / scale) / dt)
```

The RIGHT implementation names the result `mean_keypoint_velocity_nu_s` and links
`scripts/calibrate_motion_energy.py` beside `_STILL_VELOCITY_FLOOR_NU_S`.

### Caches with temporal contracts

A cache valid "per frame", "per round", or "per run" must name that epoch in its class docstring.
Invalidate it at one structural chokepoint traversed by every execution path, not separately at
each path entry. Add a staleness backstop that logs and degrades safely after missed invalidation.

**RIGHT:** `app/pipeline/gallery_cache.py` documents a per-frame cache, self-invalidates after
`max_age_s`, and logs `gallery_cache_stale_self_invalidated`. Both single-frame and batched paths
call `FrameProcessingPipeline._begin_tracker_round()` in `app/pipeline/frame_pipeline.py`.

**WRONG:** clear the gallery cache only in `_process_frame`; the cross-camera batch path then
serves entries from the previous tracker round.

Review checklist when adding an execution path: which epoch-scoped caches does this path reset?

### Frame pipeline stages

- **One stage, one responsibility.**
- **Typed stage contracts.** `FrameStage.run(ctx)`. Constructor dependencies are explicit protocols, never a whole pipeline instance.
- **Context ownership is documented.** Every `FrameContext` field has one producing stage and documented consuming stages. No untyped scratch fields.
- **Use frame time deliberately.** `capture_time` = physical observation. `event_time` = processing time. No `datetime.now(UTC)` when one of these is correct.
- **Model failures degrade by model per frame.** Detector failure = frame fails. ReID/pose/face-ID failure = missing evidence.
- **No private cross-stage reachback.** Shared logic becomes a small service injected into both stages.

### Spatial calibration and homography

- **Coordinate system is explicit.** Calibrated `FloorPoint` = shared floor-plan coordinates in mm.
- **Homographies map raw pixels to floor-plan metres.** Store floor_plan_id, image dimensions, residuals, and quality with every matrix.
- **Never compare points from different floor plans.** Hard reject with metrics.
- **Unknown geometry is not perfect geometry.** No geometry score of 1.0 for uncalibrated cameras. Use neutral scoring or reject.
- **Attach projection once.** Project before tracking. Carry the same point into tracklets, trajectories, and proto events.
- **Ground-plane homography is not person height.** Do not claim physical height from bbox projection without a documented geometry model.
- **Measurement uncertainty is anisotropic and centralized.** Per-observation covariance is
  `R = J·Σ_px·Jᵀ + R_cal` in m², computed only in `app/tracking/world/observation_model.py`.
  No Jacobian or covariance math anywhere else. See the cts-spatial-fusion skill.
- **Fuse random noise, never systematic bias.** Inverse-covariance fusion shrinks covariance
  ~1/N; apply it only to the random term and add a non-shrinking bias floor from calibration
  residuals. Fusing the bias term away makes the estimate jump on camera-set changes.

### Identity evidence and gallery governance

For the complete authority, provenance, correction, and ReID lifecycle contract, load
`/home/sriram/code/nanai/continuous-tracking/.claude/skills/cts-identity-governance/SKILL.md`.
That focused skill is normative for identity work; this section remains the architectural summary.

- **Evidence provenance is mandatory.** Source, confidence, quality, model version, timestamp, and tracklet identifier on every record.
- **Direct face evidence is distinct.** Direct ArcFace, propagated hints, operator corrections, and ReID matches are separate evidence source types with different weights.
- **Temporal prior cannot create identity.** It maintains an existing identity within a bounded window; new assignment requires sensory or operator evidence.
- **Duplicate active identity is a guarded invariant.** `IdentityResolver.resolve()` owns the occupancy check over all open PHs (`open_ph_identities` carries incumbents not observed this frame). A second active PH must not newly acquire an identity already held by another open PH unless it has strong direct recognized face evidence for that identity. Shadow mode reports `cts_identity_shadow_mismatch_total{feature="duplicate_active_identity"}` before enforcement.
- **Gallery labels are governed.** Tentative commits → quarantine → promotion after stable evidence, not immediate trust.
- **Revisions are idempotent.** Stable IDs and evidence summaries so CC can apply once and explain in the UI.

### Clinical and dementia signals

- **Signals are not diagnoses.** Vision-derived patterns are behavioral alerts and caregiver context, not a diagnosis of any medical condition. Every signal carries a non-diagnostic disclaimer.
- **Every detector has an algorithm spec.** Name, version, evidence grade, required inputs, data quality minimums, thresholds, and cited rationale.
- **Gate on data quality.** Identity confidence, observation coverage, baseline sample count, and calibration availability before emitting or escalating.
- **Cold starts are conservative.** Without a resident baseline, cap severity. Absolute safety thresholds require caregiver opt-in.
- **Use elapsed time, not frame count.** Stillness and dwell computations accumulate seconds from timestamps, capping large inter-observation gaps.
- **Prefer room taxonomy over name matching.** Configured room types from CC. Name-substring fallback must be marked in context with reduced confidence.

### Cross-service contracts

- **Proto changes are two-repo changes.** Update bindings, subscribers, publishers, tests, and docs in both repos in the same change set.
- **Do not reuse proto field numbers.** Deprecated fields are reserved. New fields get new numbers. Keep compatibility readers during migration.
- **Wire-field rename discipline.** When a serialized field is renamed (preferred: finalize the rename in the `.proto` file so the generated `_pb2.py` uses the new name), update the proto, regenerate bindings in both repos (`make proto-py`), and update every producer and consumer in the same change set. After the rename the deprecated field name must not appear anywhere in source code except: (a) `reserved` declarations in the proto, (b) `Detection.global_track_id` which remains as a deprecated wire alias for older orchestrators until a dedicated cleanup change. No business logic, no second module, and no published message may read or emit a deprecated field name once the rename is finalized. Enforce with a name-contract test (`tests/contracts/test_ph_contract_names.py`). The approved-boundary list in that test is the authoritative registry of permitted legacy references.
- **Decode-boundary rule.** A wire field may carry a deprecated name during a rename, but it must be decoded into its PH-native name at exactly one boundary, and no business logic may read the deprecated name. After decoding, only the canonical name propagates. The `test_ph_contract_names.py` enforcement verifies that no non-approved module reads or emits a deprecated field name.
- **Redis Streams stay protobuf-only.** No JSON, no base64, no dual-codec.
- **Shadow before authority.** New tracking, identity, or signal algorithms run in shadow mode with mismatch metrics before becoming authoritative.

### Cross-camera dedup pattern

The pre-association floor-point dedup pass (`app/tracking/world/dedup.py`) runs before `associate()` inside `WorldTracker.step()`. Rules:

1. **Different-camera gate.** Only pairs from different cameras are candidates; same-camera pairs are always separate observations.
2. **Geometric gate.** Both detections must have `calibrated=True` floor points within `dedup_max_distance_m` (default: 0.6 m) of each other.
3. **No-face-conflict gate.** When `dedup_require_no_face_conflict=True` (default), pairs where both detections carry committed and conflicting face anchors are not merged.
4. **Representative fusion is inverse-covariance, not crop-quality mean.** The cluster
   representative's floor point and covariance come from information-form fusion of the
   members' random covariances plus a bias floor (see cts-spatial-fusion). Crop quality is no
   longer the fusion weight (it remains an input to Σ_px scaling only).

The union-find algorithm identifies connected components; each cluster produces a representative
observation with a fused floor point and covariance. `associate()` keeps its 1-to-1 contract
because only representatives enter it; cluster membership is passed separately so the winning PH
can be updated with all contributing camera IDs.

Adding a new dedup gate condition: modify `dedup_observations()` only; do not add gate logic to `associate()` or `WorldTracker.step()`.

`WorldTracker.step()` must receive the same-time observations from overlapping cameras in a single call. If the frame pipeline batches detector inference across cameras but then runs world tracking per camera, cross-camera dedup is bypassed and duplicate unknown PHs will grow quickly. Keep post-detection batching intact through `WorldTrackingStage`, then split only downstream side effects that are camera-local.

PH keyframes are evidence, not synthetic references. `get_keyframes()` reads `continuous_tracking.tagged_keyframes` and returns real `minio_key` values. Never fabricate object keys from observation timestamps; broken images make caregiver correction unsafe.

### Quality-capture rule

Observation and PH quality is computed by the single `CropQuality` scorer (`app/pipeline/crop_quality.py`). There is no second scorer. The same scorer is used for:
- Per-observation quality scores that scale `Σ_px` in the observation model.
- The PH's `mean_quality` field (exponential moving average, updated in `WorldTracker.step()`).
- Gallery entry quality used by the identity resolver.

`mean_quality` travels to CC via the `IdentitySnapshot` proto field and then through the CC's `PersonLocationEnvelope` as the `quality` field. Do not compute quality on the CC side; read it from the proto.

### No-silent-fallback rule for stream consumers

Stream consumers that cannot process a message must:
1. XACK the message to keep the pending-entry list clean.
2. Log at `warning` level with the error details.
3. Increment the `cts_messages_dead_lettered_total` Prometheus counter.

Never silently drop a message or return a fabricated value. The failure path must be observable.

Ruff `BLE001` (blind-except) is enabled at `error` severity in `pyproject.toml`. The allowlist of permitted catch-all sites is defined there; every new `except Exception` clause outside the allowlist is a CI failure.
