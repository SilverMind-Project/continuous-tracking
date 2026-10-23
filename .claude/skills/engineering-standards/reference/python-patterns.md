# Python patterns: SOLID, DI, immutability, imports, typing, async, errors

Read before writing or reviewing Python in `tracking-orchestrator/`. Part of the `engineering-standards` skill; the rules summary is in `../SKILL.md`.

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

## Dependency Injection

### Constructor injection only

Every dependency is an explicit constructor parameter. No service locators and no `getattr(self, "_foo", None)` fallbacks. A genuinely optional dependency is a keyword argument defaulting to `None` (next section); code that skips work because it is absent logs that once at warning.

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

### Declare all attributes in `__init__`

Every attribute is present in `__init__` with its correct `T | None` type, even if set later. Attributes first assigned outside `__init__` cause `AttributeError` and mypy cannot track them.

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

`FrozenInstanceError` is a correctness guard. Use `dataclasses.replace()` to create new instances with updated fields. Never use `object.__setattr__` to bypass the freeze -- if you need mutability, the type should not be frozen. For transport metadata alongside a frozen domain object (for example Redis message IDs), keep a side-channel `dict[int, T]` keyed by `id(obj)`.

Never re-construct a frozen domain object field-by-field to change a subset. Use
`dataclasses.replace(obj, **changed)` listing only the changed fields. Each mutation site's
changed-field set must be covered by the field-carryover test
(`tests/unit/tracking/world/test_ph_field_carryover.py` for `PersonHypothesis`); adding a
dataclass field requires classifying it there. Hand-copying every field once let five sites
silently drop `last_independent_identity_evidence_at`: a new field with
a default is silently reset to its default at every reconstruction site the author forgets, and
the failure mode is a wiped identity mid-track, not a test failure.

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
def combine_posteriors(
    prior: PosteriorDist,
    face: PosteriorDist,
    reid: PosteriorDist,
) -> PosteriorDist:
    ...

# WRONG -- reads self._config, self._identities, calls self._gallery_repo
def _from_gallery(self, entity: IdentityResolvableEntity) -> PosteriorDist:
    ...
```

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

Imports are absolute (`from app.pipeline.stages import ...`); only `__init__.py` re-export blocks may stay relative. New code uses absolute imports; the existing relative imports are being converted, and ruff `TID252` is enabled when they are gone. A `..pipeline` import from inside `app/tracking/` is also a layering hazard (see Circular Import Prevention).

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

## Error Handling

### Python

1. Domain errors use the project's `ErrorCode` enum. Never return or raise bare `Exception` for domain failures.
2. Stack traces never appear in API responses. Log the trace locally (`structlog.exception`); return only the error code and a human-readable message.
3. Async frame processing lets exceptions propagate to the `_consume_loop` handler, which logs and retries. Individual frames are never silently dropped -- failed frames produce a `FrameResponse` with `success=False` and an error code.
4. Model failures degrade per model: detector failure fails the frame; ReID/pose/face-ID failure degrades to missing evidence without dropping the frame.

## Tracking / data structures

**Use `.items()` when you need both key and value.** Build `active_tracks` with `[(key, val) for key, val in d.items() if …]` and return keys from assignment functions. `enumerate(dict.values())` gives positions that diverge from key names after any deletion.

**Explicit reverse maps for cross-namespace lookups.** When two modules use different ID namespaces for the same entity (e.g., `local_track_id` strings vs. UUID `tracklet_id`s), maintain an explicit `dict[local_id, uuid_id]` reverse map on the owning object.

**One increment per code path.** An unconditional increment followed by a conditional second increment silently doubles the count on the non-threshold branch.

**Prefer method parameters over placeholder fields.** If a helper receives `embedding` as a parameter and the detection also has a placeholder `detection.embedding`, use the parameter -- it carries the live Triton value.

**Sort trajectory windows ascending.** Dementia signal detectors need time-ordered points to compute distances and direction changes correctly.
