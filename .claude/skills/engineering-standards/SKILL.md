---
name: engineering-standards
description: Use when changing CTS architecture, repositories, configuration, tests, wire formats, units, caches, or security boundaries. For domain requirements and quality gates, see CLAUDE.md.
---

# Engineering Standards

The rules for every change in `continuous-tracking`. Each rule is one line; the detail,
examples, and rationale live in `reference/`. Domain skills (`cts-pipeline`, `cts-signals`,
`cts-spatial-fusion`, `cts-identity-governance`) are normative for their areas. Read a reference
file when its trigger applies:

| Read | When |
| --- | --- |
| `reference/python-patterns.md` | Writing or reviewing Python: SOLID, DI, immutability, imports, typing, async, errors, data-structure pitfalls |
| `reference/storage.md` | A repository, a SQL statement, or a migration |
| `reference/operations.md` | A setting or flag, an endpoint, a log line, a metric, time handling, image code, the venv |
| `reference/testing-and-gates.md` | Writing tests, ruff rules, the review checklist, declaring done |
| `reference/domain-principles.md` | Units, hardware adapters, caches, stages, calibration, identity, signals, cross-service contracts, dedup, quality, stream consumers, the wire format |
| `reference/go.md` | Code under `rtsp-ingress/` |
| `reference/libraries.md` | Adding or replacing a dependency |

This skill documents the current codebase. Update it in the same PR that changes a behavior or
contract it describes.

## Gates and data safety (check these first)

1. `make check` while iterating; `make ci` before declaring done (it adds Go, buf, and every
   integration-marked test against a testcontainer). Never deselect, skip, or weaken a test.
2. No test reads `DATABASE_URL` or touches a live database; integration tests use the session
   testcontainer, which applies every migration through `MigrationRunner`.
3. Run Python only through `tracking-orchestrator/.venv`; never system Python or `pip`. Format
   only touched files.
4. A behavior change and a refactor never share a commit.
5. A gate that enumerates a collection carries a non-vacuity canary; a shrinking population is a
   defect.

## Architecture and wiring

6. Layering `core → domain → storage → tracking.world → services → transport → routers` (import-linter). Nothing
   above `storage` imports a concrete `Postgres*` class.
7. One responsibility per module, stage, and service; services coordinate, detectors and pure
   functions own algorithms. New detectors and stages are new classes, never `elif` branches.
8. Every dependency is an explicit constructor parameter; a genuinely optional one is a keyword
   argument defaulting to `None`, and skipping work because it is absent logs once at warning.
9. No service locators, no `getattr(self, "_foo", None)` fallbacks, no new module-level
   singletons (`calibration_state` is the one tracked exception).
10. A repository is done only with all three artifacts (`Protocol`, `InMemory*`, `Postgres*`),
    explicit wiring in `app/main.py`, and an integration test on a production-shaped path.
11. A filter in a Postgres query appears identically in the `InMemory` peer and the `Protocol`
    signature with a shared default constant; new filtered reads get a parity test.
12. Code that exists only to be re-exported or tested is dead: delete it, with its config keys and
    plumbing, in the change that orphans it.

## Python

13. Domain objects are `@dataclass(frozen=True)`; change them with `dataclasses.replace(obj,
    **changed)`, never field-by-field reconstruction and never `object.__setattr__`. A new
    `PersonHypothesis` field is classified in `test_ph_field_carryover.py`.
14. All imports at the top; no conditional imports on runtime data; no import side effects; no
    wildcard imports. Imports are absolute (`from app.x import ...`); only `__init__.py` re-export
    blocks may be relative. Cross-boundary type-only imports go under `TYPE_CHECKING`.
15. `from __future__ import annotations` in every file; every signature annotated; `T | None`, never
    `Optional`; `# type: ignore[code]` only with a code; no `Any` in constructors or returns.
16. Declare every attribute in `__init__` with its type. Wrap numpy comparisons in `bool()` when
    returning `bool`.
17. Pydantic v2 at boundaries (HTTP, Redis payloads, config); frozen dataclasses internally; never
    leak a Pydantic model into services or repositories.
18. Background tasks are stored and cancelled in `stop()`; no blocking calls in async code;
    work over about 1 ms goes to `asyncio.to_thread()`.
19. Domain errors use the `ErrorCode` enum; no stack traces in API responses. Model failures
    degrade per model: a detector failure fails the frame, a ReID, pose, or face failure is missing
    evidence.
20. Broad `except Exception` only with a per-site `# noqa: BLE001` and a documented reason.

## Storage and SQL

21. asyncpg `$N` params only; schema-qualify every table as `continuous_tracking.<table>`;
    validate JSONB shape at the repository boundary (`.claude/rules/sql.md`).
22. Schema changes are new numbered `.up.sql` / `.down.sql` pairs on top of `0001_init`; the down
    file exactly reverses the up; never drop and recreate a database (`.claude/rules/migrations.md`).

## Configuration, API, observability

23. Configuration is typed frozen dataclasses validated at startup; secrets come from the
    environment, YAML holds defaults.
24. New behavior ships behind a `settings.yaml` flag defaulting to off, with shadow mode, a replay
    proof, and a guardrail test; the flag is removed one release after enablement.
25. Endpoints have explicit Pydantic request and response models and structured error `detail`
    codes; `/internal/*` endpoints are for the CC BFF only.
26. Log with `structlog` `get_logger(__name__)`, event name first, with `camera_id` and `ph_id`
    where known; never `print` or `logging.getLogger`; never log embeddings, face crops, images,
    secrets, or names.
27. Metrics register in `app/observability/metrics.py`; shadow computations are sample-gated with a
    rate defaulting to 0.0.
28. Timestamps are timezone-aware; pipeline code uses `ctx.capture_time` or `ctx.event_time`, and
    `datetime.now(UTC)` only for wall-clock side effects (`.claude/rules/time.md`).
29. Image resizing uses `opencv-python-headless` `cv2.resize(INTER_LINEAR)`; never hand-rolled.

## Domain and contracts

30. Every dimensional field names its unit (`_m`, `_m_s`, `_nu_s`, `_deg`, `_px`) and its frame;
    mixing frames is a review-blocking defect.
31. Hardware sits behind one adapter emitting a neutral contract; no vendor name past the adapter;
    capabilities are declared; every adapter family has a simulator.
32. An epoch-scoped cache names its epoch, is invalidated at one chokepoint every execution path
    crosses, and has a staleness backstop.
33. Measurement covariance is computed only in `app/tracking/world/observation_model.py`; dedup
    gates live only in `dedup_observations()`; quality comes only from `CropQuality`.
34. Redis Streams carry raw protobuf through `app/transport/codec.py` with
    `decode_responses=False`; a consumer that cannot process a message XACKs, logs at warning,
    and counts it.
35. Proto changes are two-repo changes; never reuse a field number; an optional scalar carries a
    presence flag; a renamed wire field is decoded at exactly one boundary
    (`tests/contracts/test_ph_contract_names.py`).
36. New tracking, identity, or signal algorithms run in shadow mode with mismatch metrics before
    they become authoritative. Signals are behavioral alerts, never diagnoses.

## Comments and docs

37. Comments explain intent, never milestone ids or status (`.claude/rules/comments.md`). No
    em-dashes in markdown (`make check-md`); instruction files name only files that exist
    (`make docs-check`).
