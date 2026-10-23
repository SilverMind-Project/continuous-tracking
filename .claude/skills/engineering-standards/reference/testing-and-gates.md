# Testing, quality gate, and review checklist

Read before writing tests or declaring a change done. Part of the `engineering-standards` skill; the rules summary is in `../SKILL.md`.

## Testing Strategy

### Python

1. **Test at the Protocol boundary.** Inject `InMemory*` implementations directly. Never mock `asyncpg.Pool`, Redis connections, or HTTP clients in unit tests.
2. **All repository tests are `async def`.** Configured via `asyncio_mode = "auto"` in `pyproject.toml`.
3. **Integration tests in `tests/integration/`.** Marked `integration`; `make test-integration` (part of `make ci`) runs them against testcontainers it starts itself (Docker required).
4. **Shared fixtures via `pytest.fixture`.** Build `CameraConfig`, `Detection`, `Tracklet` once and reuse. Avoid reconstructing the same frozen dataclass inline across tests.
5. **Tests assert behavior through public APIs.** Never assert on private `_attributes`. Tests that reach into private state are brittle and test implementation details, not contracts.
6. **Triton calls mocked via `TritonClientProtocol`.** All inference tests run without a GPU.
7. **Falsifiable assertions.** For signal detectors: "7-day TV-watching fixture produces zero stillness_anomaly", not just "some signal was produced". Assert counts, severity levels, and non-emissions.
8. **Hysteresis tests must be explicit.** One-run tests set `onset_consecutive_windows=1`. Full hysteresis tests use the default of 2 and verify onset debounce, cooldown, and escalation independently.

### What NOT to test

- Do not test Python dataclass constructors (the language tests field assignment).
- Do not test third-party library behavior (Redis xadd, protobuf serialization).
- Do not test trivial property accessors or getters.
- Do not write tests that exist only to hit a coverage percentage -- every test must document a real failure mode.

## Quality Gate

```bash
make check   # Python: ruff check + ruff format --check + mypy + import-linter + pytest
make check-all   # Python + Go (golangci-lint + go test -race + go build) + buf lint
make ci      # check-all + make test-integration (every integration-marked test)
```

CI's `python-test` job runs `make test` and `make test-integration`, the same targets as the
local gate. Tests start their own `timescaledb-ha:pg18` testcontainer and apply migrations
through `MigrationRunner`; no test reads `DATABASE_URL`. Pre-commit hooks (`make install-hooks`)
call the project venv and the pinned Go and buf toolchain, never separately pinned copies.

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

## Standards at a glance

| Area | Standard |
| --- | --- |
| Python version | 3.12; `from __future__ import annotations` in every file |
| Type checking | `mypy --strict` on `domain/`, `storage/`, `services/`, `transport/` |
| Go version | 1.24+ (toolchain pinned in `.tool-versions`; install with `make go-install`) |
| Go safety | `-race` mandatory in all test runs; `go vet`; `golangci-lint` |
| Protobuf | `buf lint`; `buf breaking` against last merged commit; generated bindings committed |
| Logging | `structlog` in Python (`get_logger(__name__)`); `zap.L()` in Go. Include `camera_id`, `ph_id` in every relevant log line. |
| Timestamps | Always timezone-aware. Python: `datetime.now(UTC)`. Go: `time.Now().UTC()`. |
| Image resize | `opencv-python-headless` `cv2.resize(INTER_LINEAR)`. Never hand-roll bilinear in NumPy. |
| Secrets | In environment variables or `.env` files (never committed). YAML holds defaults; env overrides. |
| SQL injection | asyncpg `$N` params only. Never build SQL with f-strings or `%`. |
| Error taxonomy | Domain errors use the project's `ErrorCode` enum. No stack traces in API responses -- log locally, return only code + message. |
| Metrics | Prometheus via `prometheus-client` (Python) and `prometheus/client_golang` (Go). Register in the respective `metrics.py` / `metrics.go` files. |
