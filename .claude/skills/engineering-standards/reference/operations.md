# Configuration, API, security, time, logging, observability, environment

Read before adding a setting or flag, an endpoint, a log line, a metric, or image code. Part of the `engineering-standards` skill; the rules summary is in `../SKILL.md`.

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

### Deletion discipline

Code that exists only to be re-exported or only to be tested is dead; delete it in the same
milestone that orphans it. A milestone that removes a mechanism must remove its config keys,
reader plumbing, and dataclass fields in the same change ("removed" with surviving plumbing is a
recording defect). The `app.tracking.identity` export surface is pinned by
`tests/test_no_dead_exports.py`.

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

## Security at Boundaries

1. **Validate at entry points only.** Pydantic v2 for HTTP bodies, Redis payloads, and config files. Internal domain objects must already be valid when constructed.
2. **Parameterized SQL, always.** asyncpg `$1..$N` -- never f-strings or `%` formatting.
3. **No secrets in code.** Environment variables only. `.env` never committed.
4. **Protobuf for cross-service messages.** Bytes on the wire parse against a typed schema. No raw JSON blobs without Pydantic validation.
5. **`go vet` + `staticcheck` + `golangci-lint` for Go.** Hard CI gates.

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

Key fields (`camera_id`, `ph_id`, `event_id`) appear in every log line where they are known.

### Log levels

- `debug`: per-frame/per-detection detail, model outputs, rejection reasons. Verbose, sampled in production.
- `info`: lifecycle events (startup, shutdown, config), signal emissions, identity commits, revision publications.
- `warning`: degraded operation (model timeout, service unavailable, stale frame), recoverable errors.
- `exception`: unrecoverable per-frame failures with full traceback. Always includes `camera_id` and `frame_index`.

### What NOT to log

- Never log embedding vectors, face crops, or full image data.
- Never log secrets, tokens, or full connection strings.
- Never log PII (person names, face images) without a documented anonymization policy.

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

## Image Preprocessing

Use `opencv-python-headless` for all letterbox and crop resizing:

```python
import cv2

resized = cv2.resize(img, target, interpolation=cv2.INTER_LINEAR)
```

Never import `opencv-python` (GUI dependencies conflict with headless containers). Never hand-roll bilinear resize in NumPy -- it's slower and less correct.

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
