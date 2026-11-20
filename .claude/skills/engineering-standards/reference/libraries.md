# Approved libraries

Read before adding or replacing a dependency. Part of the `engineering-standards` skill; the rules summary is in `../SKILL.md`.

## Python (`tracking-orchestrator/pyproject.toml`)

| Library | Role |
| --- | --- |
| `fastapi` | HTTP router and dependency injection |
| `pydantic v2` | Validation at all external boundaries (HTTP, config, Redis payloads) |
| `asyncpg` | Async Postgres driver -- `$1..$N` positional params always |
| `redis[hiredis]` | Async Redis Streams client -- consumer groups + XACK |
| `aiobotocore` | Async S3-compatible client for MinIO |
| `structlog` | Structured JSON logging -- never `logging.getLogger` or `print` |
| `numpy` | Frame data and embedding arithmetic |
| `opencv-python-headless` | Image preprocessing -- `cv2.resize(INTER_LINEAR)` for all letterbox/crop resizing |
| `scipy` | Hungarian assignment (`linear_sum_assignment`) for tracking; robust statistics for baselines (MAD, EWMA) |
| `shapely>=2.0` | Polygon containment for privacy zone enforcement |
| `protobuf` | Message contracts; generated bindings committed in `app/proto/` |
| `prometheus-client` | Metrics exposition at `GET /metrics` |
| `pytest` + `pytest-asyncio` | Test runner; all repo tests are `async def` |
| `mypy` (strict) | Type checking on `domain/`, `storage/`, `services/`, `transport/` |
| `ruff` | Lint + format |
| `import-linter` | Layering enforcement (`make check`, CI, pre-commit) |
| `triton-shared` | Shared Triton client + YOLO/CLIP/Florence pre/post-processing (git source in `[tool.uv.sources]`, pinned by `uv.lock`) |

Do not introduce `aiohttp`, `celery`, or `psycopg2`. `httpx` is permitted as a **test-only** dependency in `tracking-orchestrator`.

## Python (`rf-ingress/pyproject.toml`)

| Library | Role |
| --- | --- |
| `fastapi` + `uvicorn` | Health (`/health`) and metrics (`/metrics`) HTTP surface |
| `pydantic v2` | Settings, HTTP site-config, and adapter payload profile validation |
| `httpx` | Async client polling CC `GET /api/v1/wearables/site-config` |
| `redis[hiredis]` | Async Redis Streams publisher (`decode_responses=False`, 72 h `MINID` trim) |
| `protobuf` | `TagFix` and `AdapterCapability` wire bindings generated into `rf_ingress/proto/` |
| `cts-contracts` | Shared wearable and RF-node closed vocabularies (`cts_contracts.wearables`) |
| `numpy` + `scipy` | Multilateration solver (`scipy.optimize.least_squares`), chi-square outlier detection, 2D similarity transform |
| `structlog` | Structured JSON logging |
| `prometheus-client` | Central metrics exposition |
| `pyyaml` | Settings and simulator scenario loading |
| `aiomqtt` | Async MQTT client for `device_ranging_mqtt` (`device_ranging.v1` wearable range batches) |
| `hypothesis` (dev) | Property tests for noise-free solver accuracy |
| `testcontainers` (dev) | Ephemeral `redis:7-alpine` container for `make rf-test-integration` |

## Kotlin / Gradle (`wear-companion/gradle/libs.versions.toml`)

| Library | Role |
| --- | --- |
| `kotlinx-coroutines-core` | Async burst lifecycle and engine flows in `:core` and `:app` |
| `kotlinx-serialization-json` | Strict `device_ranging.v1` JSON micro-batch encoding in `EpochBatchPublisher` |
| `junit-jupiter` (test) | JUnit 5 unit tests in `:core` (pure JVM, no Android SDK required) |
| `ktlint` + `detekt` | Formatting and static analysis (`make wear-lint`, `make wear-check`) |
| `androidx.wear.tiles` + `androidx.wear.protolayout` | Wear OS companion status tile (`:app`) |
| `play-services-nearby` / `androidx.core:core-uwb` | FiRa 2.0 UWB controller engines (`dualMode` flavor only) |
| `org.eclipse.paho.client.mqttv3` | LAN-only MQTT batch transport (`dualMode` flavor only) |

## Go (`rtsp-ingress/go.mod`)

| Library | Role |
| --- | --- |
| `minio/minio-go/v7` | S3-compatible object storage client |
| `redis/go-redis/v9` | Redis Streams producer |
| `prometheus/client_golang` | Metrics exposition |
| `go.uber.org/zap` | Structured logging -- `zap.L()` for global logger |
| `google.golang.org/protobuf` | Protobuf runtime |
| `golang.org/x/image` | Image format decoding in media processing |
| `gopkg.in/yaml.v3` | Config file parsing |

Do not add `bluenviron/gortsplib` or `pion/rtp` -- RTSP is delegated to go2rtc.
