# Continuous Tracking System (CTS)

A real-time monitoring system for seniors with early dementia. CTS watches RTSP camera streams, tracks individuals across multiple cameras, detects dementia-relevant behavioural patterns, and surfaces alerts through the [cognitive-companion](../cognitive-companion) BFF.

## What it does

- **Pulls RTSP streams** via a go2rtc sidecar — rtsp-ingress registers cameras with go2rtc over HTTP and polls for pre-decoded JPEG frames; go2rtc owns all RTSP sessions
- **Gates on motion** to avoid uploading static frames; uploads JPEG keyframes to MinIO
- **Tracks people** frame-to-frame using BoT-SORT with Kalman filtering and appearance embeddings
- **Re-identifies individuals** across cameras using a Bayesian posterior over identity candidates, combining ArcFace face recognition and SOLIDER-REID body appearance
- **Detects dementia signals**: pacing, sundowning, bathroom anomaly, nighttime movement, prolonged stillness, unexplained absence
- **Streams results** via Redis Streams (protobuf wire format) to the cognitive-companion gateway which exposes them through a Vue admin UI

## Architecture

```text
IP Cameras (RTSP)
       │  RTSP sessions owned by go2rtc
       ▼
 go2rtc (sidecar, :1984)
  ┌──────────────────────┐
  │ RTSP session manager │◀── PUT /api/streams (register)      rtsp-ingress (Go, :8090)
  │ H264 → JPEG decode   │◀── GET /api/frame.jpeg (poll)       ┌──────────────────────────┐
  └──────────────────────┘                                      │ Motion gating            │
                                                               │ MinIO JPEG upload        │
                                                               │ frames.ready publisher   │
                                                               └──────────────────────────┘
                                                                         │ frames.ready (protobuf)
                                                                         ▼
                                                           tracking-orchestrator (Python, :8000)
                                                           ┌──────────────────────────────┐
                                                           │ YOLO26L  (person detection)  │
                                                           │ SOLIDER-REID (body embeds)   │──▶ Triton (:8001)
                                                           │ RTMPose  (pose estimation)   │
                                                           │ BoT-SORT tracker             │    YOLO26L · CLIP
                                                           │ Bayesian identity resolver   │    Florence-2
                                                           │ Dementia signal worker       │    SOLIDER-REID · RTMPose
                                                           └──────────────────────────────┘    (all INT8 ONNX)
                                                                         │ tracking.events
                                                                         │ tracking.revisions
                                                                         │ tracking.signals
                                                                         ▼
                                                            cognitive-companion (Python/FastAPI, :8080)
                                                            BFF gateway · WebSocket live view
                                                            Vue 3 admin UI · MCP tools
                                                            ┌──────────────────────────────┐
                                                            │ scene-analysis-service (:8300)│──▶ Triton (:8001)
                                                            │ YOLO26L · CLIP · Florence-2  │
                                                            └──────────────────────────────┘
```

**Infrastructure**: TimescaleDB + pgvector · Redis Streams (AOF) · MinIO · Triton Inference Server (NVIDIA or Intel Arc)

## Services

| Service | Port | Description |
| --- | --- | --- |
| `go2rtc` | 1984 | RTSP proxy sidecar — owns all camera sessions, serves JPEG frames over HTTP |
| `rtsp-ingress` | 8090 | Registers cameras with go2rtc, polls frames, motion gating, MinIO upload |
| `tracking-orchestrator` | 8000 | ML inference, tracking, identity resolution, signal detection |
| `triton` | 8001 (gRPC) | YOLO26L, CLIP ViT-L/14, Florence-2, SOLIDER-REID, RTMPose (all INT8 ONNX) |
| `scene-analysis-service` | 8300 | Scene analysis (shares Triton with CTS) |
| `redis` | 6379 | Redis Streams transport (AOF enabled) |
| `postgres` | 5432 | TimescaleDB + pgvector (tracklets, gallery, signals, trajectories) |
| `minio` | 9000 | JPEG keyframe object storage |
| `cognitive-companion` | 8080 | BFF gateway, Vue admin UI, WebSocket live view |

---

## Getting started

### Prerequisites

- Docker Compose v2
- A GPU: **NVIDIA** (CUDA / TensorRT) or **Intel Arc** (OpenVINO) — see [Model setup](#model-setup)
- RTSP cameras on the local network

### 1. Start infrastructure

```bash
docker compose up -d postgres redis minio
```

### 2. Configure cameras

See [Camera configuration](#camera-configuration) below. For a quick start, copy the example config:

```bash
cp rtsp-ingress/config/settings.yaml rtsp-ingress/config/settings.local.yaml
```

### 3. Start all services

```bash
docker compose up -d
```

### 4. Enable CTS in cognitive-companion

```yaml
# cognitive-companion/config/settings.yaml
cts:
  enabled: true
```

Restart cognitive-companion. This starts the Redis Stream subscribers and exposes the admin UI at `http://localhost:8080/admin`.

---

## Camera configuration

All cameras are managed through the cognitive-companion Admin UI. There is no static camera list in config files.

### How cameras reach go2rtc

rtsp-ingress polls `GET /api/v1/cts/cameras` on cognitive-companion every 60 s and reconciles the running set of streams:

```text
  cognitive-companion database (cts_cameras table)
         │
         │  GET /api/v1/cts/cameras  (every 60 s)
         ▼
  rtsp-ingress reconciler
         │
         ▼ PUT /api/streams  (idempotent, heals go2rtc restarts)
  go2rtc
  - owns the RTSP session
  - decodes H264 to JPEG
         │
         ▼ GET /api/frame.jpeg  (polled every frame_interval_ms)
  rtsp-ingress poll worker
  - applies motion gate
  - uploads to MinIO
  - publishes to frames.ready stream
```

go2rtc's own config (`rtsp-ingress/config/go2rtc.yaml`) has no `streams:` section; all registrations are made at runtime via HTTP. go2rtc reconnects to cameras automatically if sessions drop.

### Adding cameras

1. Go to `http://localhost:8080/admin/cts/cameras` and click **Add Camera**.
2. Fill in the fields:

| Field | Required | Description |
| --- | --- | --- |
| `id` | yes | Stable slug, e.g. `kitchen-cam-1`. Used in all metrics and streams. |
| `name` | yes | Human-readable display name. |
| `rtsp_url` | yes | Full RTSP URL including credentials, e.g. `rtsp://admin:secret@192.168.1.10:554/stream1`. |
| `location` | no | Room or location label shown in the UI. |
| `enabled` | yes | Uncheck to suspend a camera without deleting it. |

3. rtsp-ingress picks up the new camera within one reconcile interval (default 60 s). Use the **Test RTSP** button to verify connectivity before saving.

### Service credentials

rtsp-ingress authenticates to cognitive-companion with the `CC_INGRESS_API_KEY` environment variable. Set this to a key that has the `cts_ingress` permission defined in `cognitive-companion/config/auth.yaml`.

```bash
# rtsp-ingress environment
CC_INGRESS_API_KEY=<generate a random secret>
COGNITIVE_API_KEY=${CC_INGRESS_API_KEY}

# cognitive-companion environment
CC_INGRESS_API_KEY=<same secret>
```

Camera RTSP credentials (username, password) are stored in the cognitive-companion database, not in any config file on the ingress side. The full RTSP URL travels over the internal API call from cognitive-companion to rtsp-ingress, where it is passed to go2rtc. TLS between the services is configured in `cognitive-companion/config/settings.yaml` under `cts.upstream.rtsp_ingress`.

### Default frame-capture settings

Camera-level overrides are not yet exposed in the UI. The service-wide defaults in `rtsp-ingress/config/settings.yaml` apply to every camera:

```yaml
defaults:
  frame_interval_ms: 500    # Min ms between captured frames
  motion_threshold: 0.02    # Fraction of pixels that must change
  reconnect_backoff_s: 2.0  # Initial reconnect backoff (doubles with jitter, max 60 s)
```

Override any default at deploy time with the environment variables `DEFAULT_FRAME_INTERVAL_MS`, `DEFAULT_MOTION_THRESHOLD`, or `DEFAULT_RECONNECT_BACKOFF_S`.

---

## Model setup

Model binaries are not in git. All models use **INT8 quantization** for
reduced size and faster inference. Both NVIDIA and Intel Arc GPUs are
supported through Triton's ONNX Runtime backend.

**Step 1 — select GPU vendor config** (run once per machine):

```bash
python triton-models/scripts/configure_gpu.py --vendor nvidia   # NVIDIA TensorRT/CUDA (default)
python triton-models/scripts/configure_gpu.py --vendor intel    # Intel Arc OpenVINO
```

**Step 2 — export / download models:**

```bash
# YOLO26L person detector (export + quantize)
uv run --with ultralytics --with torch --with onnx --with onnxruntime --with sympy \
    python triton-models/scripts/export_yolo.py --weights yolo26l.pt
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/person-detector/1/model.onnx \
    --output triton-models/person-detector/1/model_int8.onnx

# CLIP ViT-L/14 vision encoder (export + quantize)
uv run --with open_clip_torch --with torch --with onnx --with onnxruntime --with sympy \
    python triton-models/scripts/export_clip.py
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/clip-vision/1/model.onnx \
    --output triton-models/clip-vision/1/model_int8.onnx

# Florence-2-large (download INT8 from onnx-community)
uv run --with huggingface_hub \
    python triton-models/scripts/download_florence.py

# SOLIDER-REID body embedder
python triton-models/scripts/export_reid.py --help

# RTMPose-m pose estimator
python triton-models/scripts/export_pose.py --help
```

### Model inventory

| Model | Triton name | Format | Size (INT8) | Output shape |
|-------|-------------|--------|-------------|-------------|
| YOLO26L | `person-detector` | ONNX | 24 MB | [N, 300, 6] NMS-free |
| CLIP ViT-L/14 | `clip-vision` | ONNX | 293 MB | [N, 768] |
| Florence-2-large | `florence-2` | Python (ORT) | 794 MB | [1, max_len] |
| SOLIDER-REID | `reid-solider` | ONNX | — | [N, 768] |
| RTMPose-m | `pose-rtmpose` | ONNX | — | [N, 17, 384] + [N, 17, 512] |

### Shared with scene-analysis-service

The same Triton instance serves both CTS and `scene-analysis-service`.
YOLO26L, CLIP, and Florence-2 models are shared. The `triton-shared/` package
provides the common Triton client and inference utilities used by both services.

See [triton-models/README.md](triton-models/README.md) for output shape verification, benchmark targets, and the Intel Arc container image.

---

## Development

### Python — tracking-orchestrator

```bash
# Install dependencies into .venv (uses uv)
cd tracking-orchestrator
uv sync --frozen --extra dev

# From repo root:
make check          # ruff + mypy + import-linter + pytest (full Python gate)
make lint           # ruff check only
make format         # ruff format
make mypy           # type check only
make test           # pytest only
make import-lint    # layering check only
```

All tests run without a GPU — Triton calls are mocked via `TritonClientProtocol`. Always use the project venv at `tracking-orchestrator/.venv/`; the Makefile targets use it automatically.

### Go — rtsp-ingress

```bash
# Install the pinned Go toolchain (Go 1.24, stored in ./tools/)
make go-install

make go-check       # golangci-lint + go test -race + go build (full Go gate)
make go-test        # go test -race ./... only
make go-lint        # golangci-lint only
make go-build       # go build only
```

The race detector (`-race`) is mandatory — it is a hard CI gate.

### Protobuf

```bash
make proto          # Generate Go (buf generate) + Python (protoc) bindings
make proto-lint     # buf lint only
```

Requires `protoc >= 25` (`apt install protobuf-compiler`) and `make go-install`. Generated bindings are **committed** — do not gitignore them.

### Full quality gate

```bash
make all-check      # Python (ruff + mypy + import-linter + pytest) + Go (lint + test + build) + buf lint
```

Install pre-commit hooks (ruff, mypy, golangci-lint, buf):

```bash
pre-commit install
```

### Docker

```bash
make infra-up       # Start all infrastructure services (postgres, redis, minio, triton, go2rtc)
make app-up         # Build and start all services including rtsp-ingress and tracking-orchestrator
make docker-down    # Stop everything and remove volumes
```

---

## Key design decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| RTSP ingest | go2rtc sidecar + HTTP poll | Offloads session management and H264 decoding; rtsp-ingress polls `/api/frame.jpeg` instead of managing RTSP directly |
| Camera config → go2rtc | Dynamic HTTP registration (`PUT /api/streams`) | go2rtc needs no static config file; rtsp-ingress reconciles at runtime from settings.yaml or the cognitive-companion API |
| Identity model | Bayesian posterior, not single-assignment | Seniors with dementia have irregular gait; hard thresholds misidentify too often |
| Transport | Redis Streams with consumer groups + XACK | At-least-once delivery with replay; survives orchestrator restarts |
| Wire format | Protobuf (no JSON on streams) | ~3× smaller payloads; schema-enforced contracts |
| Storage | TimescaleDB + pgvector HNSW | Time-series compression for trajectories; vector search for ReID gallery |
| Person detector | YOLO26L NMS-Free, ONNX format, INT8 | Single ONNX file runs on NVIDIA (TRT EP) and Intel Arc (OpenVINO EP) |
| Model serving | Triton Inference Server, all models ONNX | 5 models across 2 services (CTS + SAS); GPU vendor config in Triton, not client code |
| Shared client | `triton-shared/` package | Common Triton gRPC client + pre/post-processing used by CTS and SAS |
| UI gateway | cognitive-companion as BFF | No direct browser access to CTS internal services; single auth boundary |

---

## Repository layout

```text
.
├── rtsp-ingress/              Go RTSP ingest service
│   ├── config/
│   │   ├── settings.yaml      Default config (cameras, redis, minio, go2rtc addr)
│   │   └── go2rtc.yaml        go2rtc config (API listen addr only — no static streams)
│   ├── internal/config/       Config loading + .env + ${VAR} expansion
│   ├── internal/go2rtc/       go2rtc HTTP API client (register, deregister, fetch JPEG)
│   ├── internal/motion/       Motion gating (pixel diff)
│   ├── internal/media/        MinIO upload + Redis Streams publish
│   ├── internal/metrics/      Prometheus metrics
│   ├── internal/poll/         Per-camera JPEG polling worker
│   ├── internal/reconciler/   Polls cognitive-companion API; calls Supervisor.Reconcile
│   ├── internal/streams/      Stream lifecycle bookkeeping
│   └── internal/supervisor/   Registers streams with go2rtc; starts/stops poll workers
├── tracking-orchestrator/     Python ML service
│   ├── app/domain/            Frozen dataclasses (Detection, Tracklet, GlobalTrack, …)
│   ├── app/inference/         Triton gRPC client + YOLO26L/ReID/Pose wrappers
│   ├── app/tracking/          BoT-SORT, identity resolver, cross-camera association
│   ├── app/trajectory/        Trajectory writer + dementia signal detectors
│   ├── app/transport/         Redis Streams codec (protobuf), publishers
│   ├── app/storage/           Repository protocols + InMemory + Postgres impls
│   ├── app/routers/           Internal FastAPI endpoints
│   ├── app/proto/             Generated protobuf Python bindings (committed)
│   └── migrations/            SQL migrations (0001–0005)
├── triton-models/             Triton model configs + export/download scripts
│   ├── person-detector/       YOLO26L ONNX (INT8)
│   ├── clip-vision/           CLIP ViT-L/14 ONNX (INT8)
│   ├── florence-2/            Florence-2-large Python backend (INT8)
│   ├── reid-solider/          SOLIDER-REID ONNX
│   ├── pose-rtmpose/          RTMPose-m ONNX
│   └── scripts/               export, download, quantize, configure_gpu
├── proto/                     Protobuf contracts (frame, tracking, signals, scene)
├── cognitive-companion/       BFF gateway, Vue admin UI, MCP tools
├── k8s/                       Kubernetes manifests
├── docs/                      Runbook, wire-format spec
└── ../triton-shared/          Shared Triton client + inference utilities
```

---

## Docs

- [docs/runbook.md](docs/runbook.md) — operational runbook (incidents, capacity, DR)
- [docs/wire-format.md](docs/wire-format.md) — Redis Stream message contracts
- [triton-models/README.md](triton-models/README.md) — model export + Triton verification
- [k8s/README.md](k8s/README.md) — Kubernetes apply order + ops notes
- [proto/README.md](proto/README.md) — protobuf codegen workflow
