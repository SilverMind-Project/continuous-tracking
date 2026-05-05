# Continuous Tracking System (CTS)

A real-time monitoring system for seniors with early dementia. CTS watches RTSP camera streams, tracks individuals across multiple cameras, detects dementia-relevant behavioural patterns, and surfaces alerts through the [cognitive-companion](../cognitive-companion) BFF.

## What it does

- **Pulls RTSP streams** from IP cameras (overhead, eye-level, doorway) and uploads JPEG keyframes to object storage
- **Tracks people** frame-to-frame using BoT-SORT with Kalman filtering and appearance embeddings
- **Re-identifies individuals** across cameras using a Bayesian posterior over identity candidates, combining face recognition (ArcFace) and body appearance (SOLIDER-REID)
- **Detects dementia signals**: pacing, sundowning, bathroom anomaly, nighttime movement, prolonged stillness, unexplained absence
- **Streams results** via Redis Streams to the cognitive-companion gateway which exposes them through a Vue admin UI

## Architecture

```text
IP Cameras (RTSP)
       │
       ▼
 rtsp-ingress (Go)          ─── frames.ready stream ──▶  tracking-orchestrator (Python)
  ┌─────────────────┐                                      ┌──────────────────────────────┐
  │ Motion gating   │                                      │  YOLO26L (person detection)  │
  │ JPEG encoding   │                                      │  SOLIDER-REID (body embeds)  │
  │ MinIO upload    │                                      │  RTMPose (pose estimation)   │
  └─────────────────┘                                      │  Bayesian identity resolver  │
                                                           │  Dementia signal worker      │
                                                           └──────────────────────────────┘
                                                                  │  tracking.events
                                                                  │  tracking.revisions
                                                                  │  tracking.signals
                                                                  ▼
                                                     cognitive-companion (Python/FastAPI)
                                                       ┌──────────────────────────────┐
                                                       │  BFF gateway (all UI traffic)│
                                                       │  WebSocket live view         │
                                                       │  Admin UI (Vue 3)            │
                                                       │  MCP tools for AI agents     │
                                                       └──────────────────────────────┘
```

**Infrastructure**: TimescaleDB + pgvector · Redis Streams · MinIO · Triton Inference Server (NVIDIA or Intel Arc)

## Services

| Service                 | Port        | Description                                                        |
| ----------------------- | ----------- | ------------------------------------------------------------------ |
| `rtsp-ingress`          | 8090        | RTSP pull, motion gating, JPEG upload, `frames.ready` publisher    |
| `tracking-orchestrator` | 8000        | ML inference, tracking, identity resolution, signal detection      |
| `triton`                | 8001 (gRPC) | YOLO26L, SOLIDER-REID, RTMPose model server                        |
| `redis`                 | 6379        | Redis Streams transport (AOF enabled)                              |
| `postgres`              | 5432        | TimescaleDB + pgvector (tracklets, gallery, signals, trajectories) |
| `minio`                 | 9000        | JPEG keyframe object storage                                       |
| `cognitive-companion`   | 8080        | BFF gateway, Vue admin UI, WebSocket live view                     |

## Getting started

### Prerequisites

- Docker Compose v2
- A GPU: **NVIDIA** (CUDA) or **Intel Arc** (OpenVINO) — see Model setup section below
- RTSP cameras on the local network

### 1. Start infrastructure

```bash
docker compose up -d postgres redis minio
```

### 2. Configure cameras

Copy the example config and edit with your camera details:

```bash
cp rtsp-ingress/config/settings.yaml rtsp-ingress/config/settings.local.yaml
```

Create a `.env` file **in the same directory** for secrets (never commit this):

```bash
# rtsp-ingress/config/.env
CAM_BEDROOM_PASSWORD=your_camera_password
CAM_KITCHEN_PASSWORD=another_password
```

Then configure cameras in `settings.local.yaml` (see [Camera configuration](#camera-configuration) below).

### 3. Start services

```bash
docker compose up -d
```

### 4. Enable CTS in cognitive-companion

```yaml
# cognitive-companion/config/settings.yaml
cts:
  enabled: true
```

Then restart cognitive-companion. This starts the Redis Stream subscribers and exposes the admin UI at `http://localhost:8080/admin`.

## Camera configuration

Cameras can be configured statically in `rtsp-ingress/config/settings.yaml` or dynamically via the cognitive-companion Admin API. Static config is useful for development.

### Option A — host + credentials (URL built automatically)

```yaml
cameras:
  - id: cam-bedroom-1
    host: 192.168.1.100
    port: 554              # default: 554
    username: admin
    password: ${CAM_BEDROOM_PASSWORD}   # resolved from .env
    stream_path: /stream1
    type: overhead         # overhead | eye_level | doorway
    room_name: bedroom
    enabled: true
```

### Option B — explicit RTSP URL

```yaml
cameras:
  - id: cam-kitchen
    rtsp_url: rtsp://admin:${CAM_KITCHEN_PASSWORD}@192.168.1.101:554/h264/ch1/main/av_stream
    type: eye_level
    room_name: kitchen
    enabled: true
```

### .env file for secrets

Place a `.env` file alongside your config file (defaults to `rtsp-ingress/config/.env`):

```bash
CAM_BEDROOM_PASSWORD=s3cr3t
CAM_KITCHEN_PASSWORD=another_secret
MINIO_SECRET_KEY=prodkey
```

The service loads this file at startup and expands `${VAR}` placeholders in the YAML before parsing. Values already set in the process environment take precedence over the `.env` file. Set `RTSP_DOTENV_PATH` to override the default location.

### Supported camera fields

| Field                | Type   | Description                                                  |
| -------------------- | ------ | ------------------------------------------------------------ |
| `id`                 | string | Unique camera ID (used in all metrics and streams)           |
| `rtsp_url`           | string | Full RTSP URL (takes precedence over host/port/credentials)  |
| `host`               | string | Camera IP or hostname                                        |
| `port`               | int    | RTSP port (default: 554)                                     |
| `username`           | string | RTSP auth username                                           |
| `password`           | string | RTSP auth password (use `${VAR}` placeholder)                |
| `stream_path`        | string | URL path (default: `/`)                                      |
| `type`               | string | Camera placement: `overhead`, `eye_level`, `doorway`         |
| `room_name`          | string | Room this camera covers                                      |
| `enabled`            | bool   | Set `false` to temporarily disable without removing          |
| `frame_interval_ms`  | int    | Min ms between captured frames (default from `defaults`)     |
| `motion_threshold`   | float  | Fraction of pixels that must change to pass motion gate      |
| `reconnect_backoff_s`| float  | Initial reconnect backoff (doubles with jitter, max 60 s)    |

## Model setup

Model binaries are not in git. Both NVIDIA and Intel Arc GPUs are supported.

**Step 1 — select GPU vendor config** (run once per machine):

```bash
python triton-models/scripts/configure_gpu.py --vendor nvidia   # NVIDIA (default)
python triton-models/scripts/configure_gpu.py --vendor intel    # Intel Arc
```

**Step 2 — export model weights** (same ONNX format for all vendors):

```bash
pip install ultralytics>=8.4.0

# YOLO26L person detector → ONNX
python triton-models/scripts/export_yolo.py \
    --weights yolo26l.pt \
    --out triton-models/person-detector/1/model.onnx

# SOLIDER-REID body embedder → ONNX
python triton-models/scripts/export_reid.py --help

# RTMPose-m pose estimator → ONNX
python triton-models/scripts/export_pose.py --help
```

See [triton-models/README.md](triton-models/README.md) for full instructions including output shape verification and Intel Arc container image.

## Development

### Python (tracking-orchestrator)

```bash
cd tracking-orchestrator
uv sync --extra dev          # install deps into .venv

make check                   # ruff + mypy + pytest (from repo root)
make lint                    # ruff check only
make mypy                    # type check only
make test                    # pytest only
```

All tests run without a GPU — Triton calls are mocked via `TritonClientProtocol`.

### Go (rtsp-ingress)

```bash
cd rtsp-ingress
go test ./... -race           # full test suite with race detector
go vet ./...
golangci-lint run
```

### Full quality gate

```bash
make all-check               # Python + Go + proto buf lint
```

### Pre-commit hooks

```bash
pre-commit install            # ruff, mypy, golangci-lint, buf
```

## Key design decisions

| Decision        | Choice                                    | Rationale                                                                        |
| --------------- | ----------------------------------------- | -------------------------------------------------------------------------------- |
| Identity model  | Bayesian posterior, not single-assignment | Seniors with dementia have irregular gait; hard thresholds misidentify too often |
| Transport       | Redis Streams with consumer groups + XACK | At-least-once delivery with replay; survives orchestrator restarts               |
| Wire format     | Protobuf (no JSON on streams)             | ~3× smaller payloads; schema-enforced contracts                                  |
| Storage         | TimescaleDB + pgvector HNSW               | Time-series compression for trajectories; vector search for ReID gallery         |
| Person detector | YOLO26L NMS-Free, ONNX format             | Single ONNX file runs on NVIDIA (TRT EP) and Intel Arc (OpenVINO EP)             |
| UI gateway      | cognitive-companion as BFF                | No direct browser access to CTS internal services; single auth boundary          |

## Milestone status

| Milestone | Scope                                                | Status      |
| --------- | ---------------------------------------------------- | ----------- |
| M1        | Proto contracts, repos, docker-compose, CI           | Complete    |
| M2        | rtsp-ingress Go service                              | Complete    |
| M3        | Triton models + benchmark harness                    | Complete    |
| M4        | Tracking orchestrator skeleton                       | Complete    |
| M5        | Identity resolution + retroactive revision           | Complete    |
| M6        | Trajectories, room dwells, keyframes                 | Complete    |
| M7        | Admin UI (cameras, calibration, privacy, adjacency)  | Complete    |
| M8        | Dementia signals, dashboard                          | Complete    |
| M9        | Live view, identity corrections, runtime integration | Complete    |
| M10       | K8s manifests, observability, proto wire migration   | In progress |

## Repository layout

```text
.
├── rtsp-ingress/              Go RTSP ingest service
│   ├── config/settings.yaml  Default configuration
│   ├── internal/config/       Config loading + .env + placeholder expansion
│   ├── internal/rtsp/         RTSP session management and frame capture
│   ├── internal/motion/       Motion gating (pixel diff)
│   └── internal/media/        MinIO upload + Redis publish
├── tracking-orchestrator/     Python ML service
│   ├── app/inference/         Triton gRPC client + YOLO26L/ReID/Pose wrappers
│   ├── app/tracking/          BoT-SORT, identity resolver, cross-camera assoc
│   ├── app/trajectory/        Trajectory writer + dementia signal detectors
│   └── app/transport/         Redis Streams codec (protobuf)
├── triton-models/             Triton model configs + export scripts
├── proto/                     Protobuf contracts (frame, tracking, signals, scene)
├── cognitive-companion/       BFF gateway, Vue admin UI, MCP tools
├── k8s/                       Kubernetes manifests
└── docs/                      Runbook, wire-format spec
```

## Docs

- [docs/runbook.md](docs/runbook.md) — operational runbook (incidents, capacity, DR)
- [docs/wire-format.md](docs/wire-format.md) — Redis Stream message contracts
- [triton-models/README.md](triton-models/README.md) — model export + Triton verification
- [k8s/README.md](k8s/README.md) — Kubernetes apply order + ops notes
- [proto/README.md](proto/README.md) — protobuf codegen workflow
