# Continuous Tracking System

Multi-camera person tracking and dementia signal detection for senior care. Processes RTSP camera feeds through an ML pipeline (YOLO detection, BoT-SORT tracking, Bayesian identity resolution, RTMPose posture estimation) and detects clinically relevant behavioral patterns: pacing, sundowning, bathroom dwell anomalies, prolonged stillness, nighttime movement, and unexplained absence.

Results are streamed via Redis Streams to [Cognitive Companion](../cognitive-companion), the BFF gateway that serves the Vue admin UI and MCP tools.

**Documentation:** [silvermind-project.github.io](https://silvermind-project.github.io)

## Services

| Service | Port | Description |
| --- | --- | --- |
| `go2rtc` | 1984 | RTSP proxy sidecar |
| `rtsp-ingress` | 8090 | Go RTSP ingest: camera registration, motion gating, MinIO upload |
| `tracking-orchestrator` | 8000 | Python ML pipeline: detection, tracking, identity, signals |
| `triton` | 8701 | ONNX model serving (YOLO, REID, RTMPose, CLIP, Florence-2) |

## Quick start

```bash
docker compose -f ../docker-compose.db.yml -p nanai up -d  # shared Postgres
cp rtsp-ingress/config/settings.yaml rtsp-ingress/config/settings.local.yaml
docker compose up -d
```

See the [documentation site](https://silvermind-project.github.io) for camera setup, model export, and operations guides.

## Development

```bash
cd tracking-orchestrator && uv sync --frozen --extra dev
make check          # ruff + mypy + import-linter + pytest
make all-check      # Python + Go + buf lint
```

## License

AGPL-3.0-or-later
