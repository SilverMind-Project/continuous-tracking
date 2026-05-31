# Continuous Tracking System

Multi-camera person tracking and dementia signal detection for senior care. Processes RTSP camera feeds through a 15-stage ML pipeline: YOLO detection, spatial floor projection, SOLIDER-REID and RTMPose inference, cross-camera pre-association dedup, floor-plane Kalman world tracker (Hungarian association over PersonHypothesis), Bayesian identity resolution, posture classification, trajectory writing, and event publishing.

Detects clinically relevant behavioral patterns from trajectory and dwell data: pacing, sundowning, bathroom dwell anomalies, prolonged stillness, nighttime movement, and unexplained absence.

Results are streamed via Redis Streams (protobuf) to [Cognitive Companion](../cognitive-companion), the BFF gateway that serves the Vue admin UI and MCP tools.

**Documentation:** [silvermind-project.github.io](https://silvermind-project.github.io)

**Architecture reference:** [docs/systems-architecture.md](docs/systems-architecture.md) covers the PersonHypothesis world-tracker model, the identity resolver, home-camera nuances, and how CTS feeds Cognitive Companion.

## Services

| Service | Port | Description |
| --- | --- | --- |
| `go2rtc` | 1984 | RTSP proxy sidecar |
| `rtsp-ingress` | 8090 | Go RTSP ingest: camera registration, motion gating, MinIO upload |
| `tracking-orchestrator` | 8000 | Python ML pipeline: detection, tracking, identity, signals |
| `triton` | 8701 | ONNX model serving (YOLO, SOLIDER-REID, RTMPose) |

## Key design points

- **PersonHypothesis (PH)** is the world-level tracked-person entity. `ph_id` (UUID) is the single cross-camera identifier on the wire.
- **Cross-camera dedup**: before the Hungarian assignment runs, a pre-association floor-point pass collapses same-person observations from overlapping cameras into one representative, preventing duplicate PHs for a person seen at a hallway/bathroom boundary.
- **Quality capture**: each PH carries a `mean_quality` field (EMA of observation quality scores) that travels to Cognitive Companion for display in location envelopes.
- **No silent fallbacks**: stream consumers dead-letter unprocessable messages with a metric and warning log; they never silently skip.

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
make check          # ruff + mypy + import-linter + pytest (fast gate)
make all-check      # Python + Go + buf lint
make ci             # authoritative gate: all-check + integration proofs (requires Docker)
```

## License

AGPL-3.0-or-later
