# Continuous Tracking System: operational runbook

This runbook covers the day-2 operational surface for the Continuous Tracking
System (CTS). Commands show both the local Docker Compose deployment
(`docker-compose.yml`) and the Kubernetes deployment
(`../kubernetes/continuous-tracking/`).

## Service inventory

| Service | Container port | Compose host port | Owner | What it does |
|---------|----------------|-------------------|-------|--------------|
| `go2rtc` | 1984 (HTTP), 8554 (RTSP) | 1984, 8554 | CTS | Maintains steady RTSP pulls from cameras and serves latest JPEG keyframes over localhost (`network_mode: host`). |
| `rtsp-ingress` | 8090 | 8090 | CTS | Polls `go2rtc` for keyframes, gates on pixel-diff motion, uploads JPEGs to MinIO, and publishes `frames.ready`. |
| `tracking-orchestrator` | 8000 | 8500 | CTS | Consumes `frames.ready` and `cc.identity_assertions`, runs Triton + `person-identification-service` inference, advances the world tracker, and emits `tracking.events`, `tracking.revisions`, `tracking.signals`, `tracking.continuations`, `tracking.room_transitions`, and `scene.samples`. |
| `triton` | 8000 (HTTP), 8001 (gRPC), 8002 (metrics) | 8700, 8701, 8702 | CTS / shared | Serves YOLO26L (`person-detector`), SOLIDER-REID (`reid-solider`), RTMPose (`pose-rtmpose`), Buffalo_L (`face-*`), and Depth Anything V2 (DGX only). |
| `person-identification-service` | 8000 | 8200 | Shared | HTTP face detection, ArcFace embedding matching, and face attribute/quality service called by `FaceIdentityStage` (`POST /api/v1/identify`) and `/internal/identity/readiness`. |
| `redis` | 6379 | 6379 | Shared | Redis Streams transport with AOF persistence enabled. |
| `postgres` | 5432 | 5435 | Shared | TimescaleDB + pgvectorscale (`continuous_tracking` schema: `person_hypotheses`, `person_trajectories`, `reid_gallery`, `identity_decisions`, `ph_revisions`, `dementia_signals`, `homography_calibrations`, `face_observation_stats`). |

## Top-line dashboards

- `cts_frame_end_to_end_latency_ms` p99 by camera: budget 450 ms.
- `cts_tracking_events_published_total` rate by camera: tracks live event emission (throttled per camera by `pipeline.live_publish_max_hz`, default 3.0 Hz) against `cts_frames_consumed_total`.
- `cts_identity_revisions_total` rate by reason: sustained > 2/min is
  a smell (see Incident 2 below).
- `cts_posterior_entropy_bits` p95: > 2.5 bits sustained means the
  resolver is undecided; investigate face-ID availability, calibration readiness (`GET /internal/identity/readiness`), or gallery staleness.
- `cts_frames_dropped_stale_total` and `cts_frames_failed_total`: frame-age shedding (`max_frame_age_ms`) and stage execution errors.
- Redis Streams `XLEN frames.ready` minus consumer last-delivered ID:
  primary backpressure signal.

## Common incidents

### 1. A camera stopped producing frames

Symptom: `cts_frames_consumed_total{camera_id="X"}` flatlined.

```bash
# Confirm worker is registered and inspect per-camera rtsp-ingress counters.
# Docker Compose:
curl -fsS http://localhost:8090/metrics | grep -E 'rtsp_active_workers|camera_id="X"'

# Kubernetes:
kubectl -n cts exec deploy/rtsp-ingress -- \
  wget -qO- http://localhost:8090/metrics | grep -E 'rtsp_active_workers|camera_id="X"'
```

Distinguish camera/stream failure from motion-gate suppression using the actual `rtsp-ingress` Prometheus metrics:

- `rtsp_active_workers` dropped or `camera_id="X"` absent: verify that Cognitive Companion returns the camera with `cts_enabled: true` on `GET /api/v1/cts/cameras`. `rtsp-ingress` reconciles its worker list from CC every `CC_POLL_INTERVAL` (default 30 s).
- `rtsp_fetch_errors_total{camera_id="X"}` climbing and `rtsp_camera_backoff_seconds{camera_id="X"}` elevated: `go2rtc` cannot pull or transcode the camera's RTSP stream (`http://localhost:1984/api/frame.jpeg?src=X` is failing).
- `rtsp_frames_filtered_total{camera_id="X"}` climbing while `rtsp_frames_published_total{camera_id="X"}` is flat and `rtsp_last_fetch_success_timestamp_seconds{camera_id="X"}` is current: `go2rtc` is healthy, and the pixel-diff motion gate is filtering static scenes as expected.

Triage when `rtsp_fetch_errors_total` is climbing:

1. Check `go2rtc` stream health directly: `curl -fsS "http://localhost:1984/api/streams?src=X" | jq`.
2. Probe the camera RTSP endpoint from the ingress host: `ffprobe -rtsp_transport tcp rtsp://camera-host:554/stream1`.
3. If `ffprobe` fails, the camera or network VLAN is offline. If `ffprobe` succeeds, inspect `docker logs cts-go2rtc` and verify the camera's `rtsp_url` in Cognitive Companion (`/admin/cameras`).

### 2. Identity revisions storming

Symptom: `rate(cts_identity_revisions_total[5m]) > 2`.

Diagnosis is almost always one of:

- Two residents in similar clothing or with uncalibrated face scores are oscillating near the commit margin (`commit_probability = 0.75`, `commit_margin = 0.25`).
- Polluted or stale `reid_gallery` entries in `auto_verified` or `operator_verified` (`VOTING_STATES`) are pulling the body-appearance likelihood toward the wrong identity.
- A visitor or household member has an incomplete or mis-assigned face prototype set.

Triage:

```bash
# 1. Inspect the recent revision audit log via the CC BFF.
curl -sH "X-API-Key: $CC_API_KEY" \
  "http://localhost:8000/api/v1/cts/revisions?since_minutes=15" | jq

# 2. Check household identity readiness and gallery health directly on CTS.
curl -fsS http://localhost:8500/internal/identity/readiness | jq
```

If revisions cluster on one `ph_id` (or a recurring pair of `committed_identity_id` values):

1. Check `/admin/identity-setup` (`GET /internal/identity/readiness`) to verify enrolled prototype diversity and calibration coverage across cameras.
2. Inspect `/admin/cts/reid-review` (`GET /internal/gallery/candidates`) for mis-verified body appearance crops and reject any bad gallery entries (`POST /internal/gallery/candidates/{id}/reject`). Only rows in `VOTING_STATES` (`auto_verified`, `operator_verified`) younger than `gallery_vote_max_age_s` (12 hours) vote in `IdentityResolver`.
3. If fresh face crops need to be enrolled as face prototypes, use the governed face calibration promotion endpoint (`POST /api/v1/face-calibration/candidates/{candidate_id}/promote`) or member prototype upload (`POST /api/v1/persons/{person_id}/prototypes`).
4. Use PH identity correction (`/admin/cts/corrections`, `POST /internal/corrections/ph/{ph_id}`) only to relabel a specific PersonHypothesis (`full_ph` or `time_range`). Operator PH corrections are scoped to the PH and never modify face enrollment or the ReID gallery.

### 3. End-to-end latency p99 > 450 ms

Symptom: `cts_frame_end_to_end_latency_ms{quantile="0.99"} > 450`.

Pull the per-stage and Triton latency metrics:

```bash
# Docker Compose (host port 8500 -> container port 8000):
curl -fsS http://localhost:8500/metrics | grep -E 'cts_triton_inference_latency_ms|cts_frame_end_to_end_latency_ms'

# Kubernetes:
kubectl -n cts port-forward svc/tracking-orchestrator 8000:8000 &
curl -fsS http://localhost:8000/metrics | grep -E 'cts_triton_inference_latency_ms|cts_frame_end_to_end_latency_ms'
```

If Triton is the dominant stage, check Triton queue and compute metrics on port `8702` (`8002` in container) and verify whether `cts_frames_dropped_stale_total` is incrementing (`pipeline.max_frame_age_ms`, default 2000 ms, drops frames that wait too long in `frames.ready`). On an 8 GB Jetson (`docs/jetson-int8-deployment.md`), confirm `triton.detector_static_batch_size: 1` and `triton.depth_enabled: false`. If the dominant stage is the identity resolver or database I/O, inspect `reid_gallery` row counts and TimescaleDB query latency.

### 4. Redis Streams consumer lag

Symptom: `XLEN frames.ready` or consumer group lag growing continuously for > 1 minute.

```bash
# Inspect stream and consumer group state.
redis-cli XINFO STREAM frames.ready
redis-cli XINFO GROUPS frames.ready
redis-cli XPENDING frames.ready cts-orchestrator
```

`RedisStreamsTransport` uses at-most-once delivery after a processing attempt: `_finish_frame` always `XACK`s the entry whether the frame succeeded, was dropped as stale (`cts_frames_dropped_stale_total`), or failed (`cts_frames_failed_total`) so a poison frame cannot wedge the consumer group.

If `pending` entries accumulate because an orchestrator replica crashed mid-frame, `RedisStreamsTransport.consume_frames()` automatically runs `XAUTOCLAIM` once at startup for entries idle longer than `redis.pending_reclaim_min_idle_s` (60 s default). Restarting or scaling `tracking-orchestrator` reclaims orphaned PEL entries automatically.

### 5. Postgres backup / disaster recovery

PITR is enabled via the Timescale Operator in Kubernetes deployments (or standard `pg_basebackup` / WAL archiving on Docker hosts). Recovery target in Kubernetes:

```bash
# 1. Identify the most recent base backup and WAL chunk.
kubectl -n cts exec sts/postgres -- ls /backup
# 2. Restore via the operator's restore CR.
cat <<EOF | kubectl apply -f -
apiVersion: timescale.com/v1alpha1
kind: TimescaleDBRestore
metadata: { name: cts-restore, namespace: cts }
spec:
  targetTime: 2026-04-21T12:00:00Z
  backupRef: latest
EOF
```

Recovery point objective: 5 minutes (matches WAL archive cadence).
Recovery time objective: 30 minutes for the database; the orchestrator
recovers automatically once Postgres is back since all of its state is
either in the DB or replayable from Redis (which has its own AOF).

## Capacity planning

| Resource | Per-camera budget | Per-resident budget |
|----------|-------------------|---------------------|
| Frames/sec at ingress | 2 (motion-gated) | -- |
| Triton GPU memory | 200 MB warm (DGX); see `docs/jetson-int8-deployment.md` for Jetson 8 GB | -- |
| `tracking.events` rate | up to `live_publish_max_hz` (default 3 Hz/camera) | scales with active PHs |
| `person_trajectories` rows/day | -- | ~150,000 |
| `reid_gallery` rows | -- | up to 50 active per identity (`pending_review` + `auto_verified` + `operator_verified`) |
| `dementia_signals` rows/day | -- | ~30 |

For a 12-camera, 3-resident home:

- One Triton replica with one A2000-class GPU (or an 8 GB Jetson Orin Nano Super running the INT8 model set for 6 to 8 cameras) is sufficient.
- Two orchestrator replicas at 1 vCPU / 2 GiB is the steady-state target.
- Postgres at 4 vCPU / 8 GiB sustains the trajectory write rate
  comfortably; tune `chunk_time_interval` per phase-1 section 1.4 if the
  resident count scales beyond 5.
- Redis at 1 GiB peak; AOF rewrite cadence default is fine.

## Retiring deprecated Redis stream keys

The `tracking.responses` stream (retired 2026-06, TD-006) and the
`tracking.presence` and `tracking.dwell` streams (retired in M37; marked
`deprecated = true` in `proto/continuoustracking/v1/tracking.proto`) have no
active producer in `tracking-orchestrator`. It is safe to delete any leftover
keys on existing deployments after upgrading:

```bash
redis-cli DEL tracking.responses tracking.presence tracking.dwell
```

The corresponding proto messages (`FrameResponse`, `PresenceEvent`,
`DwellEvent`) remain in `tracking.proto` solely for wire-format backwards
compatibility.

## Verification and characterization testing

Deterministic end-to-end world tracker, spatial fusion, cross-camera continuity,
fall-sequence, and wire-contract suites live under
`tracking-orchestrator/tests/`:

- `tracking-orchestrator/tests/integration/` (`test_world_tracker_e2e.py`, `test_fusion_acceptance.py`, `test_m2_ph_continuity.py`, `test_m5_cross_camera.py`, `test_fall_sequences.py`, `test_ph_repos.py`)
- `tracking-orchestrator/tests/characterization/` (`test_world_tracker_decisions.py`, `test_signal_detector_fires.py`, `test_reid_candidate_stage.py`)
- `tracking-orchestrator/tests/test_proto_contracts.py` and `tracking-orchestrator/tests/contracts/`

To run the fast local quality gate or the full CI gate (including testcontainer Postgres integration tests):

```bash
make check        # fast local gate (ruff, mypy, import-linter, unit + characterization tests, check-md, docs-check)
make ci           # full CI gate (check-all + pytest -m integration)
```

See `docs/wire-format.md` for the per-stream wire contract.
