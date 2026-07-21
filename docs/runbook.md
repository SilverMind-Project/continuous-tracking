# Continuous Tracking System: operational runbook

This runbook covers the day-2 ops surface for the production CTS
deployment. It assumes the manifests in `k8s/` are applied and the
services are healthy at baseline.

## Service inventory

| Service | Port | Owner | What it does |
|---------|------|-------|--------------|
| `rtsp-ingress` | 8090 | CTS | Pulls RTSP, gates on motion, uploads JPEG, publishes `frames.ready`. |
| `tracking-orchestrator` | 8000 | CTS | Consumes `frames.ready`, runs Triton inference, emits `tracking.events`, `tracking.revisions`, `tracking.signals`, `scene.samples`. |
| `triton` | 8001 (gRPC) | CTS | YOLO26L + SOLIDER-REID + RTMPose. |
| `redis` | 6379 | CTS | Streams transport. AOF on. |
| `postgres` | 5432 | CTS | TimescaleDB + pgvector. Holds tracklets, gallery, signals, trajectories. |

## Top-line dashboards

- `cts_frame_end_to_end_latency_ms` p99 by camera — budget 450 ms.
- `cts_tracking_events_published_total` rate by camera — should track
  motion-gate output of `cts_frames_consumed_total`.
- `cts_identity_revisions_total` rate by reason — sustained > 2/min is
  a smell (see runbook entry below).
- `cts_posterior_entropy_bits` p95 — > 2.5 bits sustained means the
  resolver is undecided; investigate face-ID outages or gallery drift.
- Redis Streams `XLEN frames.ready` minus consumer last-delivered ID —
  primary backpressure signal.

## Common incidents

### 1. A camera stopped producing frames

Symptom: `cts_frames_consumed_total{camera_id="X"}` flatlined.

```bash
# Confirm worker is registered.
kubectl -n cts exec deploy/rtsp-ingress -- \
  wget -qO- http://localhost:8090/metrics | grep 'rtsp_active_workers\|camera_id="X"'

# Distinguish RTSP flap from decode failure.
# - rtsp_reconnects_total climbing  -> camera offline / wrong VLAN
# - rtsp_decode_errors_total climbing while reconnects flat  -> bitstream issue
```

Triage: ssh to the ingress node, run
`ffprobe rtsp://camera-host:554/stream1`. If ffprobe also fails, the
camera is the issue, not CTS. If only rtsp-ingress fails, force software
decode for that camera by patching its row in
`continuous_tracking.cameras` (`decode_preferred='software'`).

### 2. Identity revisions storming

Symptom: `rate(cts_identity_revisions_total[5m]) > 2`.

Diagnosis is almost always one of:

- A face in the gallery has aged (drift).
- Two residents in similar clothing trip the resolver back-and-forth.
- A new guest was incorrectly enrolled with a household member's id.

Triage:

```bash
# Pull the recent revision audit log via the CC BFF.
curl -sH "X-API-Key: $CC_API_KEY" \
  http://cc/api/v1/cts/revisions?since_minutes=15 | jq
```

If revisions cluster on one global_track_id, the gallery is at fault.
Promote a fresh face anchor via the CC corrections UI, then prune older
gallery rows for that identity.

### 3. End-to-end latency p99 > 450 ms

Symptom: `cts_frame_end_to_end_latency_ms{quantile="0.99"} > 450`.

Pull the per-stage latency histogram:

```bash
kubectl -n cts port-forward svc/tracking-orchestrator 8000 &
curl -s http://localhost:8000/metrics | grep cts_triton_inference_latency_ms
```

If Triton is the dominant stage, scale Triton's batch_size down (improves
tail latency at the cost of throughput) or bring up a second Triton
replica behind a model-instance group. If the dominant stage is the
identity resolver, investigate gallery size — beyond ~5k rows the HNSW
search tail starts climbing; tune `ef_search` or prune older rows.

### 4. Redis Streams consumer lag

Symptom: `cts_frames_ready_consumer_lag > 1000` for > 1 minute.

```bash
# Inspect XINFO.
kubectl -n cts exec sts/redis -- redis-cli XINFO STREAM frames.ready
kubectl -n cts exec sts/redis -- redis-cli XINFO GROUPS frames.ready
```

If `consumers` is fewer than the orchestrator replica count, the HPA
hasn't scaled up. Bump replicas manually:

```bash
kubectl -n cts scale deploy/tracking-orchestrator --replicas=4
```

If consumers are present but `pending` is high, a worker is stuck.
Kill the affected pod; XAUTOCLAIM will reclaim its pending entries
within `reclaim_idle_ms` (60 s default).

### 5. Postgres backup / disaster recovery

PITR is enabled via the Timescale Operator. Recovery target:

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
| Triton GPU memory | 200 MB warm | -- |
| `tracking.events` rate | up to 6 events/sec | scales with detections |
| `person_trajectories` rows/day | -- | ~150,000 |
| `reid_gallery` rows | up to 50 per identity | up to 50 per identity |
| `dementia_signals` rows/day | -- | ~30 |

For a 12-camera, 3-resident home:

- One Triton replica with one A2000-class GPU is sufficient.
- Two orchestrator replicas at 1 vCPU / 2 GiB is the steady-state target.
- Postgres at 4 vCPU / 8 GiB sustains the trajectory write rate
  comfortably; tune `chunk_time_interval` per phase-1 §1.4 if the
  resident count scales beyond 5.
- Redis at 1 GiB peak; AOF rewrite cadence default is fine.

## Retiring the tracking.responses stream key (post-2026-06 upgrades)

The `tracking.responses` Redis stream has no producer since the 2026-06
release. It is safe to delete the key on existing deployments after upgrading:

```bash
kubectl -n cts exec sts/redis -- redis-cli DEL tracking.responses
```

The cognitive-companion drain subscriber will then idle on an empty stream
without error. The `FrameResponse` proto message is retained in the schema
for wire-format compatibility but is never populated.

## Replay testing

The deterministic replay suite lives in
`tracking-orchestrator/tests/replay/`. Nightly CI runs the canonical
10-minute scenario and asserts the latency + identity budgets in
phase-0 §0.18.6. To replay locally:

```bash
make -C tracking-orchestrator validate
```

The CI run also asserts that every Redis Streams message round-trips
through proto encode/decode unchanged.  See `docs/wire-format.md` for
the per-stream contract.
