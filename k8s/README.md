# Kubernetes manifests

Production deployment for the Continuous Tracking System (M10).

## Layout

| File | What it deploys |
|------|-----------------|
| `namespace.yaml` | The `cts` namespace with restricted PodSecurity. |
| `configmap.yaml` | Orchestrator + ingress runtime configuration. |
| `secrets.example.yaml` | Templates for credentials. Replace with real values via sealed-secrets / external-secrets. |
| `postgres.yaml` | TimescaleDB StatefulSet (pgvector + TimescaleDB pre-installed). |
| `redis.yaml` | Redis StatefulSet with AOF persistence (consumer-group offsets must survive restart). |
| `triton.yaml` | Triton Inference Server Deployment (GPU node). |
| `rtsp-ingress.yaml` | Go ingress Deployment + camera-only NetworkPolicy. |
| `tracking-orchestrator.yaml` | Python orchestrator Deployment, HPA, PDB. |

The CC backend lives in a separate `cc` namespace; the
`deny-browser-ingress` NetworkPolicy in this directory restricts the CTS
services to traffic from `cc` and `monitoring` namespaces only (phase-0
gateway rule).

## Apply order

```bash
kubectl apply -f namespace.yaml
kubectl apply -f secrets.example.yaml   # after replacing values
kubectl apply -f configmap.yaml
kubectl apply -f postgres.yaml
kubectl apply -f redis.yaml
kubectl apply -f triton.yaml
kubectl apply -f rtsp-ingress.yaml
kubectl apply -f tracking-orchestrator.yaml
```

## Operational notes

- **Wire format.** All Redis Streams use a proto-only envelope: one
  named field per message carrying the raw `Message.SerializeToString()`
  body. See `docs/wire-format.md` for the per-stream contract.
- **HPA.** The custom metric `cts_frames_ready_consumer_lag` must be
  exposed via prometheus-adapter. The default scaling target is 200
  pending messages per pod; tune for your camera count.
- **GPU.** The Triton Deployment uses `Recreate` strategy because
  TensorRT model load is slow. On rolling deploys the orchestrator's
  `frames.ready` consumers buffer in Redis until Triton is ready again.
- **Redis persistence.** AOF is mandatory: consumer-group offsets and
  pending entry lists must survive restarts so we don't lose in-flight
  TrackingEvent or IdentityRevision messages.
- **Backups.** Postgres backups are out of scope here; use the
  Timescale Operator or pgBackRest in production for PITR.
