# Kubernetes manifests

Production deployment for the Continuous Tracking System (M10).

**These manifests have been migrated** to the unified `kubernetes/` directory at
the repo root alongside all other subproject manifests.  The unified structure
uses the `nanai` namespace (replacing the old `cts` namespace).

See `../kubernetes/continuous-tracking/` for the current manifests and
`../kubernetes/` for the full monorepo deployment layout including the shared
PostgreSQL StatefulSet, Redis, and application deployments (cognitive-companion,
continuous-tracking, person-identification, tts-service).

## Layout

| File | What it deploys |
|------|-----------------|
| `configmap.yaml` | Orchestrator + ingress runtime configuration (namespace: nanai). |
| `secrets.example.yaml` | Templates for credentials. Replace with real values via sealed-secrets / external-secrets. |
| `redis.yaml` | Redis StatefulSet with AOF persistence (consumer-group offsets must survive restart). |
| `triton.yaml` | Triton Inference Server Deployment (GPU node). |
| `rtsp-ingress.yaml` | Go ingress Deployment + camera-only NetworkPolicy. |
| `tracking-orchestrator.yaml` | Python orchestrator Deployment, HPA, PDB. |

The CC backend lives in the same `nanai` namespace; the `deny-browser-ingress`
NetworkPolicy restricts the CTS services to traffic from `nanai` namespace only
(phase-0 gateway rule).  The shared PostgreSQL and Redis stateful sets are in
`../kubernetes/infrastructure/`.

## Apply order

```bash
kubectl apply -k ../kubernetes/   # Kustomize deploys all resources in order
```

## Operational notes

- **Wire format.** All Redis Streams use a proto-only envelope.
- **HPA.** The custom metric `cts_frames_ready_consumer_lag` must be exposed via prometheus-adapter.
- **GPU.** The Triton Deployment uses `Recreate` strategy.
- **Redis persistence.** AOF is mandatory for consumer-group offset survival.
- **Backups.** Postgres backups are out of scope here; use the Timescale Operator or pgBackRest in production for PITR.
