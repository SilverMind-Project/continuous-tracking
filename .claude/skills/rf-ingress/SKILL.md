---
name: rf-ingress
description: Use when building or modifying continuous-tracking/rf-ingress/, adding an RF hardware adapter or payload profile, changing the multilateration solver, epoch builder, clock offset tracker, replay tooling, or adapter conformance suite.
paths: ["**/rf-ingress/**"]
---

# rf-ingress

Hardware-neutral Python ingestion service (`continuous-tracking/rf-ingress/`, port `8095`) that turns RF hardware or simulator inputs into validated `TagFix` protobuf messages on `wearable.fixes` and `AdapterCapability` heartbeats on `wearable.capabilities`.

## Non-negotiable rules

1. **Hardware stops at one adapter module.** Every radio or vendor format sits behind one adapter module under `rf_ingress/adapters/` implementing `RangeAdapter`, `PositionAdapter`, and/or `RoomAdapter` (`rf_ingress/adapters/base.py`). Vendor payload Pydantic models live inside that module and are selected by a versioned `payload_profile`. No code outside an adapter names a vendor or firmware; downstream code branches only on declared capability and covariance.
2. **Conformance kit is the definition of "adapter done".** Every adapter (present and future) adds a factory case in `tests/conformance/conftest.py` and must pass `tests/conformance/test_adapter_conformance.py` unchanged.
3. **Per-epoch fixes only; never smooth across epochs.** `EpochBuilder` (`rf_ingress/solver/epoch.py`) groups ranges per `(adapter_id, tag_id)` by explicit `epoch_seq` or `epoch_window_s` (0.25 s). `tracking-orchestrator` is the only temporal filter in the chain.
4. **Wearables are identity evidence only.** `rf-ingress` never drops `OFF_BODY` fixes (it passes `wear_state` and `motion_state` through truthfully so CTS decides). A missing fix is never evidence of absence.
5. **Accuracy is covariance.** Every `RangeMeasurement` (`rf_ingress/domain.py`) carries a required positive `variance_m2` and a closed `RangingTechnology` (`ble_rssi`, `ble_channel_sounding`, `uwb` from `cts_contracts.wearables`). `MultilaterationSolver` (`rf_ingress/solver/multilateration.py`) weights strictly by `variance_m2`, never branches on technology, and never drops wide-covariance fixes (`max_sigma_m: null` by default). Reject only on non-convergence, collinear/degenerate geometry (`cond > max_condition`), or non-positive-definite covariance.
6. **Wrist-local azimuth is telemetry only.** `MultilaterationSolver` ignores `azimuth_deg` whenever `azimuth_reference_frame != "floor_plan"` (including `"device_local"` from wrist-worn watches) so 3+ anchor distance solves are invariant to wrist rotation. Optional bearing residuals apply only when 2 fixed anchors report calibrated `"floor_plan"` bearings.
7. **Core room fallback.** When an epoch has fewer than 3 known RF nodes, `room_fallback_from_epoch` (`rf_ingress/solver/epoch.py`) emits `FIX_KIND_ROOM` for the nearest node's `room_id` with margin-based `room_probability`. Adapters do not reimplement this fallback.
8. **Site-config revision atomicity.** `HttpSiteConfigClient` (`rf_ingress/site_config.py`) polls CC `GET /api/v1/wearables/site-config` with `X-API-Key` every 60 s and keeps the last valid snapshot on errors. Whenever `version` changes, `EpochBuilder.on_site_config_version` drops all open epochs so no solve mixes nodes from two floor-plan revisions.
9. **Fail-closed registration and binding gate.** `RegistrationGate` (`rf_ingress/registration.py`) publishes a fix only when a site-config snapshot is loaded, the `tag_id` is registered and `is_active`, and a binding covers `measured_at` (`[valid_from, valid_to)`).
10. **Privacy and bounded labels.** Raw fixes on `wearable.fixes` are trimmed with `XADD ... MINID ~ <now - 72h>` (`rf_ingress/publish.py`). Logs use `structlog` and never tie positions to member IDs. Enum Prometheus labels (`ReportKind`, `DropReason`, `FixKind`) are closed `StrEnum`s in `rf_ingress/observability.py`; member IDs never appear in any metric label.

## Architecture and modules

| Module | Responsibility |
| --- | --- |
| `rf_ingress/domain.py` | Frozen dataclasses (`RfNodeGeometry`, `RangeMeasurement`, `PositionReport`, `RoomReport`, `RangeEpoch`, `SolveResult`, `EmittedFix`, `SiteConfigSnapshot`); zero internal imports |
| `rf_ingress/config.py` | Pydantic v2 settings models; YAML + `${ENV}` interpolation (defaulted `${VAR:-x}` disallowed) |
| `rf_ingress/adapters/base.py` | `RangeAdapter`, `PositionAdapter`, `RoomAdapter` protocols, `AdapterDeps`, `AdapterHealth` |
| `rf_ingress/adapters/registry.py` | `AdapterRegistry` mapping `kind` to factory; unknown `kind` or `payload_profile` fails startup |
| `rf_ingress/adapters/simulator.py` | Seeded deterministic simulator (all 3 protocols, UWB/CS/BLE-RSSI technology classes, burst batching, NLOS/body shadow) and JSONL replay |
| `rf_ingress/solver/epoch.py` | `EpochBuilder` and `room_fallback_from_epoch` |
| `rf_ingress/solver/multilateration.py` | Pure numpy/scipy `MultilaterationSolver` (3D when >= 4 nodes with vertical spread >= 0.5 m, else 2D at `assumed_tag_height_m=1.0` with `tag_height_sigma_m=0.3` vertical-Jacobian variance inflation; leave-one-out chi-square 99% outlier drop; `cov = inv(J.T W J) * max(1, chi2_red)`) |
| `rf_ingress/solver/frame_transform.py` | 2D similarity transform (`fit_similarity_transform`) for vendor-frame `PositionReport`s |
| `rf_ingress/clock.py` | `BurstClockOffsetTracker` (minimum-delay `arrival - sent_at` over last 20 batches per registered `tag_id`, exporting `rf_ingress_device_clock_offset_ms`) and `AdapterClockSkewEstimator` (60 s median per adapter) |
| `rf_ingress/site_config.py` | `SiteConfigProvider` protocol, `HttpSiteConfigClient`, `InMemorySiteConfigProvider` |
| `rf_ingress/registration.py` | `RegistrationGate` |
| `rf_ingress/publish.py` | `TagFixPublisher` protocol, `RedisTagFixPublisher` (raw protobuf, `decode_responses=False`, 72 h `MINID`), `InMemoryTagFixPublisher` |
| `rf_ingress/pipeline.py` | `RfIngressRuntime` composition and background loops |
| `rf_ingress/health.py` | FastAPI `/health` and `/metrics` |
| `rf_ingress/replay.py` | CLI `python -m rf_ingress.replay record|play` (raw adapter reports only; 60-minute recording cap) |

## How to add a hardware adapter

1. Create one module `rf_ingress/adapters/<name>.py` implementing `RangeAdapter`, `PositionAdapter`, or `RoomAdapter` with strict Pydantic v2 config and versioned `payload_profile` models inside that module.
2. Drop and count unregistered hardware keys/tags (`UNREGISTERED_TAG_TOTAL`, `DropReason.UNREGISTERED_TAG`) and malformed payloads (`DropReason.MALFORMED_INPUT`) without ever raising out of the async iterator.
3. For burst-batched senders (such as `device_ranging.v1`), populate `RangeMeasurement.sent_at` from the batch flush timestamp and `measured_at` from the per-epoch capture timestamp so `BurstClockOffsetTracker` decouples queueing jitter from clock skew.
4. Register the factory in `rf_ingress/adapters/registry.py`.
5. Add a conformance case to `tests/conformance/conftest.py` and adapter unit tests with pinned sample payloads.
6. Verify `make rf-check` and `make ci` pass without changing any file under `tracking-orchestrator/app/`.

## Streams and gates

- `wearable.fixes`: field `"fix"`, raw `TagFix` protobuf, `MINID` trimmed to 72 hours.
- `wearable.capabilities`: field `"capability"`, raw `AdapterCapability` protobuf, 60 s heartbeat.
- Fast gate: `make rf-check` (ruff check, ruff format --check, mypy strict, import-linter, unit + conformance pytest).
- Full gate: `make ci` (includes `make rf-test-integration` against a Redis testcontainer).
- Use `rf-ingress/.venv/` (Python 3.12), never system Python. Sync with `cd rf-ingress && uv sync --frozen --extra dev`.
