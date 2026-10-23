# Domain design principles and the wire format

Read before touching units, hardware adapters, caches, stages, calibration, identity, signals, cross-service contracts, dedup, quality, or stream consumers. Part of the `engineering-standards` skill; the rules summary is in `../SKILL.md`.

## Redis Stream Wire Format

**Redis Streams carry raw protobuf bytes -- no JSON, no base64.** Use `app/transport/codec.py`:

```python
from app.transport.codec import encode, decode

# publish
await redis.xadd("tracking.events", encode(event_msg, field="event"))

# consume
event = decode(fields, TrackingEvent, field="event")
```

| Stream | Field | Message type |
| --- | --- | --- |
| `frames.ready` | `"frame"` | `FrameReady` |
| `tracking.events` | `"event"` | `TrackingEvent` |
| `tracking.revisions` | `"revision"` | `IdentityRevision` |
| `tracking.signals` | `"signal"` | `DementiaSignal` |
| `scene.samples` | `"sample"` | `SceneSample` |

1. Redis clients use `decode_responses=False` so binary payloads round-trip unchanged.
2. No codec discriminator field -- consumers know the message type from the stream.
3. No dual-codec shim; protobuf-only is the sole wire format for new streams.

Known exceptions in the code today: `tracking.continuations`
(`app/transport/ph_continuation_publisher.py`) and `tracking.room_transitions`
(`app/transport/room_transition_publisher.py`) publish one Redis field per value, not a protobuf
message. `tracking.presence` and `tracking.dwell` are declared in `proto/` and have metrics but no
publisher. Do not copy the field-map shape; converting those two streams is a two-repo change.

## Domain-Specific Design Principles

### Units are part of the interface

Every dimensional numeric field names its unit: `_m`, `_m_s`, `_nu_s`, `_deg`, or `_px`. Its
docstring also names the reference frame, such as image-normalized, crop-normalized, or
floor-plane meters.

Threshold constants sit beside a comment stating the unit and calibration source. Link the
calibration script or fixture. Multiplying or dividing quantities from incompatible frames is a
review-blocking defect even when typical-scale output looks plausible.

```python
# WRONG: crop-normalized displacement / bbox_diagonal_px / dt_s
# RIGHT: app/trajectory/motion_energy.py converts to image px, then divides by bbox px and dt_s.
frame_vel = float(np.mean(disp / scale) / dt)
```

The RIGHT implementation names the result `mean_keypoint_velocity_nu_s` and links
`scripts/calibrate_motion_energy.py` beside `_STILL_VELOCITY_FLOOR_NU_S`.

### Hardware and vendor abstraction

Sensors and devices (cameras, lenses, UWB anchors and tags, BLE receivers, wearables) are chosen
by parts availability, so vendor specifics must never leak past one adapter boundary.

- Every hardware integration is an adapter that implements a consumer-owned `Protocol` (Python)
  or interface (Go) and emits one hardware-neutral, versioned contract (a protobuf message for
  anything crossing a process boundary). Downstream code depends on the contract only.
- Vendor SDKs, firmware message formats, and device-specific units are confined to the adapter
  module. No vendor name appears in a domain type, a proto field name, a database column, or a
  settings key outside the adapter's own config block.
- What a device can do is declared, not inferred: adapters publish a capability descriptor
  (for example: reports position with covariance, room only, or proximity only; update rate;
  whether it reports wear state). Consumers branch on declared capability, never on vendor or
  model identity.
- Adapter selection is configuration, resolved through a registry mapping a config
  discriminator to a builder (the CC presence-provider factory pattern). No `if vendor == ...`
  chains in business logic.
- Every adapter family has a simulator or replay adapter that produces the same contract. CI and
  integration tests run against it; no test requires physical hardware.
- Physical placement (camera pose, anchor position, mount height) is site configuration in the
  shared floor-plan frame, owned by CC and synced through `cc_config_sync`, never hard-coded.

### Caches with temporal contracts

A cache valid "per frame", "per round", or "per run" must name that epoch in its class docstring.
Invalidate it at one structural chokepoint traversed by every execution path, not separately at
each path entry. Add a staleness backstop that logs and degrades safely after missed invalidation.

**RIGHT:** `app/pipeline/gallery_cache.py` documents a per-frame cache, self-invalidates after
`max_age_s`, and logs `gallery_cache_stale_self_invalidated`. Both single-frame and batched paths
call `FrameProcessingPipeline._begin_tracker_round()` in `app/pipeline/frame_pipeline.py`.

**WRONG:** clear the gallery cache only in `_process_frame`; the cross-camera batch path then
serves entries from the previous tracker round.

Review checklist when adding an execution path: which epoch-scoped caches does this path reset?

### Frame pipeline stages

- **One stage, one responsibility.**
- **Typed stage contracts.** `FrameStage.run(ctx)`. Constructor dependencies are explicit protocols, never a whole pipeline instance.
- **Context ownership is documented.** Every `FrameContext` field has one producing stage and documented consuming stages. No untyped scratch fields.
- **Use frame time deliberately.** `capture_time` = physical observation. `event_time` = processing time. No `datetime.now(UTC)` when one of these is correct.
- **Model failures degrade by model per frame.** Detector failure = frame fails. ReID/pose/face-ID failure = missing evidence.
- **No private cross-stage reachback.** Shared logic becomes a small service injected into both stages.

### Spatial calibration and homography

- **Coordinate system is explicit.** Calibrated `FloorPoint` = shared floor-plan coordinates in mm.
- **Homographies map raw pixels to floor-plan metres.** Store floor_plan_id, image dimensions, residuals, and quality with every matrix.
- **Never compare points from different floor plans.** Hard reject with metrics.
- **Unknown geometry is not perfect geometry.** No geometry score of 1.0 for uncalibrated cameras. Use neutral scoring or reject.
- **Attach projection once.** Project before tracking. Carry the same point into tracklets, trajectories, and proto events.
- **Ground-plane homography is not person height.** Do not claim physical height from bbox projection without a documented geometry model.
- **Measurement uncertainty is anisotropic and centralized.** Per-observation covariance is
  `R = J·Σ_px·Jᵀ + R_cal` in m², computed only in `app/tracking/world/observation_model.py`.
  No Jacobian or covariance math anywhere else. See the cts-spatial-fusion skill.
- **Fuse random noise, never systematic bias.** Inverse-covariance fusion shrinks covariance
  ~1/N; apply it only to the random term and add a non-shrinking bias floor from calibration
  residuals. Fusing the bias term away makes the estimate jump on camera-set changes.

### Identity evidence and gallery governance

For the complete authority, provenance, correction, and ReID lifecycle contract, load
`/home/sriram/code/nanai/continuous-tracking/.claude/skills/cts-identity-governance/SKILL.md`.
That focused skill is normative for identity work; this section remains the architectural summary.

- **Evidence provenance is mandatory.** Source, confidence, quality, model version, timestamp, and tracklet identifier on every record.
- **Direct face evidence is distinct.** Direct ArcFace, propagated hints, operator corrections, and ReID matches are separate evidence source types with different weights.
- **Temporal prior cannot create identity.** It maintains an existing identity within a bounded window; new assignment requires sensory or operator evidence.
- **Duplicate active identity is a guarded invariant.** `IdentityResolver.resolve()` owns the occupancy check over all open PHs (`open_ph_identities` carries incumbents not observed this frame). A second active PH must not newly acquire an identity already held by another open PH unless it has strong direct recognized face evidence for that identity. Shadow mode reports `cts_identity_shadow_mismatch_total{feature="duplicate_active_identity"}` before enforcement.
- **Gallery labels are governed.** Tentative commits → quarantine → promotion after stable evidence, not immediate trust.
- **Revisions are idempotent.** Stable IDs and evidence summaries so CC can apply once and explain in the UI.
- **Spatial gates fail closed: evidence without a floor point may match at room granularity with a confidence haircut, and evidence with neither floor point nor room matches nothing. Proto3 scalar floats cannot represent "unset"; any optional value on the wire carries an explicit presence flag, and (0,0) is a real position, never a sentinel.**
- **External identity evidence (`FaceAnchor.origin != 'native'`) is its own grade: it votes in the likelihood with a configured scale and real quality/frontality, and it is explicitly excluded from ArcFace authority, the independent-evidence clock, duplicate-identity ranking, and association vectors. Absent metadata defaults conservative (worst-case yaw, mid quality), never perfect. New external sources ship shadow-first with outcome metrics before they may influence the posterior.**

### Clinical and dementia signals

- **Signals are not diagnoses.** Vision-derived patterns are behavioral alerts and caregiver context, not a diagnosis of any medical condition. Every signal carries a non-diagnostic disclaimer.
- **Every detector has an algorithm spec.** Name, version, evidence grade, required inputs, data quality minimums, thresholds, and cited rationale.
- **Gate on data quality.** Identity confidence, observation coverage, baseline sample count, and calibration availability before emitting or escalating.
- **Cold starts are conservative.** Without a resident baseline, cap severity. Absolute safety thresholds require caregiver opt-in.
- **Use elapsed time, not frame count.** Stillness and dwell computations accumulate seconds from timestamps, capping large inter-observation gaps.
- **Prefer room taxonomy over name matching.** Configured room types from CC. Name-substring fallback must be marked in context with reduced confidence.

### Cross-service contracts

- **Proto changes are two-repo changes.** Update bindings, subscribers, publishers, tests, and docs in both repos in the same change set.
- **Do not reuse proto field numbers.** Deprecated fields are reserved. New fields get new numbers. Keep compatibility readers during migration.
- **Wire-field rename discipline.** When a serialized field is renamed (preferred: finalize the rename in the `.proto` file so the generated `_pb2.py` uses the new name), update the proto, regenerate bindings in both repos (`make proto-py`), and update every producer and consumer in the same change set. After the rename the deprecated field name must not appear anywhere in source code except: (a) `reserved` declarations in the proto, (b) `Detection.global_track_id` which remains as a deprecated wire alias for older orchestrators until a dedicated cleanup change. No business logic, no second module, and no published message may read or emit a deprecated field name once the rename is finalized. Enforce with a name-contract test (`tests/contracts/test_ph_contract_names.py`). The approved-boundary list in that test is the authoritative registry of permitted legacy references.
- **Decode-boundary rule.** A wire field may carry a deprecated name during a rename, but it must be decoded into its PH-native name at exactly one boundary, and no business logic may read the deprecated name. After decoding, only the canonical name propagates. The `test_ph_contract_names.py` enforcement verifies that no non-approved module reads or emits a deprecated field name.
- **Redis Streams stay protobuf-only.** No JSON, no base64, no dual-codec.
- **Shadow before authority.** New tracking, identity, or signal algorithms run in shadow mode with mismatch metrics before becoming authoritative.

### Cross-camera dedup pattern

The pre-association floor-point dedup pass (`app/tracking/world/dedup.py`) runs before `associate()` inside `WorldTracker.step()`. Rules:

1. **Different-camera gate.** Only pairs from different cameras are candidates; same-camera pairs are always separate observations.
2. **Geometric gate.** Both detections must have `calibrated=True` floor points within `dedup_max_distance_m` (default: 0.6 m) of each other.
3. **No-face-conflict gate.** When `dedup_require_no_face_conflict=True` (default), pairs where both detections carry committed and conflicting face anchors are not merged.
4. **Representative fusion is inverse-covariance, not crop-quality mean.** The cluster
   representative's floor point and covariance come from information-form fusion of the
   members' random covariances plus a bias floor (see cts-spatial-fusion). Crop quality is no
   longer the fusion weight (it remains an input to Σ_px scaling only).

The union-find algorithm identifies connected components; each cluster produces a representative
observation with a fused floor point and covariance. `associate()` keeps its 1-to-1 contract
because only representatives enter it; cluster membership is passed separately so the winning PH
can be updated with all contributing camera IDs.

Adding a new dedup gate condition: modify `dedup_observations()` only; do not add gate logic to `associate()` or `WorldTracker.step()`.

`WorldTracker.step()` must receive the same-time observations from overlapping cameras in a single call. If the frame pipeline batches detector inference across cameras but then runs world tracking per camera, cross-camera dedup is bypassed and duplicate unknown PHs will grow quickly. Keep post-detection batching intact through `WorldTrackingStage`, then split only downstream side effects that are camera-local.

PH keyframes are evidence, not synthetic references. `get_keyframes()` reads `continuous_tracking.tagged_keyframes` and returns real `minio_key` values. Never fabricate object keys from observation timestamps; broken images make caregiver correction unsafe.

### Quality-capture rule

Observation and PH quality is computed by the single `CropQuality` scorer (`app/pipeline/crop_quality.py`). There is no second scorer. The same scorer is used for:
- Per-observation quality scores that scale `Σ_px` in the observation model.
- The PH's `mean_quality` field (exponential moving average, updated in `WorldTracker.step()`).
- Gallery entry quality used by the identity resolver.

`mean_quality` travels to CC via the `IdentitySnapshot` proto field and then through the CC's `PersonLocationEnvelope` as the `quality` field. Do not compute quality on the CC side; read it from the proto.

### No-silent-fallback rule for stream consumers

A stream consumer that cannot process a message must:
1. XACK the message to keep the pending-entry list clean.
2. Log at `warning` level with the error details.
3. Increment a dead-letter counter in `app/observability/metrics.py`.

Never silently drop a message or return a fabricated value. The failure path must be observable.
Gap in the code today: the `frames.ready` consumer (`app/transport/redis_streams.py`) logs a
decode failure and skips the message without an XACK, and no dead-letter counter exists yet.

Ruff `BLE001` (blind-except) is selected in `tracking-orchestrator/pyproject.toml`; a catch-all is
permitted only with a per-site `# noqa: BLE001` and a documented reason. Every other new
`except Exception` is a CI failure.

### Stable signal IDs

Dementia signals use `uuid.uuid5(NAMESPACE_DNS, "{identity_id}\x00{signal_kind}\x00{window_start}\x00{window_end}")` so the same detection window always maps to the same UUID; this makes the `ON CONFLICT` upsert idempotent on retry. The namespace constant is `_SIGNAL_NS` in `app/trajectory/dementia_signals.py` and `app/services/identity_rewriter.py`. cognitive-companion's `IdentityRewriter` replicates the derivation in `cognitive-companion/backend/services/cts/signal_store.py::derive_signal_id`, pinned by a cross-repo golden-vector test on both sides: change both together.
