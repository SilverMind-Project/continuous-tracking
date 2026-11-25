# Wearable Identity Evidence Governance

Wearable localization fixes (`wearable.fixes` protobuf `TagFix`) are consumed by
`tracking-orchestrator/app/services/wearable_fix_subscriber.py` into
`tracking-orchestrator/app/services/wearable_fix_buffer.py`, associated with open Person
Hypotheses by `tracking-orchestrator/app/tracking/world/wearable_association.py`, and
scored into the Bayesian resolver by
`tracking-orchestrator/app/tracking/identity/wearable_channel.py`.

## Evidence grade (`WEARABLE`)

Registered once in `tracking-orchestrator/app/tracking/identity/evidence_grades.py`:

| Flag | Value | Why |
| --- | --- | --- |
| `grants_authority` | `False` | Wearable never overrides operator or calibrated direct-face authority |
| `refreshes_evidence_clock` | `True` | Independent non-visual sensory evidence refreshes `last_independent_identity_evidence_at` using `newest_fix_at` |
| `ranks_in_duplicate_guard` | `True` | Strong wearable evidence (`p_w >= wearable.duplicate_guard_min_probability`, 0.90) ranks contenders when an identity is unoccupied |
| `grants_duplicate_bypass` | `False` | Never bypasses an incumbent open PH holding that identity |
| `enters_association` | `False` | Wearable identity evidence never alters camera-to-PH association. The separate, default-off spatial fusion path (`wearable.spatial_fusion.mode`) may move PH state; see "Spatial fusion" below |
| `uses_own_capture_time` | `True` | Clock refresh uses the physical `newest_fix_at`, never round wall time |

## Covariance-first routing and parked/multi-tag guards

- **Zero vendor or technology branching.** Association inspects only fix kind, covariance
  major-axis sigma (`sqrt(lambda_max(Sigma_f)) <= wearable.trajectory_max_sigma_m`, default
  `0.75 m`), room polygon mass, `wear_state`, and `motion_state`.
- **Room-id namespace.** Room polygons, RF nodes, and every fix's `room_id` use Cognitive
  Companion room ids (numeric strings). Wearable association resolves a PH's room from the
  polygons, then from `camera_room_ids` (camera id to room id, every bound camera, built by
  `WorldTrackingStage` from `CameraRoomMap`), and never from the tracker's
  `metadata["last_room_id"]` or `camera_room_map`, which hold room *names* for any PH outside
  every polygon (all uncalibrated cameras). Tests must use a numeric room id on the fix and a
  display name in the camera map; matching strings on both sides hides a namespace bug.
- **Uncalibrated PH safety and `FIX_KIND_POSITION` room fallback.** `tracking-orchestrator/app/tracking/world/ph_floor_history.py`
  records samples only on rounds updated by calibrated observations. Uncalibrated PHs never
  enter trajectory scoring; they use room-class scoring only when they are the sole open PH in
  the room for the entire sliding window (`wearable.window_s`, 10 s). Room-class scoring runs
  over every fix of a tag, `FIX_KIND_POSITION` included: a position fix reduces to room mass
  over the polygons, or to its `room_id` when CTS has no polygons, so 4-anchor UWB rooms give
  room-grade evidence on cameras awaiting homography calibration.
- **Off-body and parked tags.** `WEAR_STATE_OFF_BODY` is dropped at subscriber decode. A tag
  with no room change and stillness (explicit `STILL`, or `UNSPECIFIED` positions staying
  within their own 95% covariance ellipse) for `wearable.parked_after_s` (600 s) is marked
  parked and yields zero evidence until `MOVING` or a room change. A tag already in a room
  before a PH appears earns no room-class evidence for that PH until it moves after the PH
  appeared (a respawned PH of a seated resident waits for the watch to report motion).
- **Multiple tags per member and physical devices.** Fixes are grouped by bound `member_id`
  before joint assignment, and each member is scored by its best single tag (never summed
  across tags). Logical tags of one radio (a dual-mode watch's 1 Hz BLE beacon `tag_ble` and
  its duty-cycled UWB `tag_uwb`) share a Cognitive Companion `physical_device_id`, which the
  subscriber stamps on each fix (`WearableFix.device_key` is that id, or `tag:<tag_id>` for a
  single-tag device). CC guarantees that a device has one tag per adapter and that its bound
  tags are bound to one member.
  - **One segment per physical device.** A segment survives a switch between tags of the same
    member and the same device, keeps the tag that opened it as `segment.tag_id` (the
    provenance key; the repository never rewrites it), and never downgrades its latched
    grade. `WearableEvidence.tag_id` is the tag whose fixes produced that round's evidence. A
    switch to another member, or to another physical device of the same member (a pendant
    left on a table), closes the segment with `reassigned`.
  - **Disagreement across devices only.** `_check_tag_disagreement` gives each physical
    device the room of its tightest-covariance tag and counts a device only when that room
    holds at least `wearable.tag_disagreement_min_room_share` (0.6) of its window mass. Two
    devices of one member disagreeing for `wearable.tag_disagreement_window_s` (10 s) emit
    `tag_disagreement` and yield zero evidence for the member. One radio never disagrees with
    itself.
  - **Swap counting per device.** Face conflicts count per physical device, so three
    conflicts split across a watch's two tags still mark `swap_suspected`.
- **Segment hold.** A segment whose PH is not accepted in a round is held, emitting no
  evidence, for up to `wearable.segment_hold_s` (75 s, longer than the watch's 60 s burst
  cooldown) after its last accepted round. Accepted again for the same member and device on
  the same PH, it resumes with its latched grade, and its room-attestation span restarts.
  The hold ends early (`member_moved`) when the member is accepted on another PH; on expiry
  the segment closes as `association_lost` with `ended_at` at its last accepted round. This
  is what carries a trajectory grade across a burst cooldown in a shared room, where BLE room
  evidence is never eligible.

## Graded attestation and gallery independence

Association and attestation take zero ReID gallery inputs. Persisted rows in
`continuous_tracking.wearable_associations` (migration `0008`) carry
`attestation_grade IN ('none', 'room', 'trajectory')`:

- **`trajectory`:** a trajectory streak spanning at least
  `wearable.trajectory_attest_min_duration_s` (10.0 s) with at least
  `wearable.trajectory_attest_min_fixes` (8) matched trajectory fixes, `window_llr >= 15.0`,
  both margins `>= 4.0`, zero face conflicts. The streak counts trajectory-class fixes only
  and restarts after a gap longer than `wearable.window_s` between matched trajectory fixes,
  so earlier room-class time on the segment and an earlier burst never count toward it.
- **`room`:** continuous room-class acceptance spanning at least
  `wearable.room_attest_min_duration_s` (60.0 s), sole open PH throughout, tag reported
  `MOVING` at least once in span, tag entered room within `wearable.room_entry_window_s`
  (20.0 s) of the PH appearing in it, zero face conflicts.

## Channel weighting and calibrated-face conflict rule

- `WearableChannel` maps `window_llr` via `min(cap, logistic((window_llr - llr_mid) / llr_scale))`
  (`p_max = 0.97` trajectory, `p_max_room = 0.90` room), builds a full-support distribution over
  all enrolled identities plus `UNKNOWN`, applies `wearable.weight_multiplier` (2.0) as a
  log-likelihood power (`p ** weight`) before normalization, and emits `support_multiplier = 1.0`.
- **Calibrated-face vs. wearable conflict:** When a qualifying calibrated direct-face anchor
  (`calibrated_confidence >= resolver.arcface_authority_calibrated_confidence`, 0.80) names
  identity A while accepted live wearable evidence on the same PH names identity B (`A != B`),
  the resolver forces `Unknown` with `conflict="wearable_face"`, increments
  `cts_wearable_face_conflict_total`, and breaks attestation on the active segment. Three or
  more conflicts for one tag within 10 minutes mark `swap_suspected = true` and log a warning
  carrying `tag_id` only (never a display name). An uncalibrated face (`calibrated_confidence`
  is `None`, e.g., an uncovered resident) never triggers this conflict.

## Spatial fusion (`wearable.spatial_fusion`)

Separate from the identity channel: it moves PH floor state, never identity.
`tracking-orchestrator/app/tracking/world/wearable_spatial_fusion.py` holds the pure math;
`WorldTracker._fuse_wearable_positions` runs it after association and before identity
resolution (the resolver writes identity through targeted repository operations, so a later
full PH save would overwrite them). Modes: `off` (default, no change at all), `shadow`
(measure only, no other effect), `enabled` (apply). Either non-off mode needs `wearable.mode` shadow or enabled.

- **Eligibility.** An open PH whose evidence this round carries the `trajectory` grade, with a
  `FIX_KIND_POSITION` fix from that evidence's tag no wider than `max_sigma_m` (0.35 m),
  within `max_dt_s` (3 s) of the PH state time, each fix fused once.
- **Time model.** Stored PH state is the posterior at `last_seen_at`. A fix older than that
  is moved to the state time along the PH velocity with `dt^2 * P_vv` added to its
  covariance; only a camera-occluded PH may be advanced by a newer fix (predict, update,
  `last_seen_at` moves to the fix time).
- **Covariance.** `R = Sigma_fix + b^2 I` (`bias_floor_m` 0.12 m), 2-dof gate `gate_chi2`
  (9.21), then the posterior's smallest position eigenvalue is lifted back to
  `min(b^2, its prior value)`: fused fixes never make a PH more certain than the wearable's
  systematic error.
- **Circular-feedback firewall.** In `enabled`, `PHFloorHistory` records each round's raw
  calibrated camera measurements (position and the covariance the Kalman update used)
  instead of the posterior, so trajectory association never scores a wearable against a
  state that already absorbed its own fixes. `shadow` never moves PH state, so it keeps the
  posterior history and changes no association, evidence, or attestation.
- **Coasting bridge.** In `enabled`, a trajectory-attested PH the camera lost may be
  advanced by fused fixes and stays open while its last fused fix is within
  `ph_close_grace_s + max_dt_s` and its last camera observation within
  `max(coasting_bridge_max_s, ph_close_grace_s)` (10 s). In practice it is shorter: once the
  window's fixes outrun the camera history, trajectory evidence and with it the bridge stop.
- **Metrics.** `cts_wearable_spatial_fusion_total{mode, outcome}` (`applied`, `bridged`,
  `gated`, `no_fix`) and `cts_wearable_spatial_fusion_shift_m{mode}`.

## Rollout and purge

- `wearable.mode` ships `"off"` (subscriber does not connect; resolver path is untouched).
  `"shadow"` evaluates associations, persists segments, and records counterfactual outcomes in
  `cts_wearable_shadow_outcomes_total` with zero effect on live identity.
- Identity purge (`PostgresGalleryRepository.purge_identity` and `IdentityRosterSyncService`)
  deletes `wearable_associations` rows, buffered fixes, and in-memory bindings for the purged
  identity. Unknown-PH purge cascades `wearable_associations` rows via `ON DELETE CASCADE`.
