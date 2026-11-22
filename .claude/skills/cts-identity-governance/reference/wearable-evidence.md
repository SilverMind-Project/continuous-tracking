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
| `enters_association` | `False` | Wearable never alters camera-to-PH Kalman association |
| `uses_own_capture_time` | `True` | Clock refresh uses the physical `newest_fix_at`, never round wall time |

## Covariance-first routing and parked/multi-tag guards

- **Zero vendor or technology branching.** Association inspects only fix kind, covariance
  major-axis sigma (`sqrt(lambda_max(Sigma_f)) <= wearable.trajectory_max_sigma_m`, default
  `0.75 m`), room polygon mass, `wear_state`, and `motion_state`.
- **Uncalibrated PH safety and `FIX_KIND_POSITION` room fallback.** `tracking-orchestrator/app/tracking/world/ph_floor_history.py`
  records samples only on rounds updated by calibrated observations. Uncalibrated PHs never
  enter trajectory scoring; they use room-class scoring only when they are the sole open PH in
  the room for the entire sliding window (`wearable.window_s`, 10 s). When a `FIX_KIND_POSITION`
  fix cannot be trajectory-scored against an uncalibrated PH, its `fix.room_id` or floor-polygon
  containment (`room_polygons`) feeds room-class scoring so 4-anchor UWB rooms still provide
  room-grade evidence on cameras awaiting homography calibration.
- **Off-body and parked tags.** `WEAR_STATE_OFF_BODY` is dropped at subscriber decode. A tag
  with no room change and stillness (explicit `STILL`, or `UNSPECIFIED` positions staying
  within their own 95% covariance ellipse) for `wearable.parked_after_s` (600 s) is marked
  parked and yields zero evidence until `MOVING` or a room change. A tag already in a room
  before a PH appears earns no room-class evidence for that PH until it moves after the PH
  appeared.
- **Multiple tags per member and dual-mode (`tag_ble <-> tag_uwb`) segment continuity.** Fixes
  are grouped by bound `member_id` before joint assignment. Each member is scored by its best
  single tag (never summed across tags). When an active `WearableAssociationSegment` on a PH
  continues with the same `identity_id == pair.member_id` but switches between co-worn tags of
  that member (such as a dual-mode watch's `tag_uwb` 12-second UWB burst handing back to its
  continuous `tag_ble` room beacon), the segment remains open (`tag_id` updated in place) and
  never downgrades its latched `attestation_grade` (`trajectory > room > none`). Two non-parked
  tags of one member disagreeing on room for `wearable.tag_disagreement_window_s` (10 s) emit
  `tag_disagreement` and yield zero evidence.

## Graded attestation and gallery independence

Association and attestation take zero ReID gallery inputs. Persisted rows in
`continuous_tracking.wearable_associations` (migration `0008`) carry
`attestation_grade IN ('none', 'room', 'trajectory')`:

- **`trajectory`:** continuous trajectory-class acceptance spanning at least
  `wearable.trajectory_attest_min_duration_s` (10.0 s) and
  `wearable.trajectory_attest_min_fixes` (8) trajectory fixes, `window_llr >= 15.0`, both
  margins `>= 4.0`, zero face conflicts.
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

## Rollout and purge

- `wearable.mode` ships `"off"` (subscriber does not connect; resolver path is untouched).
  `"shadow"` evaluates associations, persists segments, and records counterfactual outcomes in
  `cts_wearable_shadow_outcomes_total` with zero effect on live identity.
- Identity purge (`PostgresGalleryRepository.purge_identity` and `IdentityRosterSyncService`)
  deletes `wearable_associations` rows, buffered fixes, and in-memory bindings for the purged
  identity. Unknown-PH purge cascades `wearable_associations` rows via `ON DELETE CASCADE`.
