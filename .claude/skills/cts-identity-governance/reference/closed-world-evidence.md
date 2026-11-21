# Closed-world identity evidence

How `ClosedWorldChannel` (`app/tracking/identity/closed_world_channel.py`) uses household-wide
presence and simultaneous multi-person observations to shape Bayesian identity priors without ever
committing an identity on its own.

## Core invariants

- **Priors only (`counts_as_commit_evidence = False`).** The `CLOSED_WORLD` evidence grade
  (`app/tracking/identity/evidence_grades.py`) sets `counts_as_commit_evidence = False`,
  `grants_authority = False`, `refreshes_evidence_clock = False`, `ranks_in_duplicate_guard = False`,
  `enters_association = False`, `enters_contradiction = False`, and `propagates = False`. An Unknown
  PH with only closed-world support stays Unknown even when a single resident remains after
  elimination and the posterior clears `commit_prob` and `commit_margin`.
- **Roster-completeness gate.** The channel is active only when the latest `HouseholdPresence`
  snapshot from `cc.household_presence` reports `roster_complete == True`. Otherwise it returns an
  empty distribution and increments `cts_closed_world_inactive_total{reason="roster_incomplete"}`
  (or `no_presence` / `no_identities`). If Cognitive Companion cannot reach
  `person-identification-service`, it publishes `roster_complete = false` so CTS fails closed.
- **Explicit guest slot (`UNKNOWN`).** Base `UNKNOWN` factor is `1.0` (never penalized). After
  normalizing over all enrolled residents plus `UNKNOWN`, if the normalized `UNKNOWN` share falls
  below `closed_world.guest_floor` (`0.10` normally, or `closed_world.guest_floor_visitors = 0.30`
  when `visitors_present` is true), `UNKNOWN` is raised to that floor and resident shares are
  rescaled proportionally.

## Physical factors

For PH `j` and enrolled identity `i`, `f_ij = clamp(home_i * elim_ij, factor_floor, factor_ceiling)`:

1. **Home status (`home_i`):** `away_factor` (`0.3`) only when CC reports `PresenceStatus.AWAY`
   with `confidence >= away_min_confidence` (`0.8`) AND the snapshot is not stale
   (`age <= presence_stale_after_s = 180.0 s`) AND `i` has no fresh non-parked bound wearable fix
   within `wearable_home_window_s` (`600.0 s`, tracked per member in
   `WearableAssociationTracker` in fix capture time). A parked tag does not override `away`; a
   fresh non-parked fix only raises `home_i` back to `1.0` and never penalizes anyone else.
   Stale snapshots (`> 180 s`) treat every member as `UNKNOWN` (`home_i = 1.0`).
2. **Spatially verified joint elimination (`elim_ij`):** `occupied_factor` (`0.2`) only when
   another open PH `k != j` holds `i` AND all four guards hold:
   - **Fresh non-room-wearable evidence:** `k` has independent evidence within
     `occupancy_fresh_s` (`60.0 s`) that is not solely room-class wearable evidence.
   - **Simultaneous observation:** neither `j` nor `k` is coasting during `ph_close_grace_s`, and
     `abs(last_seen_j - last_seen_k) <= simultaneous_obs_max_skew_s` (`1.0 s`, capture time).
   - **Proven spatial separation:** either (a) both `j` and `k` have calibrated floor positions
     with no active `CoPresenceLink` between them and floor distance `>= min_elimination_distance_m`
     (`2.0 m`), or (b) both have known different `room_id`s with no registered `OverlapGroup`
     connecting cameras in those rooms (or their observed cameras).

## Rollout and observability

- Ships under `closed_world.mode` (`off`, `shadow`, `enabled`; `settings.yaml` default `shadow`).
- Shadow mode evaluates the counterfactual with identical `commit_evidence_ids` and records
  `cts_closed_world_shadow_outcomes_total{outcome}` (`would_name_unknown`, `would_change`,
  `agrees`) plus `cts_identity_shadow_mismatch_total{feature="closed_world"}`.
- Activity and inactive reason are reported on `GET /internal/identity/readiness` (`closed_world`)
  and surfaced on the `roster` step of Cognitive Companion's identity readiness service.
