# World Tracker Time Model and Coasting State

## Stored vs. predicted Kalman state

`PersonHypothesis.state_mean` and `PersonHypothesis.state_cov` always represent the Kalman
posterior at `PersonHypothesis.last_seen_at` (the timestamp of the latest observation that
updated the hypothesis).

- **Step 1 (Association prediction):** `WorldTracker.step` predicts each open PH from
  `ph.last_seen_at` to `now` (`batch_time`) transiently in memory (`predicted_states` and
  `predicted_by_ph`). Those predictions feed `associate()` and snapshot building, and are
  never written back to `PersonHypothesis` on their own.
- **Step 4 (Matched update):** When a PH matches an observation, the filter predicts from
   the stored state at `ph.last_seen_at` to `obs.captured_at` (or sequentially across
  distinct `captured_at` timestamps in a multi-camera dedup cluster), applies `update()`
  and ZUPT, and sets `last_seen_at = max(ph.last_seen_at, obs.captured_at)`.
- **Step 7 (Unmatched / coasting PHs):** `_advance_unmatched_ph` closes a PH when
  `(now - ph.last_seen_at).total_seconds() > ph_close_grace_s`. While within the grace
  window, `_advance_unmatched_ph` returns `ph` unchanged, and its `ph_id` is recorded in
  `WorldTrackerResult.coasting_ph_ids`. Because stored state is never overwritten by an
  unmatched prediction, coasting behavior is invariant to the empty-round rate.
- **Step 10 (Snapshots and transit suppression):** `WorldFrameSnapshot` for a coasting PH
  (`ph.ph_id in coasting_ph_ids`) uses the in-round prediction `predicted_by_ph[ph.ph_id]`
  for `floor_x_m`, `floor_y_m`, `floor_vx_m_s`, `floor_vy_m_s`, `position_sigma_m`,
  `floor_speed_m_s`, and room resolution, and sets `coasting = True`.
  `WorldTrackingStage._publish_transit_events` skips any PH in `coasting_ph_ids` so
  predicted motion never emits a room transition.

## Capture-time clock

Every timestamp on `PersonHypothesis` (`born_at`, `last_seen_at`, `closed_at`) and on
`WorldObservation` (`captured_at`) is a **capture-time** instant (`FrameContext.capture_time`,
derived from `FrameReady.capture_time_unix_ns`).

- `WorldTrackingStage.run_many` computes `batch_time = max(ctx.capture_time for ctx in contexts)`
  and passes `now = batch_time` to `WorldTracker.step`, CC assertion matching, face templates,
  and transit detection.
- `FrameContext.event_time` (`received_time_unix_ns`) is reserved for transport/pipeline lag
  diagnostics and is never mixed with `capture_time` in Kalman `dt`, grace-window checks, or
  revival age checks.

## Out-of-order and equal-time observations

Multi-camera batches or delayed frames can deliver an observation with
`obs.captured_at <= ph.last_seen_at`:

- `kalman.predict` returns the state unchanged (`dt = 0`) when `now <= state.updated_at`.
- `update()` folds the position measurement into the state at `dt = 0`.
- `last_seen_at` is updated via `max(ph.last_seen_at, obs.captured_at)` so it never moves
  backward.
- When `obs.captured_at < ph.last_seen_at`, the tracker increments
  `cts_worldtracker_out_of_order_observations_total{camera_id}`.

## Flagged motion and dedup extensions

- **`world_tracker.damped_velocity_prediction` (default `false`):** When enabled,
  `kalman.predict(..., damped=True)` integrates exponential velocity decay `dv/dt = -v / tau`
  over `dt`, giving `x' = x + v * tau * (1 - exp(-dt / tau))` and `v' = v * exp(-dt / tau)`,
  and propagates covariance through the damped transition matrix `F` (`F @ P @ F.T + Q`).
  Displacement is strictly monotonic in `dt` and bounded by `v * tau`.
- **`world_tracker.dedup_time_skew_speed_m_s` (default `0.0`):** Widens the pairwise
  cross-camera floor-distance dedup gate in `dedup._effective_distance_gate_m` by
  `dedup_time_skew_speed_m_s * abs((obs_i.captured_at - obs_j.captured_at).total_seconds())`,
  still capped by `dedup_max_distance_ceiling_m`. A multi-camera dedup representative carries
  `captured_at = max(o.captured_at for o in cluster)`.
