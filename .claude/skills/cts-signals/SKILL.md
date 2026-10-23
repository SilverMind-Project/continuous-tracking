---
name: cts-signals
description: Use when adding or modifying a CTS dementia signal, detector, baseline, robust z-score rule, hysteresis behavior, severity policy, AlgorithmSpec, protobuf kind, CC plumbing, or signal test.
paths: ["**/tracking-orchestrator/app/trajectory/**", "**/proto/**"]
---

# CTS Signals

This skill makes "add or modify a dementia signal" a recipe. Ten protobuf kinds are active: `pacing`, `sundowning_index`, `bathroom_dwell_anomaly`, `nighttime_movement`, `stillness_anomaly`, `absence`, `gait_slowing`, and `agitation_index` (computed by `DementiaSignalWorker`), `fall_suspected` (`FallDetectionStage`, `app/pipeline/stages/fall_detection.py`), and `same_clothes_suspected` (`AppearanceEvaluator`, `app/trajectory/appearance_profile.py`, always `info`, off by default under `hygiene.same_clothes.enabled`).

## The golden rule

> A new windowed behavior signal is a new detector method on `DementiaSignalWorker`, never a new `elif` branch in `_process_identity`. Acute per-frame signals that require pipeline-only inputs, such as `fall_suspected`, belong in a pipeline stage and still follow the kind-plumbing and metadata rules below.

Files:
- Worker: `tracking-orchestrator/app/trajectory/dementia_signals.py`
- Specs: `tracking-orchestrator/app/trajectory/signal_specs.py`
- Stats: `tracking-orchestrator/app/trajectory/stats.py`
- Baseline protocol + InMemory: `tracking-orchestrator/app/storage/signals.py`

## Anatomy of a signal

```
DementiaSignalWorker.run_once(now)
  _get_tracked_identities()           # fetch identity IDs with points in window
  asyncio.gather(_process_identity * N, semaphore=max_concurrent_identities)
    _process_identity(identity_id, now)
      incremental window fetch + rolling state merge
      _compute_signals(identity_id, window, dwells, now)
        _compute_pacing()
        _compute_sundowning()
        _compute_bathroom_dwell_anomaly()
        _compute_nighttime_movement()
        _compute_stillness_anomaly()
        _compute_absence()
        _compute_gait_slowing()
        _compute_agitation_index()
        _check_data_quality()         # demote warning/emergency to info if coverage < 0.1 or confidence < 0.3
        _apply_algorithm_metadata()   # stamp AlgorithmSpec fields onto DementiaSignal
      _clear_inactive()               # reset hysteresis for signals no longer triggered
    signal_repo.upsert_signal() each signal
    evict stale rolling state (> 2x window duration)
```

Each detector method:
1. Computes a scalar metric from the window (rate, duration, count).
2. Calls `_compute_z_score(value, signal_kind, identity_id, now)` to compare against baseline.
3. Maps z-score to severity.
4. Calls `_hysteresis.should_emit(identity_id, signal_kind, severity, now, cooldown_minutes)`.
5. Returns a `DementiaSignal` with a UUID5 stable ID, or returns empty and calls `clear_trigger`.

## The baseline contract

`BehaviorBaselineRepository` (`app/storage/signals.py`) provides three methods:

| Method | Returns | Feeds |
|--------|---------|-------|
| `dwell_durations(identity_id, room_predicate, since, until)` | `list[float]` of closed-dwell durations (seconds) | `bathroom_dwell_anomaly`, `stillness_anomaly` |
| `hourly_activity(identity_id, since, until)` | `dict[int, HourlyActivitySummary]` keyed by hour-of-day 0-23 | `pacing`, `sundowning_index`, `nighttime_movement`, `absence` |
| `stillness_episodes(identity_id, since, until)` | `list[StillnessEpisode]` | `stillness_anomaly` |

`HourlyActivitySummary` fields: `transition_count: int`, `observed_minutes: int`. These are aggregated across all days in the query range, not per calendar day -- `hourly_activity(since=now-14days, until=now)` sums every 2:00 PM slot over 14 days into `result[14]`.

**Three iron rules:**

1. **Baselines derive from raw behavior tables, never from emitted signals.** `BehaviorBaselineRepository` queries `person_trajectories` and `room_dwells`, not `dementia_signals`. Using prior signal values as a baseline creates a circularity: the baseline shifts toward whatever the detector already fires on, making the z-score converge toward zero over time. Audit note: the pacing baseline once read emitted pacing signals.

2. **The current value must never be in its own baseline sample set.** The current observation window is excluded from the `since`/`until` query range. A signal that includes its own measurement in the denominator will never trigger -- the z-score is 0 by construction. Audit note: the baseline window once included the current day.

3. **Baseline samples must be in the same units and window shape as the compared value.** If you compare a 24-hour transition rate, the baseline samples must each be 24-hour transition rates, not hourly counts or raw point counts. Audit note: the pacing baseline once compared per-minute rates against per-30-minute counts.

## Statistics

`robust_z(value, samples)` in `app/trajectory/stats.py` is the only sanctioned z-score function. It computes the modified z-score using median and MAD (median absolute deviation):

```
modified_z = 0.6745 * (value - median(samples)) / MAD(samples)
MAD = median(|x_i - median(samples)|)
```

The 0.6745 constant normalizes MAD to be equivalent to standard deviation for Gaussian data.

Special case: if `MAD == 0` (all samples identical) and `value != median`, `modified_z = inf`. If `value == median`, `modified_z = 0`.

Returns a frozen `RobustZ(median, mad, modified_z, n)` dataclass.

**Minimum baseline gating.** `_compute_z_score` checks `len(samples) >= min_baseline_n` before calling `robust_z`. If there are fewer samples, it returns `(baseline=None, z_score=None)`. Each signal's spec defines its own `min_baseline_n` (see AlgorithmSpec table below). Use the spec value, not the global `SignalConfig.min_baseline_n`.

**Cold-start fallbacks by signal:**

| Kind | Cold-start behavior | Rationale |
|------|---------------------|-----------|
| `pacing` | `rate > 0.3 tpm` -> emergency, `> 0.15` -> warning (absolute thresholds) | Acute/visible; caregiver needs immediate notification even without history |
| `bathroom_dwell_anomaly` | `duration >= 2700 s (45 min)` -> emit; severity capped at warning | 45 min is an unambiguous fall/medical concern independent of baseline |
| `nighttime_movement` | `transitions >= 5` -> emergency, `>= 3` -> warning | Absolute thresholds meaningful for nighttime safety |
| `stillness_anomaly` | Cold-start caps emergency -> warning (cannot escalate to emergency without baseline posture context) | Emergency requires both long duration AND lying posture history to confirm |
| `sundowning_index` | Silent if `today_rate < 0.03`; emit info otherwise | Trend signal; meaningless without enough evening history to compare |
| `absence` | Emits from `absence_threshold_minutes` with tiers at +60 and +120 min (see the severity table) | Camera coverage context needed only to contextualize, not to gate |

**Trend and experimental signals** must stay silent in cold start (no fallback severity). Acute signals (falls, bathroom, nighttime) may use fallback severity with a cap at warning.

## Hysteresis API

`SignalHysteresis` in `dementia_signals.py` prevents alert fatigue across `run_once` calls:

**`should_emit(identity_id, signal_kind, severity, now, cooldown_minutes, episode_key="") -> bool`**

Returns True if the signal should be emitted. Internal logic in order:

1. **Severity escalation within an open episode**: always emits immediately (bypasses debounce and cooldown). Severity within an episode is monotonically increasing.
2. **Equal severity within an open episode**: returns False (idempotent upsert handles persistence; no re-alert).
3. **Onset debounce**: trigger must hold for `min_consecutive` consecutive runs (default: 2) before first emission.
4. **Cooldown**: after emission, the same `(identity_id, signal_kind)` will not re-emit for `cooldown_minutes` (default: 60).

**`clear_trigger(identity_id, signal_kind)`**: resets the consecutive counter and closes the episode. Call this when a detector finds no signal (e.g., bathroom dwell closed normally, nighttime transitions below threshold).

**`close_episode(identity_id, signal_kind)`**: closes the episode only (does not reset consecutive counter). Used when the condition resolves but tracking should resume immediately.

**Rules:**
- A detector method must never read or write `_consecutive_count`, `_last_emission`, or `_episode_severity` directly. These are hysteresis private state. Always go through `should_emit` / `clear_trigger`.
- The `cooldown_minutes` argument to `should_emit` comes from `self._cfg.cooldown_minutes`. All kinds share the same cooldown value in the current config; pass it through cleanly so per-kind overrides can be added later without touching the hysteresis class.

## Severity policy

| Severity | Caregiver meaning | Reserved for |
|----------|-------------------|--------------|
| `info` | Interesting pattern, no action needed | Mild z-score deviation; cold-start fallback signals |
| `warning` | Requires attention within hours | Sustained above-baseline behavior; falls-risk posture with long dwell |
| `emergency` | Act within minutes | The top tier of each kind below |

Tiers per kind, as implemented in `app/trajectory/dementia_signals.py`:

| Kind | With a baseline | Cold start |
| --- | --- | --- |
| `pacing` | Absolute rate, always: info; warning above 0.15 transitions per minute; emergency above 0.3 (the z-score is recorded, not used for tiering) | same |
| `sundowning_index` | z at least `sundowning_z_threshold`: info; z at least 3.0: warning; z at least 4.0: emergency | silent if today's rate is below 0.03, else info |
| `bathroom_dwell_anomaly` | z at least 4.0: warning; z at least 5.0: emergency; else info | warning when the open dwell reaches `bathroom_absolute_threshold_seconds` (default 2700 s); never emergency |
| `nighttime_movement` | z at least 3.0: warning; z at least 4.0: emergency; else info | gate, gate+2, gate+4 transitions: info, warning, emergency (`nighttime_transition_threshold`) |
| `stillness_anomaly` | lying posture: warning, then emergency at `stillness_emergency_minutes` | emergency demoted to warning |
| `absence` | `absence_threshold_minutes`, +60 min, +120 min: info, warning, emergency; demoted one tier when historically absent at this hour more than half the time | same |
| `gait_slowing` | warning when the decline exceeds 20% or the recent median is below 0.6 m/s, else info | silent (needs 10 baseline and 10 recent days) |

**Rules:**
- Severity tiers derive from configured thresholds (`SignalConfig`, read from the `signal:` block of `config/settings.yaml`), never literals in a detector body. Every cut point above is a field (for example `sundowning_warning_z`, `pacing_emergency_rate_per_min`, `absence_warning_extra_minutes`); `tests/unit/trajectory/test_signal_severity_config.py` pins the defaults, so changing one is a behavior change that updates this table in the same commit.
- A new kind declares its tiers in this table and caps cold-start severity at `warning` unless an absolute safety threshold justifies more.
- Data quality demotion: if `identity_confidence_mean < 0.3` OR `coverage_ratio < 0.1`, `warning` and `emergency` are demoted to `info` before emission. The signal is still emitted (so the upsert records it), but caregiver notifications should filter on severity.
- Dementia signals may gate on persisted trajectory confidence fields:
  `position_sigma_m`, `primary_camera_id`, `contributing_camera_count`, and
  `footpoint_reliable`. Low-confidence trajectory points should not escalate
  pacing/wandering.

## Stable IDs and idempotency

**UUID5 rule** (from CLAUDE.md):

```python
_SIGNAL_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_DNS

signal_id = str(uuid.uuid5(_SIGNAL_NS, f"{identity_id}\x00{signal_kind}\x00{window_start.isoformat()}\x00{window_end_str}"))
```

Use `_stable_signal_id(identity_id, signal_kind, window_start, window_end)` -- do not inline this.

**Open dwell convention**: signals for events still in progress (open bathroom dwell, open absence gap) use `window_end="open"` as the string sentinel. This distinguishes them from a closed dwell that ended at the same instant the worker ran.

**Upsert semantics**: `signal_repo.upsert_signal` performs `ON CONFLICT (signal_id) DO UPDATE SET ...`. Running `run_once` twice for the same window produces one database row. Callers must not attempt deduplication before calling `upsert_signal`.

## Kind plumbing checklist

When adding a new `DementiaSignalKind`, update all of these in the same PR:

1. **`cts-contracts` first**: add the kind to `DementiaSignalKind` in `cts_contracts/signals.py` (the shared vocabulary repository), then bump the `cts-contracts` pin in both CTS and CC. The CTS `DementiaSignalKind` Literal in `tracking-orchestrator/app/domain/__init__.py` is pinned to it; add the string there too.

2. **`_SIGNAL_SPEC` constant and `AlgorithmSpec`** in `tracking-orchestrator/app/trajectory/signal_specs.py` -- create a module-level `MY_KIND_SPEC` and add it to the `_SIGNAL_SPEC` dict. Fields to fill:
   - `name`: `"my-kind-v1"` (slug + version, incremented when detection logic changes)
   - `version`: integer, incremented on logic changes
   - `evidence_grade`: one of `clinical_review | observational_study | caregiver_guidance | local_baseline_only | experimental`
   - `clinical_label`: human-readable name for clinical documentation
   - `min_baseline_samples`: how many history points before z-score is valid
   - `required_inputs`: tuple of data source names (for documentation)

3. **Settings knobs + rationale comments** in `SignalConfig` (`dementia_signals.py`). Every threshold exposed in config must have a comment explaining its clinical or operational rationale. Hardcoded thresholds that vary by deployment go in `SignalConfig`, not in the detector body.

4. **Protobuf enum value** in `proto/continuoustracking/v1/signals.proto` -- add `DEMENTIA_SIGNAL_KIND_MY_KIND = N;` then run `make proto` to regenerate bindings.

5. **CC kind mapping** in `cognitive-companion/backend/services/cts/subscriber.py` -- add the proto enum -> string kind mapping.

6. **CC kind inventory grep** to verify all references are consistent:
   ```bash
   grep -rn "my_kind" cognitive-companion/backend/ cognitive-companion/frontend/ -l
   ```
   Verify the kind string appears in: `cognitive-companion/backend/services/cts/signal_config.py` (`ALL_SIGNAL_KINDS` derives from `cts_contracts`), `cognitive-companion/backend/services/cts/signal_narratives.py` (narrative text), and `cognitive-companion/backend/filters/builtin/dementia_signal.py` (kind list).

7. **CC frontend kind inventories** -- grep for kind arrays and maps in `cognitive-companion/frontend/src/views/`, `cognitive-companion/frontend/src/components/pipeline/steps/index.js`, and Tracking panels. Add labels/icons where the surface has a kind-specific map, while preserving the generic unknown-kind fallback.

8. **Dementia-signals docs table row** in the portal docs if a dementia-signals reference page exists.

9. **Non-diagnostic disclaimer** -- use `NON_DIAGNOSTIC_DISCLAIMER` from `signal_specs.py` as the `disclaimer` field on every `AlgorithmSpec`. Do not write a custom disclaimer unless legal has reviewed it.

## Test matrix

Every new signal must have all of the following tests. Use `InMemoryTrajectoryRepository` and `InMemoryBehaviorBaselineRepository` -- no database required.

| Test | What it verifies |
|------|-----------------|
| fires-on-synthetic-positive | Detector returns a signal with correct kind and severity >= warning when metrics clearly exceed threshold |
| silent-on-confound | Detector returns empty given the kind-specific confound (e.g., bathroom dwell that ended normally; pacing with < 8 room changes; nighttime points in daytime hours) |
| cold-start behavior | Correct fallback when baseline repo has fewer than `min_baseline_n` samples: fires with capped severity OR stays silent per the cold-start policy table above |
| baseline self-exclusion | Baseline samples with `since` / `until` that exclude the current window; verify that including current window would change the z-score |
| hysteresis debounce | First trigger fires False from `should_emit` (debounce count = 1 of 2); second consecutive trigger fires True |
| hysteresis cooldown | After emission, a second trigger within `cooldown_minutes` returns False; a trigger after cooldown expires returns True |
| severity tier boundaries | Parametrized: at each threshold boundary, severity is exactly as expected (use `@pytest.mark.parametrize` over boundary values) |
| data-quality demotion | When `identity_confidence < 0.3` or `coverage_ratio < 0.1`, a warning-level trigger is demoted to info |
| timezone boundary case | If the signal is time-windowed (sundowning: 17:00-22:00, nighttime: 22:00-06:00), verify points exactly at the boundary hour are included or excluded as specified |

See `tests/test_dementia_signals.py` for existing patterns. Point factory helpers `_point(room, offset_minutes)` and `_dwell(room, entered_offset, duration_seconds)` can be copied.

## Verification commands

```bash
# Full quality gate (Python lint + type check + tests)
cd continuous-tracking && make check

# Signal tests only
tracking-orchestrator/.venv/bin/pytest tests/test_dementia_signals.py -v

# New kind by name
tracking-orchestrator/.venv/bin/pytest tests/test_dementia_signals.py -k "my_kind" -v

# Verify kind is plumbed everywhere in CC
grep -rn "my_kind" cognitive-companion/backend/ cognitive-companion/frontend/ -l

# Check proto bindings are current
cd continuous-tracking && make proto-lint
```
