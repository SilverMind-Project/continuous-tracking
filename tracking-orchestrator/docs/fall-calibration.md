# Fall Detection Threshold Calibration

## Overview

This document records the calibration harness output for the `fall_detection` threshold
parameters and the live-enable runbook required before setting `fall_detection.enabled: true`
in any deployment.

Fixture source: `tests/fixtures/fall_sequences/*.jsonl`.  Regenerate with:

```bash
cd tracking-orchestrator
uv run python scripts/synthesize_fall_sequence.py
```

Calibrate with:

```bash
uv run python scripts/calibrate_fall_thresholds.py
# Include privately recorded sequences (not in git):
uv run python scripts/calibrate_fall_thresholds.py --extra-dir /path/to/real_sequences
```

---

## Shipped thresholds

```yaml
fall_detection:
  enabled: false   # enable per-deployment only after following the runbook below
  max_descent_rate_hps_threshold: 0.8
  height_ratio_threshold: 0.55
  lying_score_threshold: 0.4
```

---

## Harness summary (synthetic fixtures only)

**Date:** 2026-06-11  
**Fixtures:** 10 (4 positive, 5 no-detect, 1 warning-max)

| Fixture | Expectation | Detected | Confirmed emergency |
|---|---|---|---|
| fall_forward_fast | detect | YES | no |
| fall_low_fps | detect | YES | no |
| fall_with_pose_loss | detect | YES | no |
| fall_slump_slow | warning-max | YES | no |
| sit_down_normal | no-detect | NO | no |
| sit_down_heavy | no-detect | NO | no |
| lie_on_bed | no-detect | NO | no |
| bend_to_pick_up | no-detect | NO | no |
| tie_shoes | no-detect | NO | no |
| child_or_pet_proxy | no-detect | NO | no |

**Sensitivity (detect fixtures):** 100.0% (3/3)  
**Specificity (no-detect + warning-max, by confirmed escalation):** 100.0% (7/7)

"Confirmed escalation" means `is_escalatable() = True` when `post_event_motion_nu_s`
is non-None — i.e., the 2-second post-event motion window has closed and the person
remains immobile.  The posture-proxy branch (when the window is still open) can fire
earlier but is not counted as a confirmed emergency for calibration purposes.

**Grid result:** all 60 threshold combinations (descent_rate × height_ratio × lying_score)
produce 100% / 100% on synthetic fixtures.  The grid is most useful when real recorded
sequences are added via `--extra-dir`; synthetic fixtures alone are not sufficient to
discriminate between threshold choices.

---

## Fixture-level notes

### Positive: fall_slump_slow

This is the "hard case" — a two-phase slump with only a ~0.92 hps peak rate.  At the
shipped 0.8 threshold it detects at warning.  Post-event motion (0.095 nu/s > 0.05
stillness floor) prevents confirmed escalation to emergency.

If a threshold of 1.0 hps is preferred (for lower false-positive risk in noisy
deployments), `fall_slump_slow` would be a false negative.  This trade-off should be
revisited when real fall sequences are available.

### Guardrail: sit_down_heavy

Descent rate can reach ~2.0 hps (exceeds rule 2), but height_ratio stays at 0.67 > 0.55
(person remains at chair height, not floor level) and lying_score stays at 0.05 < 0.4
(seated posture, not lying).  Rules 3 and 4 independently veto detection.

### Guardrail: lie_on_bed

Physically identical to a fall (fast descent, lying posture), but the resting-room veto
(rule 6) suppresses the signal.  This is the primary defense against bed-flop false
positives.

---

## Live-enable runbook

Enable `fall_detection.enabled: true` in a deployment only after the following steps
are complete:

1. **Fixture gate.** Run `make check` and confirm all fall-sequence tests pass.

2. **Log-only phase (minimum 7 days).**  Deploy with:

   ```yaml
   fall_detection:
     enabled: true
   # CC side: route fall_suspected signals to a quiet log-only channel,
   # not caregiver SMS/push notifications.
   ```

   Export metrics daily:

   ```bash
   curl -s http://cts:8000/metrics | grep cts_fall_suspected_total
   curl -s http://cts:8000/metrics | grep cts_fall_descent_rate
   ```

3. **False-positive review.**  Pull signal contexts from the `tracking.signals` stream
   or the CC database.  For each `fall_suspected` signal, review the `context.ph_id`,
   `context.room_name`, and `context.height_ratio`.  Flag any that are clearly
   bed-flops, heavy sits, or pet detections.

4. **Re-calibrate if needed.**  Export the signal context data and pass to the
   calibration harness via `--extra-dir`.  Adjust thresholds, regenerate fixtures,
   and re-run the test gate.

5. **Enable caregiver notifications.**  Route `fall_suspected` signals to the
   caregiver alert channel.  Monitor `cts_fall_suspected_total` by severity label
   for the first 48 hours.

6. **Ongoing defense in depth.**  The `stillness_anomaly` signal (60-minute lying
   threshold) remains active regardless of `fall_detection.enabled`.  A missed fall
   will still be caught as a long-lie event.
