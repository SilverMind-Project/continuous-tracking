# Association Parameter Sweep — June 2026

**Generated:** 2026-06-12  
**Harness:** `scripts/sweep_association.py`  
**Grid:** `scripts/grids/default.yaml`  
**Seed:** 42 (deterministic; two identical runs produce byte-identical CSV)

## How to rerun

```bash
cd tracking-orchestrator
uv run python scripts/sweep_association.py --grid scripts/grids/default.yaml --out /tmp/sweep
# Results: /tmp/sweep/sweep_results.csv and /tmp/sweep/sweep_summary.md
```

## Setup

- **Fixtures:** 10 synthetic WorldObservation replay scenarios committed under
  `tests/fixtures/frame_replays/` (see `scripts/synthesize_replay_fixture.py`)
- **Ground truth:** per-fixture `*.truth.json` sidecars (detection_id → true person label)
- **Params swept:** `gate_chi2`, `alpha_geo`, `alpha_app`, `observation_noise_m`,
  `ph_close_grace_s` — 144 admissible combinations (after filtering `alpha_geo + alpha_app > 1`)
- **Total runs:** 1 440 (144 combos × 10 fixtures)
- **Elapsed:** 6.3 s on the dev box (DGX H100)

## Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| `identity_preservation` | IDF1-like: 2TP/(2TP+FP+FN) averaged over true persons, where TP = observations of person P in its best-covering PH | ≥ 0.90 |
| `phantom_rate` | Fraction of PH-observation-frames with no ground-truth match | 0.000 |
| `fragmentation` | Mean distinct PH lineages per true person (1.0 = perfect continuity) | ≤ 1.5 |
| `identity_contamination` | Frames where a PH's committed identity differs from the true person | **= 0 (hard gate)** |

## Results

### Contamination guardrail

**All 144 combinations scored zero identity contamination on all 10 fixtures.**
The shipped constants pass the clinical guardrail unconditionally.
No combination is inadmissible.

### Key sensitivity

The dominant differentiator is **`ph_close_grace_s`**:

| `ph_close_grace_s` | Avg IDF1 | Avg fragmentation | Δ vs shipped |
|--------------------|----------|-------------------|--------------|
| 3.0 | 0.877 | 1.500 | worse |
| **5.0 (shipped)** | **0.891** | **1.400** | baseline |
| 8.0 | 0.943 | 1.200 | **+5.8 pp IDF1, −0.20 frag** |

`gate_chi2` (5.99 / 9.21 / 13.82) and the alpha weights show no measurable
difference on these synthetic fixtures.  The cost matrix patterns are
geometrically unambiguous — all three gates produce identical associations.
Differentiating gate sensitivity requires live-camera diversity not available
in the current fixture suite.

### Per-fixture breakdown at shipped defaults (gate_chi2=9.21, alpha_geo=0.5, alpha_app=0.4, obs_noise=0.25m, grace=5.0s)

| Fixture | IDF1 | Frag | Notes |
|---------|------|------|-------|
| two_cameras_one_room | 1.000 | 1.00 | Calibrated dedup works perfectly |
| two_rooms_two_people | 1.000 | 1.00 | Two well-separated people, perfect |
| hallway_bathroom_door | 1.000 | 1.00 | 8-frame gap < grace_s |
| resident_plus_stranger | 1.000 | 1.00 | Guardrail holds; stranger stays separate |
| two_people_one_room | 1.000 | 1.00 | Crossing paths handled correctly |
| uncalibrated_two_people_home | 1.000 | 1.00 | Two PHs stay distinct on uncalibrated cam |
| uncalibrated_pacing | 0.794 | 2.00 | 12-frame (6s) gap > grace_s → forced close |
| single_camera_turn | 0.690 | 2.00 | 12-frame occlusion gap → forced close |
| mixed_calibration_entry | 0.750 | 2.00 | Calibrated→uncalibrated handoff, no revival |
| cross_camera_handoff | 0.681 | 2.00 | 32-frame (16s) gap > any grace_s tested |

The three fixtures scoring < 1.0 all involve occlusion gaps where PH closes and a new one spawns.

### At ph_close_grace_s=8.0 (same other values)

| Fixture | IDF1 | Frag | Change |
|---------|------|------|--------|
| single_camera_turn | 1.000 | 1.00 | +31 pp (6s gap now within grace) |
| uncalibrated_pacing | 1.000 | 1.00 | +21 pp (6s gap within grace) |
| mixed_calibration_entry | 0.750 | 2.00 | unchanged (handoff, not gap) |
| cross_camera_handoff | 0.681 | 2.00 | unchanged (16s gap > 8s) |

## Conclusion: keep current constants

The shipped constants (`gate_chi2=9.21`, `alpha_geo=0.5`, `alpha_app=0.4`,
`observation_noise_m=0.25`, `ph_close_grace_s=5.0`) are **safe** — zero
contamination across all fixtures.

**One change is recommended as a follow-up:**

> `ph_close_grace_s`: raise from 5.0 s to 8.0 s

This eliminates PH fragmentation on the occlusion scenarios that are clinically
representative (a senior momentarily out of frame, or passing through a blind
spot).  The 8.0 s value adds 3 s of extra coasting on missed frames, which is
within acceptable limits given the pacing hazard detection signal already
requires a minimum 60 s window.

The cross-camera handoff fixture (16 s gap) remains fragmented at all tested
grace values; fixing that path requires cross-camera revival (M5.1, already
implemented but flagged `enable_cross_camera_revival=False`).

**`gate_chi2` and alpha weights:** no recommendation for change.  These parameters
are insensitive in the current synthetic fixture space.  Tuning them would
require live-camera replay with floor-point noise representative of the DGX
deployment.

## Follow-up PR (out of scope for this task)

File a settings-only PR targeting `world_tracker.ph_close_grace_s = 8.0` in
`settings.yaml`, with this report linked.  Do not enable without a live-system
shadow test to confirm no adverse behaviour on corner cases outside the fixture
suite (e.g. multi-person rooms with longer stationary periods).
