# Continuous Tracking Repository Documentation

This directory holds engineering and operational documentation needed inside a standalone checkout of `continuous-tracking`.

## Documentation Delineation Standard (Code Repository vs. Skills vs. Public Portal)

Every documentation artifact across the SilverMind repositories belongs in exactly one of three tiers. Never duplicate prose across tiers.

| Tier | Location | Primary Audience | What Belongs Here | What Does Not Belong Here |
| --- | --- | --- | --- | --- |
| **1. Normative Agent and Contributor Rules** | `CLAUDE.md`, `.claude/skills/`, `.claude/rules/` | AI coding agents and contributors modifying code in this repository | Enforced architectural invariants, layering rules, stage and module maps, covariance/time/identity governance contracts, and test gates (`make docs-check` validates every file reference) | Multi-page tutorials, product overviews, or historical milestone logs |
| **2. In-Repository Engineering and Operations Docs** | `docs/` (this directory) | Engineers and operators working inside a single repository checkout | 1. **`systems-architecture.md`**: Canonical single-file technical architecture reference for this repository.<br>2. **`wire-format.md`**: Redis Streams and protobuf wire contract owned by this repository.<br>3. **Repo-coupled operational runbooks** (`runbook.md`, `jetson-int8-deployment.md`) tied directly to repository scripts, compose files, and model directories.<br>4. **Active technical decision records and evaluation memos** for conditional or upcoming work (`rf-hardware-decision.md`, `agitation-model-path.md`, `identity-path-validation-2026-06.md`, `spike_m5_s5a_*.md`, `spike_m5_s5b_*.md`). | Drafts or duplicate copies of pages published on `silvermind-project.github.io/docs/`, or pre-implementation plans whose milestones have already shipped and whose present-tense "current behavior" tables contradict deployed code |
| **3. Public Documentation Portal** | `../silvermind-project.github.io/docs/` | Caregivers, care partners, operators, and external developers | Family and privacy guides (`docs/families/`), cross-service setup and deployment guides (`docs/guide/`, `docs/hardware/`), progressive concept series (`docs/features/continuous-tracking/01-*` to `07-*`), feature pages (`frame-pipeline.md`, `tracking-concepts.md`, `camera-calibration.md`, `dementia-signals.md`, `cc-integration.md`), identity-integrity architecture pages (`identity-integrity/`), and cross-repo development guides (`docs/development/`) | Repo-internal Claude hooks, throwaway spike notes, or unshipped feature claims presented as deployed |

## Documents in This Directory

| File | Kind | Purpose |
| --- | --- | --- |
| [systems-architecture.md](systems-architecture.md) | Architecture reference | Single-file technical reference for `rtsp-ingress` and `tracking-orchestrator`: the 18-stage frame pipeline, the 10-step `WorldTracker` loop, the Bayesian `IdentityResolver`, camera profiles and roster sync, home-camera nuances, and the CTS-to-CC wire boundary |
| [wire-format.md](wire-format.md) | Wire contract | Redis Streams protobuf envelope specification, active streams, field-map exceptions, schema evolution rules, and codegen workflow |
| [runbook.md](runbook.md) | Operational runbook | Day-2 operational runbook for `rtsp-ingress`, `go2rtc`, `tracking-orchestrator`, Triton, Redis Streams, and TimescaleDB: metrics, common incidents, capacity budgets, and replay verification |
| [jetson-int8-deployment.md](jetson-int8-deployment.md) | Hardware runbook | Step-by-step operator procedure for copying `triton-models-jetson/`, building TensorRT INT8 plans on an 8 GB Jetson Orin Nano Super, and pointing CTS and `person-identification-service` at the remote Triton endpoint |
| [rf-hardware-decision.md](rf-hardware-decision.md) | Decision record | Accepted RF wearable hardware decision record (`R17` to `R30`): Tier 1 coin-cell BLE beacons with ESP32 room receivers, Tier 2 motion-duty-cycled 12 s UWB bursts on Pixel Watch 5 gated on the standalone Wear OS UWB probe, shared USB FiRa anchors, RF node placement and survey rules, and review triggers |
| [agitation-model-path.md](agitation-model-path.md) | Decision memo | Phase-2 upgrade specification from the heuristic `agitation_index` signal to a skeleton-action recognition model (PoseC3D / ST-GCN), data collection, Jetson memory budget, and promotion criteria |
| [identity-path-validation-2026-06.md](identity-path-validation-2026-06.md) | Validation record | Historical validation establishing CTS as the authoritative identity/location path when `cts.enabled=true`, updated with the Wave 4 completion status (`PersonLocationService` SSOT in `cognitive-companion`) |
| [spike_m5_s5a_online_self_calibration.md](spike_m5_s5a_online_self_calibration.md) | Evaluation memo | Deferred spike evaluating opportunistic relative homography estimation from corresponded tracks across uncalibrated cameras |
| [spike_m5_s5b_appearance_topology_reid_survey.md](spike_m5_s5b_appearance_topology_reid_survey.md) | Evaluation memo | Deferred literature scan template for camera-topology-aware ReID within the approved dependency stack |

## Consolidated and Retired Drafts (Canonical Locations)

Earlier pre-implementation plans and draft pages that previously lived in `docs/` were promoted to `.claude/skills/` (normative rules) and `silvermind-project.github.io/docs/` (published tutorials and architecture pages) and removed from this directory to prevent drift:

| Former File in `continuous-tracking/docs/` | Canonical Normative Skill | Canonical Public Portal Page (`silvermind-project.github.io/docs/`) |
| --- | --- | --- |
| `camera-floor-basics.md` | `.claude/skills/cts-spatial-fusion/SKILL.md` | `features/continuous-tracking/01-camera-to-floor-basics.md` |
| `floor-fusion-uncertainty-plan.md` | `.claude/skills/cts-spatial-fusion/SKILL.md` | `features/continuous-tracking/02-why-one-dot-jitters.md` and `04-multi-camera-fusion.md` |
| `measurement-uncertainty-anisotropy.md` | `.claude/skills/cts-spatial-fusion/SKILL.md` | `features/continuous-tracking/03-measurement-uncertainty.md` |
| `floor-region-visibility-polygon.md` | `.claude/skills/cts-spatial-fusion/SKILL.md` | `features/continuous-tracking/06-floor-region-and-visibility.md` |
| `07-drift-detection.md` | `.claude/skills/cts-spatial-fusion/SKILL.md` | `features/continuous-tracking/07-operations-drift-and-recalibration.md` |
| `architecture/identity-authority-and-unknown.md` | `.claude/skills/cts-identity-governance/SKILL.md` | `features/continuous-tracking/identity-integrity/identity-authority.md` |
| `architecture/reid-gallery-governance.md` | `.claude/skills/cts-identity-governance/SKILL.md` | `features/continuous-tracking/identity-integrity/reid-gallery-governance.md` |
| `architecture/identity-revision-projections.md` | `.claude/skills/cts-identity-governance/SKILL.md` | `features/continuous-tracking/identity-integrity/revision-projections.md` |
| `architecture/identity-integrity-program-baseline.md` | `.claude/skills/cts-identity-governance/SKILL.md` | `features/continuous-tracking/identity-integrity.md` and `identity-integrity/contracts.md` |
