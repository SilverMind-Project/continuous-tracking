---
name: cts-identity-governance
description: Use when changing Person Hypothesis identity evidence, identity resolution, ReID galleries, identity revisions, operator corrections, or identity provenance in continuous-tracking.
---

# CTS Identity Governance

Load this skill together with:

- `/home/sriram/code/nanai/continuous-tracking/.claude/skills/engineering-standards/SKILL.md`
- `/home/sriram/code/nanai/continuous-tracking/.claude/skills/cts-pipeline/SKILL.md`
- `/home/sriram/code/nanai/continuous-tracking/.claude/skills/cts-spatial-fusion/SKILL.md` when world association or geometry changes

This skill defines the authority, provenance, and learning boundaries for identity in the
Person Hypothesis system. It is normative. Update it in the same change that changes any
contract described here.

## Core model

A Person Hypothesis (PH) is the persistent world-coordinate track. Identity is a governed,
revision-aware property of that PH; it is not a second tracker.

Keep these concepts separate even when one service coordinates them:

1. **Observation association** decides which physical observation updates which PH.
2. **Inferred identity evidence** records what models and priors reported.
3. **Effective identity** applies authority and revisions to produce the label consumers use.
4. **Operator truth** is an immutable correction over an explicit observation-bounded range.
5. **Learning data** is governed independently from either inferred or effective labels.

Never collapse these into a single mutable `person_id` field.

A PH is born Unknown; only the resolver writes identity. The spawn branch of `WorldTracker.step`
sets `current_identity_id=None` and the resolver, which runs in the same step with that round's
faces, decides as for any Unknown PH. Outside operator corrections, `current_identity_id` changes
only through `evidence_backed_commit`, `prior_only_update`, and `clear_to_unknown`.

## Authority order

Identity resolution uses this order:

1. Operator correction within its explicit observation range.
2. Direct, unambiguously associated ArcFace evidence with compatible calibration and
   calibrated confidence above the configured authority threshold.
3. The existing PH posterior over operator-verified ReID gallery evidence.
4. Bounded temporal prior.

Height (`HeightChannel`, `resolver.height_mode`: `off`, `shadow`, `enabled`; profiles in
`identity_height_profiles` with `learned` taking precedence over `operator`), propagated face
evidence, raw ArcFace similarity, pending gallery entries, and track-local appearance prototypes
are corroborating or tracking inputs only. They never create authority or refresh the evidence
clock.

"Compatible calibration" is checked in one place,
`FaceIdentityStage._resolve_calibrated_confidence` (`app/pipeline/stages/face_identity.py`). It
keeps a face's `calibrated_confidence` only when all of these hold, and otherwise stores `None`:

- `calibration_status == "ready"`;
- `calibration_profile == "single_frame"` (`SINGLE_FRAME_PROFILE`). An artifact fitted for a
  multi-frame profile is never authoritative on a single frame;
- `face.quality.version == face_id.expected_quality_version` (settings, default `heuristic-v1`;
  non-default recognizer profiles qualify this as `heuristic-v1@<recognizer_version>` so a
  recognizer change fails closed until recalibrated). A face without a quality block fails;
- `face.recognizer_version == face_id.expected_recognizer_version` and `face.preprocessing_version`
  matches `face_id.expected_preprocessing_version` when configured. (`FaceIdentificationClient`
  decodes `recognizer_version` with a one-release fallback to the legacy `arcface_model_version`
  wire alias, and `_build_face_id_config` keeps a one-release fallback to the legacy
  `expected_arcface_model_version` settings key.)

person-id additionally returns `None` for any identity outside the artifact's
`covered_identities`, so a resident enrolled after the fit has no calibrated confidence until the
owner recalibrates. Changing the quality scorer means refitting calibration and bumping
`face_id.expected_quality_version` together. The resolved value is also written to
`face_observation_stats.calibrated_confidence` (migration 0006) for the face resolution report;
the stats row never feeds resolution. `FaceIdClient.identify_crops` sends a `CaptureContext`
(camera, capture time, crop padding, JPEG quality) so person-id can sample faces for household
calibration; the context carries no identity claim.

When material evidence conflicts and the hierarchy does not resolve it, the effective identity is
`Unknown`. A temporary unknown is preferable to a confident swap.

## ArcFace rules

Face scores have four names, and each name means one thing:

| Name | Meaning | Where |
| --- | --- | --- |
| match score | The score the resolver weighs for a face: raw cosine similarity for native recognized and candidate anchors, the face detection score for an unrecognized marker, the calibrated confidence after the spatial-match haircut for a `cc_assertion` anchor | `FaceAnchor.match_score`, `FaceEvidence.match_score` |
| raw similarity | ArcFace cosine similarity to the nearest enrolled centroid, in [-1, 1]; not a probability | `FaceResult.raw_similarity`, `FaceAnchor.similarity`, `face_observation_stats.raw_similarity` |
| detection score | SCRFD face detector score, in [0, 1]; says a face exists, not whose it is | `FaceResult.det_score`, `FaceAnchor.det_score` |
| calibrated confidence | Calibrated probability of a match; null unless calibration is ready | `calibrated_confidence` everywhere |

- A threshold compared against raw similarity is named `*_similarity` (for example
  `face_id.min_raw_similarity`, `resolver.face_commit_min_similarity`,
  `resolver.contradiction_face_similarity`, `world_tracker.face_conflict_min_similarity`). Only a
  calibrated probability is called a confidence. `tests/contracts/test_face_vocabulary_names.py`
  keeps the old names out of `app/` and `config/settings.yaml`, including cognitive-companion's
  old camera field `face_id_min_confidence` (CC sends `face_id_min_raw_similarity`).
- Persist raw cosine similarity and calibrated confidence as different fields.
- Raw similarity is never presented or consumed as probability.
- Face image quality comes only from person-identification-service (`FaceResult.quality`,
  carried as `FaceAnchor.face_quality` with its scorer version). It is a separate field from
  `FaceAnchor.quality`, the evidence-path weight (1.0 native, attenuated on propagation, the
  assertion's own quality for `cc_assertion`); never overwrite one with the other.
  `IdentityResolver._effective_face_quality` is the only place the two combine, under
  `resolver.face_quality_mode` (`off`, `shadow`, `enabled`). Outside `off`, a native anchor with
  no score uses `resolver.face_quality_missing_default` (0.5), never 1.0; external anchors are not
  discounted again; persisted provenance (`FaceEvidence.quality`, `IdentityEvidence.quality`)
  never depends on the mode. Face quality never grants or removes ArcFace authority.
- Only direct face evidence may become authoritative. Propagated evidence cannot satisfy the
  authority threshold or refresh the authoritative-evidence timestamp.
- Authority additionally requires an unambiguous face-to-person observation association.
- Calibration artifacts are versioned by model and preprocessing profile. Missing, invalid, or
  incompatible calibration fails closed to non-authoritative evidence.
- Existing `recognized`, `candidate`, and `unrecognized` states may continue to use raw
  similarity thresholds for compatibility and weak-evidence weighting. `authoritative` is a
  separate determination.
- `face_id.evidence_policy` (`legacy`, `shadow`, `calibrated`) decides which recognized faces are
  emitted and how they weigh; under `calibrated` a face weighs by its calibrated confidence, a face
  without one is capped and only corroborates, and `shadow_only` anchors never reach a live
  decision. Read `reference/calibrated-face-evidence.md` before changing face emission or weighting.
- A face template (several faces of one PH, `aggregation="template"`) is always weighed and gated
  as calibrated evidence, grants authority only when its member faces agree, never enters
  association, contradiction, the duplicate bypass, or propagation, and its embeddings stay in
  memory. Read `reference/face-templates.md` before changing templates.
- Wearable localization evidence (`WEARABLE` grade, `wearable.mode`: `off`, `shadow`, `enabled`)
  refreshes the independent-evidence clock using `newest_fix_at`, ranks in the duplicate guard
  without bypassing incumbents, never grants authority, and forces `Unknown` (`wearable_face`)
  when a qualifying calibrated direct-face anchor disagrees. Read `reference/wearable-evidence.md`
  before changing wearable association, attestation, or channel weighting.
- An attested wearable association is a sensor-attested label: it may seed body-gallery rows by
  grade (trajectory `auto_verified` at 1.25 trust, room `pending_review`), offer face exemplars to
  person-id's operator-gated queue, backfill NULL history (trajectory only, flag off), and serve
  as image-only evaluation truth; never enrollment, authority, or unaudited calibration. Read
  `reference/sensor-attested-labels.md` before changing any wearable learner.
- Closed-world prior shaping (`CLOSED_WORLD` grade, `ClosedWorldChannel`, `closed_world.mode`:
  `off`, `shadow`, `enabled`) shapes priors only (`counts_as_commit_evidence = False`), fails
  closed when `roster_complete` is false, enforces an explicit `UNKNOWN` guest floor, and requires
  simultaneous observation plus spatial separation for joint elimination. Read
  `reference/closed-world-evidence.md` before changing closed-world reasoning.
- Household appearance classifier (`HOUSEHOLD_CLASSIFIER` grade, `HouseholdClassifierChannel`,
  `household_classifier.mode`: `off`, `shadow`, `enabled`) is a non-authoritative corroborating
  channel loaded from a versioned JSON artifact (no pickle or scikit-learn in `app/`), counts as
  commit evidence only at calibrated probability >= 0.8, never refreshes the evidence clock, and
  trains only from audited trajectory-attested wearable, direct-face authority, and
  operator-approved samples. Read `reference/household-appearance-classifier.md` before changing
  the classifier or sample capture.
- ArcFace enrollment is a golden dataset. PH inference, operator identity correction, and ReID
  review never modify enrollment. Enrollment gains entries
  only via explicit enroll calls, explicit operator naming of a visitor cluster
  (`person-identification-service`'s `VisitorStore.name_cluster`), or explicit operator promotion
  of a labeled capture; never automatically. Visitor
  clusters themselves (unmatched face embeddings persisted for cross-visit review) are a
  separate, non-authoritative dataset that never enters enrollment, identification, or tracking
  until a caregiver explicitly names one.

## Temporal prior

- The prior window is configurable; the household default is 30 seconds.
- Age is measured from the capture time of the newest contributing evidence
  (`IdentityDecision.evidence_at`: this round for a frame face or ReID vote, the newest member face
  for a face template); never from the last resolver
  evaluation or persistence write.
- Prior-only maintenance must not refresh that timestamp.
- Height and propagated face evidence must not refresh it.
- Verified ReID may refresh it only after independently clearing PH confidence, margin, quality,
  and conflict gates.
- A conflict immediately yields `Unknown`; sticky maintenance cannot hide it.
- The evidence clock advances only on identity-matched independent evidence: a direct
  recognized ArcFace anchor whose `person_id` equals the committed identity, a qualifying wearable
  evidence channel, or a verified ReID gallery vote whose score for the committed identity exceeds
  0.3 (not only the single alphabetical argmax, so a corroborating channel like height can resolve
  an identical-clothing tie). Presence of *any* evidence in the frame is never sufficient;
  smoothing mass in a likelihood distribution is not evidence. Gate with an explicit
  `evidence_identity_ids` set, never with `id in distribution`.

Persist `last_independent_identity_evidence_at` separately from identity update timestamps.
Repository APIs must make it impossible for an ordinary held decision to overwrite this field.

## Duplicate active identity invariant

At most one active global PH may hold a household identity. Multiple camera observations may be
deduplicated into that PH.

When contenders claim one identity:

- Keep a winner only when its independent evidence is clearly stronger.
- Set all other contenders to `Unknown`.
- If effectively tied, set every contender to `Unknown`.
- Emit metrics and persist conflict provenance.

This guard is authoritative, not shadow-only. Shadow evaluation may remain for threshold
comparison but cannot permit invariant violations.

## World association boundary

Physical plausibility belongs in world association, not identity resolution.

- Reject non-finite floor coordinates, covariance, Mahalanobis distance, and costs.
- Validate covariance shape, symmetry, positive semidefiniteness, and configured caps before
  Hungarian assignment.
- Operator identity and qualifying direct ArcFace conflicts are hard association gates.
  Association and revival compare an observation's recognized face with the PH's held identity;
  cross-camera dedup compares two observations' faces (`dedup._faces_conflict`: both native
  single-frame recognitions naming different people). All three use the same floor,
  `world_tracker.face_conflict_min_similarity`, so a weak recognition never splits a person.
- Verified ReID disagreement may add cost but is not a hard gate.
- If every pair is invalid, spawn or revive an `Unknown` PH; never force a least-bad match.
- Preserve current PH batch-time behavior unless a separately approved replay-backed change
  demonstrates that event-time refactoring is required.
- An observation carries a face anchor only when that anchor was produced for the same detection
  (`WorldTrackingStage._attach_face_anchors` matches on `detection_id` only; there is no
  camera-wide fallback, and an anchor without a `detection_id` attaches to nothing). At most one
  face per person crop becomes evidence, chosen by the head-region rule
  (`app/pipeline/face_attribution.py`: the face centre must lie in the top
  `face_id.head_region_height_frac` of the person box, widened by
  `face_id.head_region_side_margin_frac`, including the upward crop padding); two faces in the
  region are ambiguous and yield no anchor. Attribution is decided inside `_gate_outcome`
  (outcomes `dropped_not_in_head_region`, `dropped_ambiguous`), never in a second gate.

## Track-local appearance versus identity gallery

PH `gallery_mean` and view prototypes are inferred tracking state, not labeled identity data.

- Update them only after association, crop-quality, finite-vector, and consistency gates pass.
- Abruptly inconsistent observations are diagnostic outliers and do not update prototypes.
- Partition/reset prototypes at a confirmed handoff or split.
- A merge preserves source partitions and rebuilds prototypes from accepted observations; never
  average conflicting prototypes blindly.
- Track-local prototypes cannot enter the identity gallery directly.

## ReID gallery lifecycle

Every ReID entry has one explicit state: `pending_review`, `auto_verified`, `operator_verified`,
`rejected`, or `expired` (migration `0011`). Pending, rejected, and expired entries never
vote, including through caches, fallback queries, or compatibility code; `auto_verified` and
`operator_verified` vote at their configured trust (`resolver.gallery_auto_verified_trust_multiplier`
default 1.5, `resolver.gallery_verified_trust_multiplier` default 2.0).

The only pipeline write into `reid_gallery` is `ReIDCandidateStage` ->
`GalleryRepository.create_review_candidate`, always with crop + frame provenance,
`origin_tracklet_id`/`ph_id`, and a `seed_source` (`face`, `wearable_attested`,
`wearable_room_attested`, `operator`; migration `0009`). A face-derived candidate requires the recognized ArcFace identity to
equal the committed identity, with calibrated confidence at or above the authority threshold. The
minting state is `pending_review` unless the calibrated confidence also clears
`reid_candidates.auto_verify_min_confidence` (default 0.90), in which case it mints straight into
`auto_verified`; raw (uncalibrated) confidence never mints `auto_verified` (fail-closed, same
posture as the ArcFace authority gate). The per-(identity, orientation) working-set cap
(`reid_candidates.max_entries_per_orientation`, default 10) counts `pending_review` +
`auto_verified` + `operator_verified` rows (`PENDING_AND_VERIFIED` in `app/storage/gallery.py`)
within the active vote window (`seen_since = capture_time - gallery_vote_max_age_s`), never
`rejected` or `expired` rows and never rows already outside the vote window.
Review mutations go through `apply_review_action`/`compensate_review` only; retention expiry goes
through `GalleryRetentionService` (`app/services/gallery_retention.py`, default `enabled` mode;
14d `auto_verified`, 30d `pending_review`, 90d `operator_verified`, deleting vector and crop on
expiry and keeping audit metadata), never raw SQL or `_pool`.

`app/storage/gallery.py` exports two vocabulary constants: `VERIFIED_ONLY` (`operator_verified`
only, the default for the one admin/list surface with no vote-path caller) and `VOTING_STATES`
(`operator_verified` + `auto_verified`, the default for every read whose result feeds identity
resolution: `search_similar`, `list_gallery_entries_for_tracklets`, `gallery_similarity`, and the
`GalleryCache` fallbacks that wrap them). A new gallery read chooses one of these two explicitly;
never invent a third default.

Candidate creation requires:

- finite L2-normalized embedding;
- compatible model and preprocessing versions;
- minimum dimensions and shared PH crop quality;
- acceptable truncation/occlusion;
- valid orientation metadata;
- source frame key, immutable crop key, bbox, camera, capture time, image hash, PH/observation IDs,
  and creation evidence.

A high-confidence face-derived candidate additionally requires that direct ArcFace identity equal
the resolved identity. Never seed under a held PH label when the face names another person.

Review actions are `Approve`, `Reject`, `Relabel`, and `Demote`:

- Approval and relabel act on a row in `pending_review` or `auto_verified` and promote it to
  `operator_verified` (relabel additionally records the old proposal and corrected identity).
- Demote acts only on `auto_verified` and returns it to `pending_review`, keeping the vector
  (unlike reject); it is an operator un-trusting a machine-minted row, not a rejection.
- Rejection (from `pending_review` or `auto_verified`) retains immutable audit metadata and an
  embedding fingerprint but deletes the vector and dedicated crop object.
- Undo (`compensate_review`) restores the state the most recent review event's `previous_state`
  recorded, not a hardcoded `pending_review`: an undo of approve-from-`auto_verified` restores
  `auto_verified`, while an undo of approve-from-`pending_review` restores `pending_review`. No
  audit event is mutated or deleted.

The invariant sentence: pending and rejected entries never vote, including through caches,
fallback queries, or compatibility code; `auto_verified` and `operator_verified` vote at their
configured trust.

Identity correction and ReID verification are separate actions. Correcting a bbox never silently
promotes its embedding. Failed quality gates cannot be overridden.

## ReID scoring

Preserve the existing PH k-NN posterior and commit thresholds unless a replay-backed milestone
explicitly changes them.

`app/tracking/identity/gallery_scoring.py` is the only permitted gallery vote scoring
implementation. Its `score_hits` -> `cap_votes` -> `aggregate_mean` /
`aggregate_max_over_views` pipeline is the single scorer for every gallery query path: the
per-orientation multiview query, the single-query fallback, and the shadow comparison. A new
gallery query path must call `score_hits`/`cap_votes`/an existing aggregate function; inline
scoring in the resolver (recomputing the logistic curve, trust multiplier, or recency decay by
hand instead of delegating) is a defect.

- Apply a configurable trust multiplier to verified hits before identity aggregation
  (`resolver.gallery_verified_trust_multiplier`, default 2.0) and to auto_verified hits
  (`resolver.gallery_auto_verified_trust_multiplier`, default 1.5; an auto_verified hit seeded
  by a wearable, `seed_source = wearable_attested`, uses
  `resolver.gallery_wearable_trust_multiplier`, default 1.25). Do not alter cosine similarity.
- Apply configurable exponential recency decay (`resolver.gallery_recency_half_life_days`,
  default 2.0) with no floor.
- Cap or cluster near-duplicate votes by source episode, camera, and orientation.
- Partition embeddings by model/preprocessing version and query only compatible entries.
- The schema for per-hit decision provenance (`IdentityDecisionGalleryHit`) exists end to end
  (domain type, Postgres table, repository read/write) but no gallery query path populates it
  today; `IdentityProvenanceDecision.gallery_hits` is always empty.
  `ScoredHit` already carries every field `IdentityDecisionGalleryHit` needs; wiring population
  through `resolve()` into `ProvenancePersistStage` is separate, deferred work, not part of the
  shared scorer.

### Hard vote-age cutoff

`resolver.gallery_vote_max_age_s` (default 43200s/12h; `None`/absent/zero disables it) is a hard
window evaluated on capture time, never wall-clock time (`datetime.now()` is forbidden in
`app/tracking/identity_resolver.py` and `app/storage/gallery.py` search paths). A `VoteWindow(as_of,
max_age_s)` (`app/tracking/identity/channels.py`) translates the frame's capture time into
half-open bounds `(seen_after, seen_before]` enforced at `GalleryRepository.search_similar` itself:
entries with `seen_at <= seen_after` or `seen_at > seen_before` never reach `score_hits` on any
path, so replays and live runs see identical candidate sets. `search_similar` also enforces
`active_only=True` identically in Postgres and in-memory peers so inactive identities never vote.
SOLIDER-REID embeddings are clothing-dominated and clothing validity is a step function at wardrobe
change, not a smooth decay; the cutoff encodes the step, and `gallery_recency_half_life_days`
handles staleness inside the window (also measured from capture time).

`IdentityResolver._gallery_search_kwargs(as_of=...)` is the single helper that builds the
capture-time window plus `VOTING_STATES` and `active_only` for every `search_similar` call. A new
resolver gallery query path must call it; recomputing bounds or states by hand at a new call site
is a defect, exactly like inline scoring above. The tracker's verified-ReID disagreement probe
(`WorldTracker._resolve_verified_reid_identities(observations, as_of)`) mirrors this with
`world_tracker.reid_disagreement_max_age_s` on the same capture time.

One read is deliberately exempt: `list_gallery_entries_for_tracklets`, called from the single-query
fallback's corpus-building step to build a PH's own query embedding from its recent gallery entries,
does not take the cutoff. That read describes the PH's own track, not a candidate vote.

The cutoff has a production precondition: it depends on `auto_verified` rows accumulating from
same-day, calibrated face matches. A 12-hour cutoff with no same-day population starves the
resolver's ReID vote every morning: same-day `auto_verified` promotion must be enabled in
production, with rows observably accumulating on ordinary days, before relying on these defaults.

## External identity evidence

Every `FaceAnchor`'s governance (authority, clock refresh, duplicate ranking, association,
contradiction, propagation, calibration requirement, attachment precedence, and likelihood scale)
is declared once per `(AnchorOrigin, AnchorAggregation)` in
`app/tracking/identity/evidence_grades.py` and combined through `EvidenceChannel` and
`combine_channels` (`app/tracking/identity/channels.py`, `app/tracking/identity/posterior.py`).
Inline origin or aggregation checks outside `app/tracking/identity/evidence_grades.py` are defects enforced by
`tests/architecture/test_evidence_grade_centralized.py`. Read
`reference/evidence-grades-and-channels.md` before adding a grade or channel.

For `cc_assertion` anchors matched from `cc.identity_assertions` (`ExternalIdentityAssertion` via
constructor-injected `ExternalIdentityEvidenceSource`):

- `app/tracking/world/assertion_matching.py::match_assertions_to_face_anchors` is the sole
  matcher: a floor point (camera-id gated, same-fleet ids only) beats room-name agreement
  (case-normalized, confidence-scaled by `resolver.room_match_confidence_scale`) when no floor
  point exists; an assertion with neither matches nothing, never everything. Proto3 presence
  flags (`has_floor_point`, `has_yaw`, `has_quality`) are authoritative; a value without its
  flag is absent, never a real (0, 0) position.
- Rollout is shadow-first via `resolver.cc_assertion_mode` (`off | shadow | enabled`, default
  `shadow`): shadow matches and records `cc_assertions_shadow_total{outcome=}` without
  injecting anchors into the resolver; `cc_assertions_matched_total{gate=}` and
  `cc_assertions_rejected_total{reason=}` are always-on regardless of mode once the flag is not
  `off`. Flip to `enabled` only after a week of live shadow data.

## Evidence and revision persistence

Persist a compact evidence snapshot for every PH identity decision. Use typed columns and foreign
keys for source, authority, identities, decision/revision lineage, and gallery contributors.
Versioned JSON is reserved for extensible diagnostics.

Required API vocabulary is `inferred_identity_id`, `effective_identity_id`, `authority`,
`decision_source`, and `revision_id`. `person_id` may exist only as a documented compatibility
alias during migration.

`IdentityDecision.authority` is the authority-ladder rung, always a member of
`IdentityAuthority` (`app/tracking/identity/types.py`); the producer emits
`operator | direct_face | posterior | temporal_prior | none`, with `reid_gallery` reserved
for a future governed-gallery rung and `unknown`/`height_proxy` legacy-read-only. It never
contains an identity id.
`decision_source` states which evidence led; the two are not interchangeable. Repositories
reject decisions with out-of-vocabulary authority.

Operator corrections:

- use observation boundaries, not arbitrary timestamps;
- never cross split, merge, or prior operator-revision boundaries automatically;
- preserve original inference;
- may apply frame-only or to a caregiver-confirmed bounded segment;
- affect current PH identity only when the range reaches the live edge;
- are authoritative only inside the explicit range;
- cannot be superseded by inferred revisions;
- propagate through idempotent effective-label projections.

The automatic revision horizon remains bounded. Explicit operator corrections may be historical
and complete asynchronously. CTS is the source of truth; a correction job completes only after
required projections acknowledge the same revision ID.

`resolver.revision_horizon_s` (`identity_resolver.py`, default 600.0) has a mirror on the
cognitive-companion side: `cts.revision_horizon_s` on `IdentityRewriter`, which bounds automatic
(range-less) revision supersession of `PersonLocationHistory` and `cts_dementia_signals` rows. Change both together; a value drift lets CC rewrite more or less history than the resolver
actually promises.

### Unknown-segment backfill

`UnknownBackfillService` (`app/services/unknown_backfill.py`) is the only automatic process that
may call `IdentityCorrectionService.record_inferred_range`; no other code path is permitted to
create an effective range without an operator correction.

Trigger predicate, evaluated per final decision each frame: `previous_identity_id is None` (a
first commit, not a change or demotion), `identity_id` is a real identity, and `authority ==
DIRECT_FACE` (the calibrated ArcFace-authority commit path only; posterior commits never
backfill, since the backfilled label inherits the full trust of the segment).

**The NULL-only invariant is a defect rule.** Backfill only ever fills identity-NULL rows. It must
never change a non-NULL identity value in any table, even to a value that matches the backfill's
own target identity. `IdentityRewriter.backfill_null_rows` enforces this with `AND identity_id IS
NULL` in its SQL, the mirror image of `rewrite()`'s `old_identity_id IS NOT NULL` guard. A rewriter
change that lets either method touch the other's row set is release-blocking.

Clip order when computing the backfill range (start = PH `born_at`, end = commit time): (1) clip
forward past the end of the latest earlier decision on the PH naming a different identity; (2)
clip forward past the end of any overlapping operator range, or skip entirely if an operator range
covers the live edge (operator authority always outranks the automatic process); (3) cap the
maximum span (`resolver.backfill_max_range_s`). A range that collapses to empty is skipped, not
clamped to a minimum.

A second trigger exists for wearables: a segment transition to `trajectory` attestation
backfills identity-NULL rows over `[segment.started_at, capture time]`, once per segment, with
the same clips and NULL-only rewriter, behind `resolver.enable_wearable_backfill` (default false,
waiting on the owner's confirmation; room grade never backfills). See
`reference/sensor-attested-labels.md`.

Staged rollout: `resolver.enable_unknown_backfill` (default false), `resolver.backfill_shadow`
(default true). Shadow mode computes the range, increments metrics, and logs, with zero database
or stream writes. Enabled mode records the inferred range, relabels rows, and publishes one
`IdentityRevision` with `revision_kind=inferred_backfill` and `required_projections=["cc"]`.

## Required tests

Every identity change must include focused tests for applicable categories:

- resolver unit tests for authority order, conflict-to-unknown, prior expiry, and timestamp refresh;
- duplicate-active batch tests including ties and incumbent PHs absent from the frame;
- world-association tests for non-finite geometry and identity hard gates;
- repository parity tests for in-memory and Postgres implementations;
- gallery state/filter/cache tests proving pending/rejected vectors cannot vote;
- seed-invariant tests proving face identity equals candidate identity;
- revision idempotency, stale-version, boundary, overlap, and compensating-revision tests;
- protobuf/API compatibility tests for every producer and consumer;
- replay tests with two-person crossings, occlusions, camera transitions, and explicit unknowns;
- scoring-unification guards proving no gallery path can bypass the central scorer;
- vote-window parity tests for cutoff and decay behavior;
- backfill invariant tests proving the NULL-only update rule;
- visitor-gate tests for offline clustering, naming, and tracking promotion.

An authoritative identity swap in a reviewed two-person replay is release-blocking. Increased
`Unknown` rate is acceptable initially and must be measured rather than hidden by threshold
relaxation.

## Review checklist

- [ ] Raw inference is immutable.
- [ ] Effective identity is explicit and revision-aware.
- [ ] No inferred label becomes training truth; sensor-attested labels only where
      `reference/sensor-attested-labels.md` allows.
- [ ] No raw similarity is named confidence or probability.
- [ ] Prior-only evaluation cannot refresh independent evidence time.
- [ ] Pending/rejected gallery entries are excluded at repository and cache boundaries.
- [ ] Duplicate active identities cannot escape the resolver transaction.
- [ ] PH-local appearance and identity gallery remain separate.
- [ ] Every new field is wired across domain, migration, repositories, protobuf/API, and tests.
- [ ] Failure paths emit metrics and fail closed.

