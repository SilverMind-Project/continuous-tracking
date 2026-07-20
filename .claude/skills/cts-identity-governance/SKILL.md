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

## Authority order

Identity resolution uses this order:

1. Operator correction within its explicit observation range.
2. Direct, unambiguously associated ArcFace evidence with compatible calibration and
   calibrated confidence above the configured authority threshold.
3. The existing PH posterior over operator-verified ReID gallery evidence.
4. Bounded temporal prior.

Height, propagated face evidence, raw ArcFace similarity, pending gallery entries, and
track-local appearance prototypes are corroborating or tracking inputs only. They never create
authority.

When material evidence conflicts and the hierarchy does not resolve it, the effective identity is
`Unknown`. A temporary unknown is preferable to a confident swap.

## ArcFace rules

- Persist raw cosine similarity and calibrated confidence as different fields.
- Raw similarity is never presented or consumed as probability.
- Only direct face evidence may become authoritative. Propagated evidence cannot satisfy the
  authority threshold or refresh the authoritative-evidence timestamp.
- Authority additionally requires an unambiguous face-to-person observation association.
- Calibration artifacts are versioned by model and preprocessing profile. Missing, invalid, or
  incompatible calibration fails closed to non-authoritative evidence.
- Existing `recognized`, `candidate`, and `unrecognized` states may continue to use raw
  similarity thresholds for compatibility and weak-evidence weighting. `authoritative` is a
  separate determination.
- ArcFace enrollment is a golden dataset. PH inference, operator identity correction, and ReID
  review never modify enrollment.

## Temporal prior

- The prior window is configurable; the household default is 30 seconds.
- Age is measured from the last independent qualifying evidence, never from the last resolver
  evaluation or persistence write.
- Prior-only maintenance must not refresh that timestamp.
- Height and propagated face evidence must not refresh it.
- Verified ReID may refresh it only after independently clearing PH confidence, margin, quality,
  and conflict gates.
- A conflict immediately yields `Unknown`; sticky maintenance cannot hide it.
- The evidence clock advances only on identity-matched independent evidence: a direct
  recognized ArcFace anchor whose `person_id` equals the committed identity, or a ReID vote
  whose argmax equals the committed identity. Presence of *any* evidence in the frame is never
  sufficient; smoothing mass in a likelihood distribution is not evidence. Gate with an explicit
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
- Verified ReID disagreement may add cost but is not a hard gate.
- If every pair is invalid, spawn or revive an `Unknown` PH; never force a least-bad match.
- Preserve current PH batch-time behavior unless a separately approved replay-backed change
  demonstrates that event-time refactoring is required.

## Track-local appearance versus identity gallery

PH `gallery_mean` and view prototypes are inferred tracking state, not labeled identity data.

- Update them only after association, crop-quality, finite-vector, and consistency gates pass.
- Abruptly inconsistent observations are diagnostic outliers and do not update prototypes.
- Partition/reset prototypes at a confirmed handoff or split.
- A merge preserves source partitions and rebuilds prototypes from accepted observations; never
  average conflicting prototypes blindly.
- Track-local prototypes cannot enter the identity gallery directly.

## ReID gallery lifecycle

Every ReID entry has one explicit state: `pending_review`, `auto_verified`, `operator_verified`, or
`rejected` (identity-continuity M02 added `auto_verified`). Pending and rejected entries never
vote, including through caches, fallback queries, or compatibility code; `auto_verified` and
`operator_verified` vote at their configured trust (`resolver.gallery_auto_verified_trust_multiplier`
default 1.5, `resolver.gallery_verified_trust_multiplier` default 2.0).

The only pipeline write into `reid_gallery` is `ReIDCandidateStage` →
`GalleryRepository.create_review_candidate`, always with crop + frame provenance and
`origin_tracklet_id`/`ph_id`. A face-derived candidate requires the recognized ArcFace identity to
equal the committed identity, with calibrated confidence at or above the authority threshold. The
minting state is `pending_review` unless the calibrated confidence also clears
`reid_candidates.auto_verify_min_confidence` (default 0.90), in which case it mints straight into
`auto_verified`; raw (uncalibrated) confidence never mints `auto_verified` (fail-closed, same
posture as the ArcFace authority gate). Caps count `pending_review` + `auto_verified` +
`operator_verified` rows (`PENDING_AND_VERIFIED` in `app/storage/gallery.py`), never `rejected`.
Review mutations go through `apply_review_action`/`compensate_review` only, never raw SQL, never
`_pool`.

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
implementation (identity-continuity M01). Its `score_hits` -> `cap_votes` -> `aggregate_mean` /
`aggregate_max_over_views` pipeline is the single scorer for every gallery query path: the
per-orientation multiview query, the single-query fallback, and the shadow comparison. A new
gallery query path must call `score_hits`/`cap_votes`/an existing aggregate function; inline
scoring in the resolver (recomputing the logistic curve, trust multiplier, or recency decay by
hand instead of delegating) is a defect.

- Apply a configurable trust multiplier to verified hits before identity aggregation
  (`resolver.gallery_verified_trust_multiplier`, default 2.0) and to auto_verified hits
  (`resolver.gallery_auto_verified_trust_multiplier`, default 1.5). Do not alter cosine
  similarity.
- Apply configurable exponential recency decay (`resolver.gallery_recency_half_life_days`,
  default 2.0 as of identity-continuity M03, shortened from the original 7.0) with no floor.
- Cap or cluster near-duplicate votes by source episode, camera, and orientation.
- Partition embeddings by model/preprocessing version and query only compatible entries.
- The schema for per-hit decision provenance (`IdentityDecisionGalleryHit`) exists end to end
  (domain type, Postgres table, repository read/write) but no gallery query path populates it
  today; `IdentityProvenanceDecision.gallery_hits` is always empty (verified 2026-07-20, M01).
  `ScoredHit` already carries every field `IdentityDecisionGalleryHit` needs; wiring population
  through `resolve()` into `ProvenancePersistStage` is separate, deferred work, not part of the
  shared scorer.

### Hard vote-age cutoff (identity-continuity M03)

`resolver.gallery_vote_max_age_s` (default 43200s/12h; `None`/absent/zero disables it) is a hard
filter enforced one level below the shared scorer, at `GalleryRepository.search_similar` itself:
gallery entries older than the cutoff never reach `score_hits` at all, on any path. SOLIDER-REID
embeddings are clothing-dominated and clothing validity is a step function at wardrobe change, not
a smooth decay; the cutoff encodes the step, and `gallery_recency_half_life_days` handles staleness
inside the window.

`IdentityResolver._gallery_search_kwargs()` is the single helper that builds the cutoff plus the
`VOTING_STATES` filter for every `search_similar` call. A new resolver gallery query path must call
it (`**self._gallery_search_kwargs()`); recomputing `max_age_seconds`/`states` by hand at a new call
site is a defect, exactly like inline scoring above. The tracker's verified-ReID disagreement probe
(`WorldTracker._resolve_verified_reid_identities`) mirrors this with its own
`world_tracker.reid_disagreement_max_age_s`, so a future flip of `enable_reid_disagreement_cost`
inherits correct temporal behavior without further changes.

One read is deliberately exempt: `list_gallery_entries_for_tracklets`, called from the single-query
fallback's corpus-building step to build a PH's own query embedding from its recent gallery entries,
does not take the cutoff. That read describes the PH's own track, not a candidate vote.

The cutoff has a production precondition: it depends on `auto_verified` rows accumulating from
same-day, calibrated face matches (M02). A 12-hour cutoff with no same-day population starves the
resolver's ReID vote every morning (V4) -- M02 must be landed and enabled in production, with
`auto_verified` rows observably accumulating on ordinary days, before relying on the M03 defaults.

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
(range-less) revision supersession of `PersonLocationHistory` and `cts_dementia_signals` rows
(M06). Change both together; a value drift lets CC rewrite more or less history than the resolver
actually promises.

### Unknown-segment backfill (identity-continuity M04)

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
- replay tests with two-person crossings, occlusions, camera transitions, and explicit unknowns.

An authoritative identity swap in a reviewed two-person replay is release-blocking. Increased
`Unknown` rate is acceptable initially and must be measured rather than hidden by threshold
relaxation.

## Review checklist

- [ ] Raw inference is immutable.
- [ ] Effective identity is explicit and revision-aware.
- [ ] No inferred label becomes training truth.
- [ ] No raw similarity is named confidence or probability.
- [ ] Prior-only evaluation cannot refresh independent evidence time.
- [ ] Pending/rejected gallery entries are excluded at repository and cache boundaries.
- [ ] Duplicate active identities cannot escape the resolver transaction.
- [ ] PH-local appearance and identity gallery remain separate.
- [ ] Every new field is wired across domain, migration, repositories, protobuf/API, and tests.
- [ ] Failure paths emit metrics and fail closed.

