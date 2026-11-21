# Sensor-Attested Labels (Wearable Supervision)

An attested wearable association (`WearableEvidence.attested`, grade `room` or `trajectory`; see
`wearable-evidence.md`) is a **sensor-attested label**: direct sensor evidence that a household
member was on a PH for an interval. It is distinct from an inferred label (a posterior commit, a
propagated face) and from operator truth. It may teach only where this file says, and only at
the grade it says.

## Where a sensor-attested label may be used

| Learner | `trajectory` grade | `room` grade | Code |
| --- | --- | --- | --- |
| Body gallery rows | mints `auto_verified`, `seed_source = wearable_attested`, votes at `resolver.gallery_wearable_trust_multiplier` (1.25) | mints `pending_review`, `seed_source = wearable_room_attested`; votes only after operator approval | `candidate_eligibility.WearableSeed`, `ReIDCandidateStage` |
| Face exemplars (person-id) | operator-gated `attested` captures | same, tagged with the grade | `WearableFaceAttestationService`, person-id `POST /api/v1/calibration/attestations` |
| Unknown-history backfill | NULL-only, over the attested segment, behind `resolver.enable_wearable_backfill` (owner confirmation pending) | never | `UnknownBackfillService._handle_wearable_segment` |
| Image-only evaluation truth | yes | yes, reported separately | `IdentityResolver._image_only_evaluation`, `ImageOnlyEvaluationService` |

## Where it may never be used

- Enrollment or ArcFace prototypes without an explicit operator promotion.
- Authority: a wearable is never an authority rung (`WEARABLE.grants_authority` is False).
- Calibration fitting, until the person-id audit bar is met (at least 200 audited, Wilson 95%
  lower bound of precision 0.99 or better); the manifest builder enforces it, not a setting.
- Any learner while `wearable.mode` is not `enabled` for gallery seeding: an `auto_verified` row
  votes at once, so startup refuses `reid_candidates.wearable_seeding_enabled` otherwise.

## Gallery seeding rules

- The wearable path runs only when no face seed qualifies on the detection. Gates, in order:
  `no_identity`, `wearable_not_attested`, `identity_mismatch` (the tag's member must equal the
  PH's committed identity), `face_conflict` (any recognized face of anyone else on the detection,
  live or shadow-only, at any similarity), then the shared orientation, embedding, and quality
  gates.
- Spacing and diversity read the identity's own rows from the repository (no in-memory map):
  at most one wearable seed per identity and orientation per
  `reid_candidates.wearable_min_seed_interval_s`, none whose cosine to a same-day row exceeds
  `wearable_max_similarity_to_recent`, at most `wearable_room_grade_max_per_day` room-grade seeds
  per identity per UTC day, and the shared per-orientation creation cap.
- Every seeded row records `seed_association_id` (the association segment), so a wrong binding
  is demoted per segment through the review service, never raw SQL.
- `gallery_scoring.score_hits` is the only place the trust multiplier is applied; an
  `operator_verified` row votes at full trust whatever its seed source.

## Image-only evaluation

For a sampled share (`wearable.evaluation.sample_rate`, default 0.0) of attested PH resolutions,
the resolver decides the PH twice without the wearable channel, as if it held no identity (the
held identity may itself come from the wearable): control without any wearable-taught row,
treatment with them. In both arms no row seen in the last `wearable.evaluation.exclusion_s`
votes, so the arms differ only in older wearable-taught rows. Arm reads pass a `_GalleryArm`
through `_gallery_search_kwargs`, never kwargs built at a call site. Seconds are capture-time
gaps between consecutive sampled evaluations of a PH (capped by `max_gap_s`) and persist per
identity, UTC day, arm, and grade in `image_only_evaluation_daily` (migration `0010`).
`scripts/image_only_evaluation_report.py` reports the paired difference with a bootstrap
interval over days.

## Confirmation-loop guard

Attestation of either grade reads only wearable fixes, PH floor and room history, and face anchors
for the conflict rule; never gallery votes or the posterior. Learning from attested segments
therefore cannot feed back into what attests them.
