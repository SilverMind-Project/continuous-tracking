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
| Training the household appearance classifier | direct capture (`label_source = wearable_attested`, audited `>= 100` spot-check with Wilson 95% lower bound `>= 0.99`) | only after operator approval (`operator_verified` room-grade gallery rows imported offline) | `AppearanceSampleRecorder`, `scripts/train_household_classifier.py` |

## Where it may never be used

- Enrollment or ArcFace prototypes without an explicit operator promotion.
- Authority: a wearable is never an authority rung (`WEARABLE.grants_authority` is False).
- Calibration fitting, until the person-id audit bar is met (at least 200 audited, Wilson 95%
  lower bound of precision 0.99 or better); the manifest builder enforces it, not a setting.
- Any learner while `wearable.mode` is `off`: startup refuses
  `reid_candidates.wearable_seeding_enabled` when `wearable.mode == "off"`. Under
  `wearable.mode: shadow`, trajectory-grade seeds are skipped (`channel_not_live`, because
  `auto_verified` rows would vote immediately), while room-grade seeds may still propose
  `pending_review` rows (`wearable_room_attested`) for operator inspection when the PH has no
  conflicting committed identity (`committed_identity_id is None` or matches the tag's member) and
  no face conflict.

## Gallery seeding rules

- The wearable path runs only when no face seed qualifies on the detection. Gates, in order:
  `wearable_not_attested`, `channel_not_live` (trajectory grade under `wearable.mode != "enabled"`),
  `teaching_held` (active hold on `tag_id` in `wearable_teaching_holds`), `no_identity` (unless
  room-grade shadow proposal with `committed_identity_id is None`), `identity_mismatch` (committed
  identity differs from the tag's member), `face_conflict` (any recognized face of anyone else on
  the detection, live or shadow-only, at any similarity), then the shared orientation, embedding,
  and quality gates.
- Spacing and diversity read the identity's own rows from the repository (no in-memory map):
  at most one wearable seed per identity and orientation per
  `reid_candidates.wearable_min_seed_interval_s`, none whose cosine to a same-day row exceeds
  `wearable_max_similarity_to_recent`, at most `wearable_room_grade_max_per_day` room-grade seeds
  per identity per UTC day, and the shared per-orientation working-set cap (counted inside the
  active vote window).
- Every seeded row records `seed_association_id` (the association segment), so a wrong binding
  is demoted per segment through `WearableTeachingRevocationService`, never raw SQL.
- `gallery_scoring.score_hits` is the only place the trust multiplier is applied; an
  `operator_verified` row votes at full trust whatever its seed source.

## Revoking what a wearable taught

`WearableTeachingRevocationService` (`app/services/wearable_revocation.py`, migration `0012`)
undoes everything a single association segment (`segment_id`) taught, idempotently:

1. Demotes every `auto_verified` gallery row with `seed_association_id = segment_id` back to
   `pending_review` via `GalleryRepository.apply_review_action(action="demote", reason="wearable_revoked:<reason>")`
   (vector kept, one audit event per row; `operator_verified` rows are never touched).
2. Calls person-id `POST /api/v1/calibration/attestations/revoke` to mark every attested face
   capture from that segment revoked (excluded from manifest, audit, and promotion, and purged by
   retention even if previously audited `correct`). Already-promoted prototypes are returned for
   manual review and never removed automatically.
3. Compensates live inferred backfill ranges with `source_association_id = segment_id` through
   `IdentityCorrectionService.compensate_inferred_range` (setting `source_association_id = None` on
   the compensating range, clearing backfilled rows to `NULL` via `IdentityRewriter.clear_identity_rows`,
   and publishing a compensating `IdentityRevision`).
4. Marks any `appearance_training_samples` rows with `seed_association_id = segment_id` as revoked
   (`revoked_at` stamped via `AppearanceSampleRepository.revoke_by_association`), excluding them
   from offline classifier training.
5. Opens a teaching hold on `segment.tag_id` and on every other tag sharing its
   `physical_device_id` (a dual-mode watch's BLE and UWB tags) in `wearable_teaching_holds`
   (at most one open hold per tag). While held, gallery seeding, face attestation, wearable
   backfill, and appearance training sample capture skip those tags until an operator clears
   the hold via `POST /internal/wearables/tags/{tag_id}/resume-teaching`, which clears every
   tag of the device. Holding one tag alone would let the next segment, opened through the
   sibling tag, keep teaching.
6. Stamps `revoked_at`, `revoked_by`, and `revoke_reason` on the association segment.

When `wearable.revocation.auto_on_swap` is true (default), `WearableAssociationPersistStage`
invokes `revoke_for_swap(segment_id, window_s=wearable.swap_conflict_window_s)` whenever a
segment transitions to `swap_suspected`, revoking the triggering segment plus any other unrevoked
segments of the same physical device's tags overlapping the swap-conflict window and holding
those tags.

## Image-only evaluation

For a sampled share (`wearable.evaluation.sample_rate`, default 0.0) of attested PH resolutions,
the resolver decides the PH twice without the wearable channel, as if it held no identity (the
held identity may itself come from the wearable): control without any wearable-taught row,
treatment with them. In both arms no row seen in the last `wearable.evaluation.exclusion_s`
votes, so the arms differ only in older wearable-taught rows. Arm reads pass a `_GalleryArm`
through `_gallery_search_kwargs`, never kwargs built at a call site. Seconds are capture-time
gaps between consecutive sampled evaluations of a PH (capped by `max_gap_s`) and persist per
identity, UTC day, arm, and grade in `image_only_evaluation_daily` (migration `0010`).
`app/services/image_only_evaluator.py` owns the shared daily-rate and paired-bootstrap summary
used by both `GET /internal/identity/readiness` (`wearable_learners`) and
`scripts/image_only_evaluation_report.py`.

## Confirmation-loop guard

Attestation of either grade reads only wearable fixes, PH floor and room history, and face anchors
for the conflict rule; never gallery votes or the posterior. Learning from attested segments
therefore cannot feed back into what attests them.
