# Household appearance classifier

`HouseholdClassifierChannel` (`app/tracking/identity/household_classifier.py`) is an optional,
corroborating closed-set channel trained offline on frozen body embeddings across multiple days
and outfits. It ships dark (`household_classifier.mode: "off"`,
`household_classifier.capture_enabled: false`) until the post-wearable necessity gate triggers.

## Governance and evidence grade

- **Grade (`HOUSEHOLD_CLASSIFIER`, `EvidenceGradeName.HOUSEHOLD_CLASSIFIER`):** declared in
  `app/tracking/identity/evidence_grades.py`. Never grants authority (`grants_authority = False`),
  never refreshes the independent-evidence clock (`refreshes_evidence_clock = False`,
  `evidence_at = None`), never ranks in the duplicate-active guard, never enters world association,
  and never propagates.
- **Conditional commit evidence:** `counts_as_commit_evidence = True` on the grade, but the
  channel populates `ChannelLikelihood.commit_evidence_ids` only for trained classes whose
  calibrated probability clears `household_classifier.commit_evidence_min_probability` (default
  `0.8`). Below that floor, the channel corroborates other sensory evidence without committing an
  Unknown PH on its own.
- **Weighting and open-set UNKNOWN floor:** pre-softmax logits are scaled by
  `household_classifier.weight_multiplier / temperature` with `support_multiplier = 1.0` so the
  weight does not cancel under normalization. `UNKNOWN` mass is
  `u = max(unknown_floor, 1.0 - max(p_cal))` (default `unknown_floor = 0.2`), trained classes
  receive `(1 - u) * p_weighted[c]`, and untrained enrolled relatives stay absent from the
  distribution (Laplace neutral fill in `combine_channels`).

## Sample capture and revocation

- `AppearanceSampleRecorderStage` (`app/pipeline/stages/appearance_samples.py`) runs in
  `post_world_stages` ahead of `PublishStage` across all three pipeline execution paths when
  `household_classifier.capture_enabled` is true.
- Eligible positive sources in `continuous_tracking.appearance_training_samples` (60-day
  TimescaleDB retention, `embedding vector(768)`): calibrated `direct_face` authority decisions
  and `trajectory`-grade wearable-attested intervals (excluding held tags and detections with a
  conflicting recognized face). Posterior-only inferred labels never enter training, and
  `room`-grade wearable intervals enter training only after explicit operator approval
  (`reid_gallery` rows with `state = 'operator_verified'` and
  `seed_source = 'wearable_room_attested'`).
- Open-set negatives (`label_source = 'negative'`, `identity_id IS NULL`, capped per local day)
  are captured only from named guest PHs or uncommitted PHs carrying an `unrecognized` direct face
  anchor, preventing back-view resident tracks from poisoning the negative set.
- `WearableTeachingRevocationService.revoke` stamps `revoked_at` on training samples tied to the
  revoked `seed_association_id`. Deleting a resident cascades via `ON DELETE CASCADE` and
  `IdentityRosterSyncService.purge_identity`.

## Offline training, feature seam, and artifact contract

- `AppearanceFeatureExtractor` (`app/domain/household_classifier.py`) separates embedding
  extraction from the linear head. v1 (`SoliderFeatureExtractor`) reads existing SOLIDER-REID
  view prototypes (falling back to the current observation embedding or `gallery_mean`). Swapping
  to a DINOv2 Triton model introduces a new `feature_version`, invalidating prior artifacts.
- `tracking-orchestrator/scripts/train_household_classifier.py` runs offline behind the optional
  `classifier-tools` extra (`scikit-learn`, forbidden from `app/` by `import-linter`). It enforces:
  1. Wearable association spot-check audit (`--association-audit` JSON: `sample_size >= 100` and
     recomputed Wilson 95% lower bound `>= 0.99`).
  2. At least 2 household classes, each with at least 3 distinct local days and 300 deduplicated
     samples in the pre-eval split plus held-out evaluation coverage.
  3. Leave-one-day-out CV for `C` (`LogisticRegression(class_weight="balanced")`, symmetric
     2-row export for 2-resident households) and temperature scaling on a held-out validation day.
  4. Held-out 3-day evaluation gates: macro top-1 accuracy `>= 0.85` at `p >= 0.8`, every trained
     class recall `>= 0.75` at `p >= 0.8`, ECE `<= 0.05`, and open-set false-match rate `<= 0.02`
     at `p >= 0.8`.
- Runtime loads only validated JSON (`HouseholdClassifierArtifact`, no pickle) and exposes
  `off`, `ready`, `degraded_missing`, `degraded_incompatible`, and `degraded_invalid` on
  `GET /health` and `cts_household_classifier_health{state}`.
- In `shadow` mode, `IdentityResolver` records `cts_household_classifier_shadow_outcomes_total`
  and runs the Part E `classifier` arm (`image_only_evaluation_daily`, migration `0015`) alongside
  `control` and `treatment` on the same sampled rounds.
