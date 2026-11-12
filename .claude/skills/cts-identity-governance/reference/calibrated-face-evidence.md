# Calibrated face evidence

How recognized faces become identity evidence and how much they weigh. One settings key,
`face_id.evidence_policy`, drives both the emission in `FaceIdentityStage` and the weighting in
`IdentityResolver`; `app/main.py` passes the same value to `FaceIdConfig.evidence_policy` and
`ResolverConfig.face_evidence_policy`, so the two cannot disagree. The domain type is
`FaceEvidencePolicy` (`app/domain/__init__.py`).

## Policies

| Policy | Emission (`_gate_outcome`) | Weighting (`_recognized_face_likelihood`) |
| --- | --- | --- |
| `legacy` | A recognized face below the camera floor, or `face_id.min_raw_similarity` when the camera sets none, is `dropped_min_similarity` | `_p_face(match_score, effective quality) * frontality` |
| `shadow` | As `calibrated`, except a face only `calibrated` would emit is `emitted_shadow_only` | Live as `legacy`; a calibrated decision is computed alongside and only counted |
| `calibrated` | Every recognized face is emitted unless the camera sets a floor (`face_id_min_raw_similarity` from CC, an operator escape hatch) | Calibrated confidence when present, else the capped heuristic below |

Attribution (`dropped_not_in_head_region`, `dropped_ambiguous`) is decided first and is
policy-independent: a face that is not the person's never becomes evidence, in shadow included.
Candidates and unrecognized markers are emitted identically under every policy.

## Calibrated weighting

```text
p = calibrated_confidence * propagation_or_external_multiplier * frontality(yaw)
p = min(_p_face(match_score, effective_quality), uncalibrated_face_max_likelihood)
      * propagation_or_external_multiplier * frontality(yaw)        # no calibrated confidence
```

- The calibrated confidence is already P(same identity | similarity, quality): never pass it
  through `_p_face` and never multiply face quality in again (the artifact's features include it).
  Frontality stays because yaw is not a calibration feature.
- `resolver.uncalibrated_face_max_likelihood` (0.55) caps a face without a calibrated confidence:
  a resident outside the artifact's `covered_identities`, a propagated anchor (propagation resets
  the value to `None`), or a `cc_assertion` anchor.
- The strongest recognized anchor is chosen by this likelihood, not by `match_score * quality`.
- When no relevant recognized anchor carries a calibrated confidence, the face remainder `1 - p`
  goes to `UNKNOWN` instead of spreading across every enrolled identity.

Face templates (`reference/face-templates.md`) take the calibrated branch under every policy,
because a template's raw similarity is inflated by averaging.

## Corroboration only

Under `calibrated`, `collect_evidence_identity_ids(..., calibrated_faces_only=True)` admits a face
anchor only when it is recognized and carries a calibrated confidence. Every other face
(uncalibrated recognized, candidate, propagated, external) shapes the posterior but is never commit
or clock evidence: it cannot establish an identity, cannot be the evidence for a switch, and never
refreshes `last_independent_identity_evidence_at`. ReID votes qualify as before. The
remainder-to-`UNKNOWN` rule alone does not guarantee this: a capped 0.55 face still yields a
posterior of 0.96 (one resident) to 0.66 (three residents); the evidence gate is the guarantee.

ArcFace authority is unchanged: calibrated confidence of at least
`arcface_authority_calibrated_confidence` (0.80), native, recognized, entity-matched.

## Shadow-only anchors (rollout scaffolding)

Under `shadow`, a recognized face below the legacy floor becomes a `FaceAnchor` with
`shadow_only=True` and outcome `emitted_shadow_only` (migration 0007). The stage writes it to
`FrameContext.shadow_face_anchors`, never to `face_anchors`, and builds no `FaceEvidence` for it,
so association, dedup, gallery minting (`ReIDCandidateStage`), ReID cadence, and evidence records
cannot see it. `WorldTrackingStage` appends it to the face list it passes to `WorldTracker.step`
for identity resolution only, and `IdentityResolver.resolve` removes it from the live list before
propagation. It is therefore excluded from authority, the evidence clock, both evidence sets,
duplicate ranking, and association vectors.

The shadow evaluation (`_shadow_calibrated_face`) runs for every PH with a recognized face while
the policy is `shadow`: ArcFace authority over live plus shadow anchors, then the calibrated
likelihood, then commit evidence under calibrated rules, each side with its own `_contradicted`. It
counts `cts_face_shadow_outcomes_total{outcome}` (`agrees`, `would_commit_new`,
`would_change_identity`, `would_become_unknown`) and, on disagreement,
`cts_identity_shadow_mismatch_total{feature="calibrated_face"}` and the log event
`calibrated_face_shadow_mismatch`. Delete `shadow_only`, `shadow_face_anchors`, the
`emitted_shadow_only` outcome, and the shadow evaluation one release after the flip to
`calibrated`.
