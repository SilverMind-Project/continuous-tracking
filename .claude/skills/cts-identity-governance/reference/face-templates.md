# Face templates

A face template is several faces of one Person Hypothesis matched together by
person-identification-service (`POST /api/v1/match-template`), so that several small faces of a
walking person can identify them when none can alone. Settings: `face_templates.*`
(`mode: off | shadow | enabled`, ships `off`).

## Embeddings stay in memory

- `FaceIdentityStage` asks `/identify` for embeddings (`return_embedding`) only while templates
  are on. `FaceResult.embedding` and `BufferedFace.embedding` are excluded from repr and equality.
- The f16 wire form is encoded and decoded only in `app/inference/face_id_client.py`.
- `FaceTemplateBuffer` (`app/tracking/identity/face_template_buffer.py`) holds at most `top_k`
  faces per PH in memory; nothing logs or persists an embedding. A CTS table or log line carrying
  one is a release-blocking defect.

## Which faces join

Every face `_gate_outcome` attributes to the person (`emitted`, `emitted_shadow_only`,
`dropped_min_similarity`, `dropped_no_candidate`), whatever the evidence policy: a template's
weak members are exactly the faces the single-face floor drops. Faces outside the head region or
in an ambiguous crop never join. A face joins only with an embedding and a quality score from
`face_id.expected_quality_version` at or above `min_face_quality`; a face unlike the buffer's mean
(cosine below `outlier_min_similarity`) is rejected as another person's.

## Lifecycle and resets

- Faces leave `max_age_s` (15 s, the revival window) after capture; closing a PH is not a reset,
  so a revived PH keeps its faces.
- A PH's buffer resets when its committed identity changes to a different identity, on an operator
  merge, batch merge, split, correct, or batch correct (`routers/ph.py`), and when an identity is
  purged (`IdentityRosterSyncService`). Every reset invalidates the tickets issued before it, so a
  template requested before the reset is discarded when its result arrives.
- Requests run in background tasks (`FaceTemplateMatcher`, at most `max_concurrency` at once, one
  per PH, `max_in_flight` in total); a result applies to a later round once its PH is observed,
  and expires after `max_age_s`.

## Template anchors

`FaceTemplateService.pending_anchors` turns a result into a `FaceAnchor` with
`aggregation="template"`, `tracklet_id = ph_id` (its newest member's `detection_id` is
provenance only), `match_score` = template raw similarity, `face_quality` = the template quality,
`det_score` and `yaw_deg` = member medians, and a `FaceEvidence` record with `source="template"`.

- It reaches the resolver only (`WorldTracker.step(face_anchors=...)`), never
  `_attach_face_anchors`, association, dedup, or revival.
- It is always weighed and gated as calibrated evidence, whatever `face_id.evidence_policy` says:
  averaging inflates raw similarity, so single-face raw thresholds do not transfer. Its calibrated
  confidence is kept only from a `template` calibration profile on the expected quality scorer;
  without one it is capped at `uncalibrated_face_max_likelihood` and only corroborates.
- It never feeds the raw-threshold contradiction check, the duplicate-guard direct-face bypass, or
  face propagation.
- Authority: a recognized template with calibrated confidence at or above the authority threshold
  grants `direct_face` only when every member face person-id recognized on its own names the same
  identity (`FaceAnchor.authority_eligible`); otherwise it votes but cannot decide.
  `cts_face_template_authority_blocked_total{reason}` counts `member_disagreement` and
  `uncalibrated`.
- Known limitation: a template applied in a later round is committed with that round's time as its
  evidence time, so it can extend the maintenance window by up to `max_age_s` past its newest real
  face. Carry the newest member's capture time into the evidence timestamp before `enabled`.
- Shadow: template anchors are `shadow_only` and feed their own evaluation (the live decision
  again with the templates added), counted in `cts_face_template_shadow_outcomes_total`; they never
  enter the calibrated-policy shadow. Like it, the outcome counts are a lower bound on the Unknown
  cost because the live evidence clock is shared.
