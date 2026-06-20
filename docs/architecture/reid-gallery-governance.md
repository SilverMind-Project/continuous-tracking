# ADR: ReID Gallery Governance

- Status: Accepted
- Date: 2026-06-19
- Owners: Continuous Tracking, Cognitive Companion
- Supersedes: Ungoverned `reid_gallery.face_confirmed` voting

## Context

The current `reid_gallery` stores labeled 768-dimensional body embeddings without durable crop
provenance or an explicit lifecycle. Entries can be seeded from a held PH identity even when the
recognized face names another identity. This lets one association error become long-lived semantic
pollution.

PH-local appearance prototypes are tracking state. They are not labeled identity data and are not
an alternate gallery.

## Decision

Every `reid_gallery` entry has exactly one state:

- `pending_review`: retained for operator review and excluded from all resolver votes;
- `operator_verified`: eligible to vote when model and preprocessing versions match;
- `rejected`: excluded permanently; vector and dedicated crop are deleted.

Existing rows migrate to `pending_review`, including rows where `face_confirmed=true`.

Candidate creation requires a finite L2-normalized embedding, compatible versions, shared crop
quality gates, valid orientation, acceptable truncation and occlusion, and all of this provenance:

- immutable source frame key and immutable crop key;
- frame content hash and crop content hash;
- pixel bbox, image dimensions, camera, and capture time;
- source PH, observation, keyframe, and creation-decision identifiers;
- proposed identity, creation evidence, model version, and preprocessing version.

A face-derived candidate may be labeled only when the direct recognized ArcFace identity equals
the candidate identity. A held PH label cannot override a different recognized face.

## Review actions and retention

Review actions are immutable events:

- `Approve` promotes the proposal to `operator_verified`.
- `Relabel` records the old proposal and corrected identity, then promotes it.
- `Reject` records the reason, retains audit metadata and an embedding fingerprint, and deletes the
  vector plus dedicated crop object.
- `Undo` creates a compensating review event. It does not mutate or delete prior audit events.

Identity correction and gallery verification remain separate actions. Correcting a bbox does not
promote its embedding, and failed quality gates cannot be overridden.

## Voting and version policy

Only `operator_verified` entries vote. Repository queries, caches, and compatibility paths must
enforce the filter.

Verified hits receive a trust multiplier of 2.0 before identity aggregation. The multiplier does
not alter cosine similarity. Recency uses exponential decay with a seven-day half-life and no
floor:

`recency_factor = 2 ** (-(age_days / 7))`

Votes are partitioned by model and preprocessing version. Incompatible partitions are not queried.
Near-duplicate votes are capped or clustered by source episode, camera, and orientation. Decision
provenance records the contributing entry IDs, raw similarities, trust multipliers, recency
factors, and aggregate result.

## Consequences

- The current inert five-state `gallery_governance.py` is replaced in M05 rather than extended.
- Reviewable crops are separate MinIO objects; keyframes continue to reference raw `frames/...`
  objects.
- ArcFace enrollment, calibration datasets, PH-local prototypes, and ReID review data stay
  separate.
- Rejection performs privacy-preserving deletion without erasing audit history.

## Review checklist

- [ ] Pending and rejected entries cannot vote through any query or cache.
- [ ] Every candidate is traceable to an immutable frame and crop.
- [ ] Recognized face identity equals the candidate label.
- [ ] Model and preprocessing partitions are enforced.
- [ ] Reject deletes the vector and crop while retaining audit metadata.
- [ ] Undo is compensating, never destructive.

## Related records

- [Identity authority and Unknown](identity-authority-and-unknown.md)
- [Identity revision projections](identity-revision-projections.md)
- [Program baseline](identity-integrity-program-baseline.md)

