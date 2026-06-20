# ADR: Identity Revision Projections

- Status: Accepted
- Date: 2026-06-19
- Owners: Continuous Tracking, Cognitive Companion
- Supersedes: Whole-PH mutable identity rewrites without projection acknowledgements

## Context

An inferred label describes what the resolver concluded at observation time. An operator may later
correct a bounded interval without changing the original inference. Downstream location, presence,
keyframe, and signal views must converge on the same effective label while retaining an audit path
to the original evidence.

CTS already publishes protobuf `IdentityRevision` messages on `tracking.revisions`. Cognitive
Companion already supersedes `PersonLocationHistory` rows and records `CtsIdentityRevisionLog`.
The existing projection is extended rather than replaced.

## Decision

`inferred_identity_id` is immutable resolver output. `effective_identity_id` is the revision-aware
label used by consumers. `person_id` remains the live identity key inside Cognitive Companion; it
is not deprecated.

Operator corrections:

- target explicit observation boundaries, not arbitrary timestamps;
- may target one frame or a caregiver-confirmed bounded segment;
- never cross split, merge, or prior operator-revision boundaries automatically;
- preserve the original inference and evidence;
- change the live PH label only when the corrected range reaches the live edge;
- cannot be superseded by an inferred revision.

CTS is the correction source of truth. Each revision has a stable `revision_id`, parent or
compensated revision linkage, actor, reason, range, expected version, and evidence summary.

## Projection jobs and acknowledgements

A correction creates an idempotent projection job. Each required projection records an
acknowledgement keyed by the same `revision_id`, projection name, and attempt. Required projections
include CTS history/read models and the configured Cognitive Companion location, presence,
keyframe, and signal projections.

A job is complete only when every required projection acknowledges that revision. Retries are safe
and do not duplicate rows, WebSocket events, or audit records. Partial failure remains visible as a
job state; it is not converted to success.

Compensating revisions reverse or replace an earlier effective projection while preserving the
earlier revision and all acknowledgements. No correction, review event, or revision is edited or
deleted to simulate undo.

## Wire compatibility

New protobuf fields use new tag numbers. Old readers ignore additions; new readers accept messages
without the additions during the stated compatibility window. Raw protobuf bytes remain the only
supported encoding on CTS Redis streams. Compatibility adapters exist at one decode boundary and
have an explicit removal milestone.

## Consequences

- A correction request may return before all projections complete.
- APIs expose revision and job state rather than claiming immediate global consistency.
- Keyframe cards show only effective identity; details retain original inference and revision
  history.
- Projection failures are retryable and observable by revision ID.

## Review checklist

- [ ] Original inferred identity is immutable.
- [ ] Every correction range uses observation boundaries.
- [ ] Live labels change only when the range reaches the live edge.
- [ ] Each projection is idempotent by revision ID.
- [ ] Completion requires all configured acknowledgements.
- [ ] Undo creates a compensating revision.

## Related records

- [Identity authority and Unknown](identity-authority-and-unknown.md)
- [ReID gallery governance](reid-gallery-governance.md)
- [Program baseline](identity-integrity-program-baseline.md)

