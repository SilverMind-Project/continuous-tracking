# ADR: Identity Authority and Unknown

- Status: Accepted
- Date: 2026-06-19
- Owners: Continuous Tracking, Cognitive Companion, Person Identification
- Supersedes: Ad hoc identity authority inferred from mutable PH labels

## Context

A Person Hypothesis (PH) is the persistent world-coordinate track. Association decides which
observation belongs to a PH; identity resolution decides what evidence supports a household
identity. Treating either result as permanent truth caused identity swaps, duplicate active
identities, and stale priors that renewed themselves.

Raw model output, effective labels, operator corrections, and learning data therefore have
different authority and retention rules.

## Decision

Identity authority is ordered as follows:

1. An operator correction inside its explicit observation-bounded range.
2. Direct ArcFace evidence that is unambiguously associated with the observation, has compatible
   model and preprocessing versions, and clears the calibrated authority threshold.
3. A PH posterior supported by operator-verified, version-compatible ReID entries.
4. A temporal prior no older than 30 seconds from the last independent qualifying evidence.

Raw ArcFace cosine similarity, propagated face hints, height, pending or rejected ReID entries,
and PH-local appearance prototypes are non-authoritative. They may corroborate or improve
association but cannot create an effective identity.

The independent-evidence clock is separate from resolver evaluation time, persistence time, and
identity update time. Prior-only maintenance, height, propagated face evidence, and ordinary PH
writes do not refresh it.

When material evidence conflicts and the authority order does not resolve the conflict, the
effective identity is `Unknown`. Unknown is an explicit safe result, not an error or a missing
default.

## Duplicate-active invariant

At most one open global PH may hold a household identity. Multiple same-time camera observations
may update that one PH after cross-camera deduplication.

When active contenders claim one identity:

- retain one winner only when its independent evidence is clearly stronger;
- set all other contenders to `Unknown`;
- set every contender to `Unknown` when the evidence is effectively tied;
- persist conflict provenance and emit the invariant metric.

The occupancy check includes open incumbent PHs not observed in the current frame. A shadow
comparison may tune thresholds, but shadow mode cannot be the final enforcement state.

## Consequences

- A temporary increase in `Unknown` is acceptable and measurable.
- Raw inference remains immutable; corrections affect the effective projection.
- ArcFace enrollment remains a separate golden dataset.
- Resolver and association changes require reviewed two-person replay coverage.
- The current 120-second prior and disabled duplicate guard are baseline defects scheduled for resolution.

## Review checklist

- [ ] No non-authoritative evidence can create an effective identity.
- [ ] Only independent qualifying evidence advances the evidence clock.
- [ ] Conflict resolves to `Unknown` unless higher authority resolves it.
- [ ] Duplicate-active occupancy includes incumbents outside the current frame.
- [ ] Raw similarity is never presented as calibrated confidence.

## Related records

- [ReID gallery governance](reid-gallery-governance.md)
- [Identity revision projections](identity-revision-projections.md)
- [Program baseline](identity-integrity-program-baseline.md)

