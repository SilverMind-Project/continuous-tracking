# Evidence grades and channels

How the identity resolver governs anchor kinds and combines evidence sources without inline
origin or aggregation checks.

## Evidence grade registry

`app/tracking/identity/evidence_grades.py` is the single source of governance for every
`FaceAnchor`, keyed by `(AnchorOrigin, AnchorAggregation)` (`app/domain/__init__.py`). Unknown
pairs fail loudly with `KeyError` from `grade_of(anchor)`.

| Grade (`EvidenceGradeName`) | Key `(origin, aggregation)` | Authority | Clock refresh | Duplicate rank/bypass | Association/dedup | Contradiction | Propagates | Requires calibration | Own capture time | Missing quality default | Attach precedence | Scale key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `native_frame` | `("native", "frame")` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes | 2 | none (1.0) |
| `native_template` | `("native", "template")` | Yes (with per-anchor `authority_eligible`) | Yes | No | No | No | No | Yes | Yes | Yes | 2 | none (1.0) |
| `cc_assertion` | `("cc_assertion", "frame")` | No | No | No | No | Yes | Yes | No | No | No | 1 | `cc_assertion_likelihood_scale` |

- **No inline origin or aggregation checks.** `tests/architecture/test_evidence_grade_centralized.py`
  scans `app/` with the Python AST and fails if any module outside
  `app/tracking/identity/evidence_grades.py` reads `.origin` or `.aggregation`, compares against
  `AnchorOrigin`/`AnchorAggregation` or their literals, or reads `.shadow_only` outside
  `IdentityResolver.resolve()`'s anchor-splitting block.
- **Grade-aware attachment (`WorldTrackingStage._attach_face_anchors`).** Collisions on one
  `detection_id` keep the higher `attach_precedence` grade and count cross-grade overlaps in
  `cts_face_anchor_grade_overlap_total{kept, dropped}` (closed `EvidenceGradeName` labels); only
  same-grade collisions increment the defect counter `cts_face_anchor_duplicate_detection_total`.

## Evidence channels and posterior combination

`app/tracking/identity/channels.py` defines `EvidenceChannel`, `ChannelLikelihood`, and
`ResolverInputs`, and `app/tracking/identity/posterior.py` defines
`combine_channels(prior, channels)`:

- Built-in channels run in canonical order: `FaceChannel` (`face_weight_multiplier`, default 3.0),
  `GalleryChannel` (1.0), `HeightChannel` (`height_weight_multiplier`, default 1.5).
- For each identity, `combine_channels` multiplies `prior(id)` by each channel's factor in order:
  `1.0` when the channel is empty, `dist[id] * support_multiplier` when present, and Laplace fill
  `1 / (n + 1)` when absent from a non-empty channel.
- **Full-support channel weighting rule:** `support_multiplier` only affects the normalized
  posterior when some identities are absent from `distribution`. Any new channel that populates all
  identities plus `UNKNOWN` must apply its weight as a log-likelihood power (`p ** weight`) or
  pre-softmax logit scale before constructing `PosteriorDist` and set `support_multiplier = 1.0`.
- **Per-channel evidence time:** each `ChannelLikelihood` carries `evidence_at`,
  `clock_evidence_ids`, and optional per-identity `evidence_at_by_identity`.
  `IdentityResolver._resolve_clock_evidence` selects the newest qualifying timestamp across
  channels for the decided identity without moving the clock backwards.
- **Adding a new evidence kind:** add a new `EvidenceGrade` (for face-anchor kinds) or a new
  `EvidenceChannel` class plus config block and shadow-first rollout mode; never add `if origin ==`
  branches in the resolver, commit policy, tracker, or dedup.

## Typed external assertions

`ExternalIdentityAssertion` and `ExternalIdentityEvidenceSource` live in `app/domain/__init__.py`.
`IdentityAssertionCache` (`app/services/cc_identity_assertion_subscriber.py`) implements
`ExternalIdentityEvidenceSource` and is passed through `PipelineDependencies.external_evidence`
into `WorldTrackingStage(external_evidence=...)` via constructor injection (never post-init
attribute mutation).
