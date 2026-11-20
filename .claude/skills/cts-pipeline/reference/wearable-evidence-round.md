# Wearable Evidence Round and Persistence

Wearable localization fixes enter `tracking-orchestrator` asynchronously on the `wearable.fixes`
Redis stream and join the frame pipeline once per tracker round:

1. **Subscriber (`tracking-orchestrator/app/services/wearable_fix_subscriber.py`):** started
   only when `wearable.mode != "off"`. Decodes `TagFix` protobufs respecting every presence
   flag, drops `OFF_BODY` and stale (`> wearable.max_fix_age_s`, 15 s) fixes with counters,
   resolves `member_id` via `WearableSiteConfig.member_at(tag_id, measured_at)`, dead-letters
   malformed payloads (`cts_messages_dead_lettered_total{stream="wearable_fixes"}` + XACK),
   and appends accepted fixes into `WearableFixBuffer` ordered by `measured_at`.
2. **World tracking round (`WorldTrackingStage.run_many` -> `WorldTracker.step`):**
   reads recent buffered fixes for `batch_time`, appends calibrated-only PH updates to
   `PHFloorHistory`, evaluates `WearableAssociationTracker.step`, passes accepted
   `WearableEvidence` into `IdentityResolver.resolve`, and feeds any `wearable_face` conflicts
   back into the attestation tracker.
3. **Once-per-round persistence (`WearableAssociationPersistStage`):**
   `WorldTrackingStage._populate_context` attaches `wearable_association_transitions` only to
   the primary context (`include_revisions=ctx is primary_context`), matching `new_revisions`.
   `WearableAssociationPersistStage` (`tracking-orchestrator/app/pipeline/stages/wearable_associations.py`)
   sits in `post_world_stages` ahead of `PublishStage`, so segment open/update/close rows write
   once per tracker round across direct, per-camera batch, and cross-camera batch paths and are
   never dropped by `PublishStage`'s `live_publish_max_hz` rate limit.
