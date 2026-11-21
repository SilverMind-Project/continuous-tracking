# Roster Synchronization and Identity Purge

Part of the `cts-identity-governance` skill; the rules summary is in `../SKILL.md`.

`person-identification-service` is the source of truth for which household identities exist.
CTS must never keep stale gallery rows, height profiles, training samples, or cached identities
for a person who was deleted from enrollment.

## Automated sync and internal routes

- **`IdentityRosterSyncService`** (`tracking-orchestrator/app/services/identity_roster_sync.py`)
  runs at startup and every 60 seconds against `person-identification-service` (`GET /api/v1/members`),
  creating missing identity rows and purging unenrolled identities through
  `GalleryRepository.purge_identity(identity_id)`.
- **`GalleryRepository.purge_identity`** (`app/storage/gallery.py` and
  `app/storage/postgres/gallery_repo.py`) removes the identity and all dependent CTS rows in both
  Postgres and in-memory peers.
- **Immediate cache invalidation:** after a purge, `IdentityRosterSyncService` calls
  `IdentityResolver.invalidate_identity_cache()`, clears face-template buffers and in-memory
  wearable bindings for that identity, and records `cts_identity_roster_reconcile_total`.
- **Internal routes** (`app/routers/gallery.py`):
  - `DELETE /internal/gallery/identities/{identity_id}` purges a deleted member immediately.
  - `POST /internal/gallery/identities/sync` triggers an immediate roster reconciliation pass.
  Cognitive Companion's person enrollment and deletion endpoints call these hooks immediately
  through its tracking orchestrator client so CTS does not wait for the 60 s poll.

## Rule for new tables and buffers

Every new CTS table or in-memory buffer keyed by `identity_id` must either declare
`REFERENCES continuous_tracking.identities(identity_id) ON DELETE CASCADE` or be cleared
explicitly in `GalleryRepository.purge_identity` (both Postgres and InMemory peers) and in
`IdentityRosterSyncService.purge_identity`, with a parity test covering deletion.
