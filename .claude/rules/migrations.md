---
paths: ["**/tracking-orchestrator/migrations/**", "**/tracking-orchestrator/app/storage/**"]
---

# Migrations

- Every schema change is a new `NNNN_description.up.sql` / `.down.sql` pair on top of the current
  highest number (`ls tracking-orchestrator/migrations`). The baseline `0001_init` is frozen.
- The down file exactly reverses the up file. Never edit an applied migration, never fold a change
  into the baseline, and never drop and recreate the database: live identity history must survive.
- Open with `SET search_path = continuous_tracking, public;`; non-transactional DDL declares
  `-- migrate:no-transaction` as the first line. A JSONB column the domain treats as an object
  gets `CHECK (jsonb_typeof(col) = 'object')` in the same migration.
- Proof: `make test-integration` applies every migration; test the down file with
  `cts-db rollback -n 1` on a scratch database. Detail: the engineering-standards skill,
  `reference/storage.md`.
