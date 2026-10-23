# Storage, SQL, and migrations

Read before adding or changing a repository, a SQL statement, or a migration. Part of the `engineering-standards` skill; the rules summary is in `../SKILL.md`.

## Data Abstraction Layer

The goal is backend portability: swap Postgres for a different store by replacing one class, not touching services.

### Python: Protocol + InMemory + Postgres triplet

Every persistent resource gets three artifacts:

1. A `Protocol` in `storage/base.py` -- the contract services depend on.
2. An `InMemory*` class in the same file -- fast, zero-dependency, used in all unit tests.
3. A `Postgres*` class in `storage/postgres/` -- the production asyncpg implementation.

Services import only the `Protocol`. Import-linter enforces the layering contract (`core → domain → storage → tracking.world → services → transport → routers`). Nothing above `storage` may import a concrete `Postgres*` class.

```python
# storage/base.py
class PHRepositoryProtocol(Protocol):
    async def save(self, ph: PersonHypothesis) -> None: ...
    async def get(self, ph_id: str) -> PersonHypothesis | None: ...
    async def list_open(self) -> list[PersonHypothesis]: ...
    async def evidence_backed_commit(self, ph_id: str, identity_id: str, evidence_at: datetime, committed_at: datetime) -> None: ...
    async def prior_only_update(self, ph_id: str, identity_id: str, committed_at: datetime) -> None: ...
    async def clear_to_unknown(self, ph_id: str, committed_at: datetime) -> None: ...
```

Rules:

1. All repository methods are `async`.
2. Return domain types, never raw DB rows or dicts.
3. `Protocol` carries no state -- it is a pure structural interface.
4. `InMemory` uses plain `dict` / `list`; never touches a database or network.
5. `Postgres*` receives only an `asyncpg.Pool`; holds no other state.

The shared PostgreSQL instance (`timescale/timescaledb-ha:pg18`) hosts the `continuous_tracking` database alongside `cognitive_companion` and `semantic_memory`. The database and a dedicated `continuous_tracking` schema are created by `../db/db/init-databases.sh` (mounted at `/docker-entrypoint-initdb.d/`). Every table, index, trigger, and function lives in the `continuous_tracking` schema -- nothing goes to `public`.

### Wiring completeness for repository Protocols

A repository abstraction is done only when all three artifacts exist and production wiring is
proved: a Postgres implementation, explicit injection through `app/main.py`, and an integration
test exercising a production-shaped path. The Protocol plus InMemory implementation is not done.

```python
# RIGHT: app/main.py; proof: tests/integration/test_baseline_repo_postgres.py
baseline_repo = PostgresBehaviorBaselineRepository(_pool)
deps = PipelineDependencies(baseline_repo=baseline_repo)

# WRONG: production silently reaches an InMemory default
deps = PipelineDependencies()  # initialize() substitutes InMemoryBehaviorBaselineRepository
```
An optional dependency may have an InMemory default only for tests. The production constructor
site must name the concrete class. For every repository Protocol or ABC in `app/storage/`, run:

```bash
grep -nE "class .*(Protocol|ABC)" tracking-orchestrator/app/storage/*.py
grep -n "PostgresXRepository" tracking-orchestrator/app/main.py  # repeat for each X
```

Any missing counterpart requires a written justification comment at the wiring site.

### Repository state/version filters must match across the triplet

Any filter baked into a Postgres query (state, active, version) must appear identically in
the InMemory peer and in the Protocol signature as an explicit parameter with a shared
default constant. A filter that exists in only one implementation is a parity defect even if
all tests pass. New filtered reads require a parity matrix test.

**RIGHT:** `app/storage/gallery.py` exports two shared state-filter constants, used identically
across the Protocol, `InMemoryGalleryRepository`, and `PostgresGalleryRepository`:
`VOTING_STATES` (`operator_verified` + `auto_verified`) is the default for every read that feeds
identity resolution (`search_similar`, `list_gallery_entries_for_tracklets`,
`gallery_similarity`), and `VERIFIED_ONLY` is the default for the admin listing read
(`list_gallery_entries`). See the cts-identity-governance skill for the vocabulary rule.
`tests/storage/test_gallery_state_parity.py` (InMemory, fast gate)
and `tests/integration/test_gallery_state_parity_postgres.py` (Postgres, `make ci`) prove both
implementations return identical entry-ID sets for every state-filter value.

**WRONG:** a Postgres SQL constant hard-codes `state = 'operator_verified'` while the InMemory
peer applies no state filter at all -- unit tests seeded with the default lifecycle state then
"prove" behavior (a pending/rejected row voting) that production, filtered by the SQL, can never
produce.

## Database Migrations

### Tool split (intentional, not to be unified)

| Service | Tool | Why |
|---|---|---|
| `tracking-orchestrator` | Custom `MigrationRunner` + raw SQL | asyncpg + TimescaleDB non-transactional DDL requires the `-- migrate:no-transaction` split path; no SQLAlchemy ORM |
| `cognitive-companion` | Alembic | SQLAlchemy ORM with `target_metadata` autogenerate support |

### Lifecycle: post-release, incremental

The squashed baseline `0001_init` is frozen. Every schema change gets its own numbered file on
top of the current highest number: `NNNN_description.up.sql` / `NNNN_description.down.sql`
(CTS) or `NNNN_description.py` (CC). Rollbacks are supported and the `downgrade()` /
`.down.sql` must exactly reverse the up migration. Never fold a change into the baseline and
never drop and recreate a database to apply one: live household identity history must survive
upgrades.

Shape constraints ride the migration that adds the column: JSONB columns that the domain treats
as objects must have `CHECK (jsonb_typeof(column) = 'object')`, and repository mappers must
reject invalid rows instead of silently coercing them.

### CTS migration conventions

- Files: `migrations/NNNN_description.up.sql` and `migrations/NNNN_description.down.sql`.
- Zero-pad to 4 digits: `0001`, `0002`, …; list the existing ones with `ls tracking-orchestrator/migrations`.
- Every DDL object (tables, indexes, triggers, functions, materialized views) lives in the
  `continuous_tracking` schema. Open every migration with
  `SET search_path = continuous_tracking, public;` so unqualified `CREATE TABLE` / `CREATE INDEX` /
  TimescaleDB function calls land in the correct schema.
- Non-transactional DDL (hypertables, continuous aggregates, policies) must declare:
  `-- migrate:no-transaction` as the **first line** of the file.
- The baseline down file is a no-op comment directing the operator to drop and recreate.
- Applied state is tracked in `continuous_tracking._schema_version` (managed by `MigrationRunner`,
  `app/storage/migrations.py`).
- Advisory lock key `0x4354535F4D4947` ("CTS_MIG") with `pg_try_advisory_lock` ensures only one
  replica runs migrations.
- Migrations run automatically at orchestrator startup and can also be invoked standalone:

```bash
cts-db migrate          # apply pending up migrations
cts-db rollback -n 1    # roll back last migration
cts-db status           # show applied / pending
```

- `DATABASE_URL`: asyncpg expects a plain `postgresql://` DSN. If a `postgresql+asyncpg://`
  (SQLAlchemy-style) URL is provided, `_normalize_dsn()` strips the `+asyncpg` prefix.

### CC migration conventions

- Files: `alembic/versions/NNNN_description.py`.
- The baseline `downgrade()` is intentionally a no-op (`pass`). Document this in the docstring.
- Post-release migrations must have a correct `downgrade()` that exactly reverses `upgrade()`.
- Run `alembic check` against the live models to verify no schema drift before committing.

### Verification

A new migration is proven by the integration suite: the session testcontainer applies every
migration through `MigrationRunner` (`tests/conftest.py`), so `make test-integration` fails on a
broken up file. Test the down file by hand with `cts-db rollback -n 1` against a scratch database.
`detections.global_track_id` and its index are intentionally retained as plain columns (no FK);
they carry historical frame-level data.

## SQL and asyncpg correctness

These rules come from bugs caught during implementation and are non-negotiable.

**`$N` placeholders always.** asyncpg uses `$1, $2, …` positional params. Never use `%s` or `?`, and never build SQL with f-strings or `%`. `executemany` also requires `$N` style -- one row's worth of placeholders per statement.

**Match bind parameter count exactly.** Count `$1..$N` placeholders and verify the argument tuple has the same arity. Off-by-one raises `ValueError: bind parameter $N not found` at runtime.

**Schema-qualify every table reference.** All tables, views, and materialized views live in the `continuous_tracking` schema. Every SQL statement in `storage/postgres/` must write `continuous_tracking.<table>` (e.g. `FROM continuous_tracking.dementia_signals`, `INSERT INTO continuous_tracking.person_hypotheses`). Never rely on `search_path` to resolve unqualified names; the Python application queries span many sessions and the search_path is only set during migration execution.

**PostgreSQL array concatenation is `||`.** In `ON CONFLICT DO UPDATE SET`, write `col = EXCLUDED.col || table.col`. The `array[...]` constructor is for array literals only -- wrapping a `||` expression in it is a syntax error.

**Anchor conditional SQL replacements.** When adding `AND extra_clause` to a query via `str.replace`, replace a unique substring that includes the insertion point. Never replace a trailing keyword like `LIMIT 100` and prepend `AND …` -- that produces invalid SQL (predicate after `ORDER BY`).

**JSONB shape is part of the schema.** If application code expects a JSON object, enforce `jsonb_typeof(col) = 'object'` in the migration that adds the column and validate at repository boundaries. asyncpg may return JSONB as text depending on codec configuration; parse it explicitly, reject non-object values, and do not coerce strings or arrays into empty dicts.
