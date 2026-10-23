---
paths: ["**/tracking-orchestrator/app/storage/**"]
---

# SQL

- asyncpg `$1..$N` positional params only (also in `executemany`); never `%s`, `?`, f-strings, or
  `%` formatting. Count the placeholders against the argument tuple.
- Schema-qualify every table: `continuous_tracking.<table>`. `search_path` is set only during
  migrations.
- Array concatenation in `ON CONFLICT DO UPDATE SET` is `col = EXCLUDED.col || table.col`, never
  wrapped in `array[...]`.
- When splicing a clause with `str.replace`, anchor on a unique substring at the insertion point,
  never a trailing keyword such as `LIMIT 100`.
- Parse JSONB explicitly, reject non-object values, and never coerce strings or arrays to `{}`.
- A filter in a Postgres query appears identically in the `InMemory` peer and the `Protocol`
  signature. Detail: the engineering-standards skill, `reference/storage.md`.
