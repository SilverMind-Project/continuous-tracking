# Rule inventory: EG-M09 restructure of the CTS instruction files

Every section of the pre-restructure CTS `CLAUDE.md` and `engineering-standards` SKILL.md,
and where it now lives. Sections moved verbatim (then fixed in place); SKILL.md restates their
rules as one-liners. Delete this file after the restructure is reviewed.

| Source | Section | Destination |
| --- | --- | --- |
| engineering-standards | ## Architecture and SOLID Principles | `reference/python-patterns.md` |
| engineering-standards | ## Data Abstraction Layer / ### Go: interface contracts in `internal/` | `reference/go.md` |
| engineering-standards | ## Data Abstraction Layer | `reference/storage.md` |
| engineering-standards | ## Dependency Injection | `reference/python-patterns.md` |
| engineering-standards | ## Immutability and Pure Functions | `reference/python-patterns.md` |
| engineering-standards | ## Import Discipline | `reference/python-patterns.md` |
| engineering-standards | ## Type Safety | `reference/python-patterns.md` |
| engineering-standards | ## Async/Concurrency | `reference/python-patterns.md` |
| engineering-standards | ## Error Handling / ### Go | `reference/go.md` |
| engineering-standards | ## Error Handling | `reference/python-patterns.md` |
| engineering-standards | ## Configuration Management | `reference/operations.md` |
| engineering-standards | ## API Design (FastAPI) | `reference/operations.md` |
| engineering-standards | ## Testing Strategy / ### Go | `reference/go.md` |
| engineering-standards | ## Testing Strategy | `reference/testing-and-gates.md` |
| engineering-standards | ## Security at Boundaries | `reference/operations.md` |
| engineering-standards | ## Datetime and Timezones | `reference/operations.md` |
| engineering-standards | ## Structured Logging | `reference/operations.md` |
| engineering-standards | ## Observability | `reference/operations.md` |
| engineering-standards | ## Image Preprocessing | `reference/operations.md` |
| engineering-standards | ## Python Virtual Environment | `reference/operations.md` |
| engineering-standards | ## Database Migrations | `reference/storage.md` |
| engineering-standards | ## Quality Gate | `reference/testing-and-gates.md` |
| engineering-standards | ## Code Review Checklist | `reference/testing-and-gates.md` |
| engineering-standards | ## Python Libraries | removed: merged into `reference/libraries.md` from the CLAUDE.md copy, which is the superset (adds `triton-shared`) |
| engineering-standards | ## Go Libraries | removed: merged into `reference/libraries.md` from the CLAUDE.md copy (identical rows) |
| engineering-standards | ## Redis Stream Wire Format | `reference/domain-principles.md` |
| engineering-standards | ## Domain-Specific Design Principles | `reference/domain-principles.md` |
| engineering-standards | ## 25. Code Comments and Documentation | `.claude/rules/comments.md` |
| CLAUDE.md | ## Code Architecture Patterns / ### Storage: Protocol + InMemory + Postgres triplet | `engineering-standards/reference/storage.md` |
| CLAUDE.md | ## Database Migrations | `engineering-standards/reference/storage.md` |
| CLAUDE.md | ## Coding Rules / ### Python / asyncpg | `engineering-standards/reference/storage.md` |
| CLAUDE.md | ## Coding Rules / ### SQL correctness | `engineering-standards/reference/storage.md` |
| CLAUDE.md | ## Coding Rules / ### Tracking / data structures | `engineering-standards/reference/python-patterns.md` |
| CLAUDE.md | ## Code Architecture Patterns / ### Domain objects: frozen dataclasses | `engineering-standards/reference/python-patterns.md` |
| CLAUDE.md | ## Coding Rules / ### Go | `engineering-standards/reference/go.md` |
| CLAUDE.md | ## Code Architecture Patterns / ### Go: interface contracts | `engineering-standards/reference/go.md` |
| CLAUDE.md | ## Approved Libraries | `engineering-standards/reference/libraries.md` |
| CLAUDE.md | ## Redis Stream Wire Format | `engineering-standards/reference/domain-principles.md` |
| CLAUDE.md | ## Testing Strategy | `engineering-standards/reference/testing-and-gates.md` |
| CLAUDE.md | ## Engineering Standards | `engineering-standards/reference/testing-and-gates.md` |
| CLAUDE.md | ## Code Architecture Patterns / ### RTSP ingest: go2rtc sidecar | `docs/systems-architecture.md` |
| CLAUDE.md | ## Camera Configuration | `docs/systems-architecture.md` |
| CLAUDE.md | ## Code Architecture Patterns / ### WorldTracker.step: the 10-step loop | `docs/systems-architecture.md` |
| CLAUDE.md | ## Skills to load | kept in CLAUDE.md, plus a note on `reference/` and `.claude/rules/` |
| CLAUDE.md | ## What This System Does | kept in CLAUDE.md |
| CLAUDE.md | ## Design Documents | kept in CLAUDE.md, rewritten to list what `docs/systems-architecture.md` now covers |
| CLAUDE.md | ## System Architecture (mermaid diagram) | removed: identical diagram already in `docs/systems-architecture.md`; the infrastructure, GPU, and shared-library lines kept in CLAUDE.md |
| CLAUDE.md | ## Repository Layout | kept in CLAUDE.md, corrected (no `internal/streams/` or `k8s/`; siblings listed separately) |
| CLAUDE.md | ## Code Architecture Patterns / ### Layering | kept as a CLAUDE.md invariant; detail in `reference/storage.md` |
| CLAUDE.md | ## Code Architecture Patterns / ### Identity resolution: Bayesian posterior | removed: covered in more detail by `docs/systems-architecture.md` (identity resolver); the commit rule and revisions kept as a CLAUDE.md invariant |
| CLAUDE.md | ## Camera Configuration / "The CLAUDE.md for the rtsp-ingress architecture note" block | removed: verbatim repeat of the go2rtc flow |
| CLAUDE.md | ## Coding Rules / `datetime.now(UTC)` always | merged: `reference/operations.md` Datetime and `.claude/rules/time.md` |
| CLAUDE.md | ## Coding Rules / Declare all attributes in `__init__` | `reference/python-patterns.md` |
| CLAUDE.md | ## Coding Rules / Never mutate frozen dataclasses | merged into `reference/python-patterns.md` Never mutate frozen instances |
| CLAUDE.md | ## Coding Rules / Stable signal IDs | `reference/domain-principles.md` Stable signal IDs (correction history removed) |
| CLAUDE.md | ## Testing Strategy | removed: the engineering-standards Testing Strategy is the superset |
| CLAUDE.md | ## Engineering Standards (table) | `reference/testing-and-gates.md` Standards at a glance |
| CLAUDE.md | ## Claude Code guardrails | kept in CLAUDE.md verbatim |
| CLAUDE.md | ## Quality Gate | kept in CLAUDE.md, plus `make docs-check` |
| CLAUDE.md | ## Proto Codegen | kept in CLAUDE.md |
| CLAUDE.md | ## Known Tech Debt | kept in CLAUDE.md |
| CLAUDE.md | ## Known Tech Debt / ### Hardening program (2026-07), resolved | removed: resolved findings; CLAUDE.md points to the notes directory |
| CLAUDE.md | ## Working in This Repository | kept; the phase-0 validation-gates reference now names its file |
| engineering-standards | Go subsections and CLAUDE.md Go rules | merged into `reference/go.md` (duplicates collapsed; stale `streams/` example replaced with `poll/worker.go`) |
| engineering-standards | Redis Stream Wire Format (two copies) | merged into `reference/domain-principles.md`; the non-protobuf `tracking.continuations` and `tracking.room_transitions` streams recorded |
