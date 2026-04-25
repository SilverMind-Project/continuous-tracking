# CTS protobuf schemas

Source of truth for the wire contracts shared between the
`tracking-orchestrator` (Python), `rtsp-ingress` (Go), and the
`cognitive-companion` BFF (Python).

## Schemas

| File | Messages |
|------|----------|
| `continuoustracking/v1/frame.proto` | `FrameReady`, `FrameResponse` |
| `continuoustracking/v1/tracking.proto` | `TrackingEvent`, `Detection`, `BoundingBox`, `FloorPoint`, `IdentityRevision`, `IdentityCandidate`, `FrameRef` |

Streams that have not yet been migrated to proto (tracking.revisions,
tracking.signals, scene.samples) keep their existing JSON-blob
envelopes; see [docs/wire-format-migration.md](../docs/wire-format-migration.md)
for the rollout plan.

## Code generation

Generated bindings are committed under
`tracking-orchestrator/app/proto/continuoustracking/v1/` and
`cognitive-companion/backend/integrations/proto/continuoustracking/v1/`.

### Tooling requirements

| Tool | Purpose | How to install |
|------|---------|----------------|
| `buf` | Lint + Go codegen | `make go-tools` (project-pinned to v1.50.0) |
| `protoc` >= 25 | Python codegen | `apt install protobuf-compiler` (Debian/Ubuntu) or `brew install protobuf` (macOS) |

`protoc-gen-go` lives under `tools/go-bin/` from `make go-tools`. The
Python plugin is built into protoc itself (`--python_out`,
`--pyi_out`), so no separate plugin install is needed.

### Regenerating

```bash
# Go bindings (committed alongside the .proto files):
make proto-go

# Python bindings (committed into both projects):
make proto-py

# Everything:
make proto

# Lint:
make proto-lint
```

The CI pipeline runs `buf lint` and `buf breaking` against the merged
HEAD; both must pass before merge. Generated files have to be in-tree
because the Python codegen requires `protoc` on PATH which is not
guaranteed in every reviewer's environment.

## Adding a new field

1. Add the field to the relevant `.proto` file with a fresh tag number.
2. Run `make proto` to regenerate Go + Python bindings.
3. Update the producer to populate the field. Bump
   `tests/test_codec.py` to assert round-trip if the field affects
   wire shape.
4. Update consumers to read the field. Default values are wire-safe;
   old consumers ignore unknown fields and old producers emit zero
   values for new fields.

`buf breaking` enforces wire compatibility per the WIRE rule set; the
schema package is `continuoustracking.v1` and is frozen — bump to
`v2` if a breaking change is required.
