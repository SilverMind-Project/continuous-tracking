# CTS wire format

Every CTS Redis Stream carries exactly one protobuf message type. The
envelope is dead-simple: each Redis Streams message has a single field
whose value is the raw ``Message.SerializeToString()`` output. Stream
consumers know the type they expect, so no codec discriminator is
required.

## Per-stream contract

| Stream | Field | Proto type | Producer | Consumer |
|--------|-------|------------|----------|----------|
| `frames.ready` | `frame` | `continuoustracking.v1.FrameReady` | `rtsp-ingress` (Go) | `tracking-orchestrator` |
| `tracking.events` | `event` | `continuoustracking.v1.TrackingEvent` | `tracking-orchestrator` | `cognitive-companion` |
| `tracking.revisions` | `revision` | `continuoustracking.v1.IdentityRevision` | `tracking-orchestrator` | `cognitive-companion` |
| `tracking.signals` | `signal` | `continuoustracking.v1.DementiaSignal` | `tracking-orchestrator` | `cognitive-companion` |
| `scene.samples` | `sample` | `continuoustracking.v1.SceneSample` | `tracking-orchestrator` | `cognitive-companion` (worker not yet implemented) |
| ~~`tracking.responses`~~ | `response` | `continuoustracking.v1.FrameResponse` | (retired 2026-06, TD-006) | `cognitive-companion` drain (idle) |

Field names mirror the message they carry so `XRANGE` output is
self-explanatory.

## Implementation

### Python

```python
from app.transport.codec import encode, decode
from app.proto.continuoustracking.v1 import tracking_pb2

# Publish
fields = encode(my_event, field="event")
await redis.xadd("tracking.events", fields, maxlen=10000, approximate=True)

# Consume
event = decode(fields, tracking_pb2.TrackingEvent, field="event")
```

The Redis client must run with `decode_responses=False` so binary proto
bodies round-trip unchanged.

### Go (rtsp-ingress)

```go
payload, _ := proto.Marshal(meta)
redis.XAdd(ctx, &redis.XAddArgs{
    Stream: "frames.ready",
    Values: map[string]any{"frame": payload},
})
```

## Schema evolution

* `IdentityRevision` is dual-purpose: tags 1-5 are populated when it
  appears as a `TrackingEvent` sub-message (Bayesian posterior snapshot
  per global track per event); tags 6-11 are populated when it appears
  as the standalone `tracking.revisions` message (revision_id,
  tracklet_ids, previous_identity_id, new_identity_id, reason,
  evidence_json). Consumers ignore tags they don't care about.
* Dictionary-shaped fields (`evidence`, `context`, `annotations`) are
  serialised as JSON strings (`*_json`) rather than `google.protobuf.Struct`
  to keep Go and Python code uniform and to leave wire bytes
  human-readable in `XRANGE`.
* Enums (`DementiaSignalKind`, `DementiaSignalSeverity`, `TagReason`)
  carry an explicit `_UNSPECIFIED = 0` value -- consumers reject
  messages whose enum is unspecified, so a producer that forgets to
  populate the field surfaces immediately rather than silently
  defaulting.

## Adding a new field

1. Add the field to the relevant `.proto` file with a fresh tag number.
2. Run `make proto`. The Makefile regenerates Go bindings (committed
   alongside the `.proto` files) and Python bindings (committed in both
   `tracking-orchestrator/app/proto/` and
   `cognitive-companion/backend/integrations/proto/`).
3. Update the producer to populate the field. Add a round-trip test
   under `tracking-orchestrator/tests/test_codec.py` if the field
   affects serialised bytes.
4. Update consumers to read the field. Default values are wire-safe;
   old consumers ignore unknown fields and old producers emit zero
   values for new fields.

`buf breaking` enforces wire compatibility per the `WIRE` rule set.
The schema package is `continuoustracking.v1` and is frozen; bump to
`v2` if a breaking change is required.
