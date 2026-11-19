# CTS wire format

Every protobuf-backed CTS Redis Stream carries exactly one protobuf message
type. The envelope is dead-simple: each Redis Streams message has a single field
whose value is the raw ``Message.SerializeToString()`` output. Stream
consumers know the type they expect, so no codec discriminator is
required.

## Per-stream contract

### Protobuf streams

| Stream | Field | Proto type | Producer | Consumer |
|--------|-------|------------|----------|----------|
| `frames.ready` | `frame` | `continuoustracking.v1.FrameReady` | `rtsp-ingress` (Go) | `tracking-orchestrator` |
| `tracking.events` | `event` | `continuoustracking.v1.TrackingEvent` | `tracking-orchestrator` | `cognitive-companion` (`TrackingEventSubscriber`) |
| `tracking.revisions` | `revision` | `continuoustracking.v1.IdentityRevision` | `tracking-orchestrator` | `cognitive-companion` (`IdentityRevisionSubscriber`) |
| `tracking.signals` | `signal` | `continuoustracking.v1.DementiaSignal` | `tracking-orchestrator` | `cognitive-companion` (`SignalSubscriber`) |
| `scene.samples` | `sample` | `continuoustracking.v1.SceneSample` | `tracking-orchestrator` | `cognitive-companion` (`SceneSampleSubscriber`) |
| `cc.identity_assertions` | `assertion` | `continuoustracking.v1.CCIdentityAssertion` | `cognitive-companion` (`CCIdentityAssertionPublisher`) | `tracking-orchestrator` (`CCIdentityAssertionSubscriber` -> `WorldTrackingStage`) |
| ~~`tracking.presence`~~ | `presence` | `continuoustracking.v1.PresenceEvent` | (retired in M37; `deprecated = true` in `tracking.proto`, no producer) | `cognitive-companion` (`PresenceEventSubscriber`, idle) |
| ~~`tracking.dwell`~~ | `dwell` | `continuoustracking.v1.DwellEvent` | (retired in M37; `deprecated = true` in `tracking.proto`, no producer) | `cognitive-companion` (`DwellEventSubscriber`, idle) |
| ~~`tracking.responses`~~ | `response` | `continuoustracking.v1.FrameResponse` | (retired 2026-06, TD-006; no producer) | `cognitive-companion` drain (idle) |

Field names mirror the message they carry so `XRANGE` output is
self-explanatory.

### Non-protobuf field-map streams (exceptions)

Two outbound streams carry flat string field maps published via
`RedisStreamsTransport.publish()` rather than protobuf bytes:

| Stream | Fields | Domain type | Producer | Consumer |
|--------|--------|-------------|----------|----------|
| `tracking.continuations` | `closed_ph_id`, `new_ph_id`, `from_room_id`, `to_room_id`, `gap_seconds`, `match_cost`, `occurred_at` | `PHContinuationCandidate` | `tracking-orchestrator` (`publish_continuations`) | `cognitive-companion` (`ContinuationSubscriber`) |
| `tracking.room_transitions` | `ph_id`, `identity_id`, `from_room_id`, `to_room_id`, `from_camera_id`, `to_camera_id`, `occurred_at`, `confidence` | `RoomTransitionEvent` | `tracking-orchestrator` (`publish_room_transitions`) | `cognitive-companion` (`RoomTransitionSubscriber`) |

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

* `TrackingEvent` carries live Bayesian posterior snapshots in
  `repeated IdentitySnapshot identity_snapshots = 8;` (`ph_id`,
  `committed_identity_id`, `posterior`, `entropy_bits`, `committed_at`,
  `face_confirmed`). The earlier `repeated IdentityRevision identity_revisions = 5;`
  sub-message is marked `deprecated = true` in `tracking.proto` and is no
  longer populated on `TrackingEvent`.
* `IdentityRevision` on `tracking.revisions` uses `ph_id` (tag 1), reserves
  `tracklet_ids` (tag 7), and carries typed range and projection fields
  (`reason_code`, `authority_source`, `trigger_source`, `range_mode`,
  `effective_from`, `effective_to`, `confidence`, `requires_downstream_sync`,
  `supersedes_revision_id`, `evidence_ids`, `operator_id`) so downstream
  consumers (`IdentityRevisionSubscriber` and `PersonLocationService`) can
  apply time-bounded or full-PH projections without reparsing `evidence_json`.
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
3. Update the producer to populate the field. Add a round-trip or contract test
   under `tracking-orchestrator/tests/test_proto_contracts.py` (or
   `tracking-orchestrator/tests/contracts/test_ph_contract_names.py`) if the
   field affects serialised bytes or cross-repo wire names.
4. Update consumers to read the field. Default values are wire-safe;
   old consumers ignore unknown fields and old producers emit zero
   values for new fields.

`buf breaking` enforces wire compatibility per the `WIRE` rule set.
The schema package is `continuoustracking.v1` and is frozen; bump to
`v2` if a breaking change is required.
