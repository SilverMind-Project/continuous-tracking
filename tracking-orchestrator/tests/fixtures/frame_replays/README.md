# Frame Replay Fixtures

On-disk format: a length-prefixed stream of protobuf `FrameReady` messages.

```
[u32 length BE][protobuf bytes][u32 length BE][protobuf bytes]...
```

Each length is a 4-byte unsigned integer in big-endian byte order,
immediately followed by that many bytes of serialized `continuoustracking.v1.FrameReady`
protobuf message.

## Fixtures

| File | Description | Expected size |
|---|---|---|
| `two_cameras_one_room.bin` | One person walks under two cameras whose views overlap one room (kitchen) | ~5-15 MB |
| `two_rooms_two_people.bin` | Two people in two non-overlapping rooms, brief swap | ~5-15 MB |

## Recording a fixture

Run against a live dev stack:

```bash
cd tracking-orchestrator
uv run python -m scripts.record_replay_fixture \
    --camera-id cam_kitchen_1 --camera-id cam_kitchen_2 \
    --duration-seconds 30 \
    --output tests/fixtures/frame_replays/two_cameras_one_room.bin
```

The recorder subscribes to the `frames.ready` Redis stream and captures
messages from the named cameras for the specified duration.

## Replaying

```bash
pytest -m integration tests/integration/test_world_tracker_e2e.py
```

Requires testcontainer Postgres and Redis (the test boots these automatically).

## Regeneration

These fixtures should be regenerated when:
1. The `FrameReady` protobuf schema changes.
2. The camera configuration in the dev stack changes significantly.
3. A new tracker release requires fresh fixtures for validation.

Fixtures are binary files committed to git. Add `.gitattributes`:
```
*.bin binary
```
