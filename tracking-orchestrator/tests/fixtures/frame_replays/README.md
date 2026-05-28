# Frame Replay Fixtures

On-disk format: length-prefixed JSON binary (R1). Each chunk is one call
to `WorldTracker.step()`, encoded as a JSON array of observation dicts.

```
[u32 length BE][JSON bytes] [u32 length BE][JSON bytes] ...
```

Note: `FrameReady` carries no detection data, so these fixtures store
pre-computed `WorldObservation` fields directly. See
`scripts/synthesize_replay_fixture.py` for the generator.

## Fixtures

| File | Description | Method |
|---|---|---|
| `two_cameras_one_room.bin` | One person: sequential cam-1 to cam-2 handoff (proves C1) | synthetic |
| `two_rooms_two_people.bin` | Two people in non-overlapping rooms, never merge (proves C2) | synthetic |

## Regeneration

```bash
cd tracking-orchestrator
uv run python scripts/synthesize_replay_fixture.py
```

Regenerate when `WorldObservation` fields change or tracker thresholds shift.

## Replaying

```bash
# Testcontainer Postgres started automatically; no external DB needed.
pytest -m integration tests/integration/test_world_tracker_e2e.py -v
```

`.gitattributes` marks `*.bin` as binary.
