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
| `hallway_bathroom_door.bin` | One person walking past bathroom door, two overlapping cameras (C1) | synthetic |
| `single_camera_turn.bin` | One person, one uncalibrated camera, turning front-to-back with 6 s occlusion gap (50 steps). Demonstrates PH churn + resolver demotion. M2 target: 1 PH with identity held. | synthetic (M1) |
| `cross_camera_handoff.bin` | One person crossing from cam-handoff-a to cam-handoff-b with 16 s gap (63 steps). Proves cross-camera disconnect; M5 target: identity persists. | synthetic (M1) |
| `two_people_one_room.bin` | Two enrolled people (alice + bob) crossing paths on one uncalibrated camera (26 steps). Guardrail: no over-merge. | synthetic (M1) |
| `resident_plus_stranger.bin` | Enrolled resident (alice) + unenrolled stranger crossing paths (30 steps). Clinical guardrail: resident identity never transferred to stranger. | synthetic (M1) |

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
