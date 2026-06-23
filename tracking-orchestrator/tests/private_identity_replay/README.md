# Private Identity Replay Dataset

This package provides the manifest schema, loader, and test harness for
private household identity replay data.

**Private data is never committed.** The `.gitignore` in `continuous-tracking/`
excludes `tests/private_identity_replay/data/`, populated manifests,
household images/crops, embeddings, and calibration arrays.

For full setup and labeling instructions, see the canonical public guide:

> `docs/development/private-identity-replay.md`

## What is committed

| Path | Purpose |
| --- | --- |
| `manifest.py` | Pydantic manifest model and JSON Schema |
| `loader.py` | Loader, validator, and leakage checker |
| `test_harness.py` | pytest harness (skips when private data absent) |
| `example/manifest.json` | Synthetic fictional example covering all required cases |
| `example/README.md` | Notes on the example |

## Quick start (private data)

Place your dataset under:

```
tests/private_identity_replay/data/<dataset-name>/manifest.json
```

Then run:

```bash
pytest tests/private_identity_replay/test_harness.py -v
```

When private data is absent, the test harness skips with an explicit reason.
