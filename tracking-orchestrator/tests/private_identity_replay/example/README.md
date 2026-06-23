# Synthetic Example Dataset

This directory contains a **synthetic, fictional** example manifest for testing
the replay loader and harness. No real household data is present here.

All identities (`fictional-alice-001`, `fictional-bob-002`) are entirely made up.
Frame sha256 values are placeholder zeros; actual frame bytes do not exist.

For the full replay setup guide, see the public documentation:
`docs/development/private-identity-replay.md`.

## Contents

- `manifest.json` -- synthetic manifest covering seven required replay cases.

## What must NOT be placed here

- Real household images, crops, or video.
- Embeddings or calibration arrays.
- Populated manifests with real sha256 digests.
- Generated reports with MinIO object keys.

These are all listed in `continuous-tracking/.gitignore`.
