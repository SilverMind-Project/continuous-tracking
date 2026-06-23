"""Replay harness for private identity replay datasets.

Skips with a clear message when private data is absent.
Runs deterministically when private data is present.

Usage:
  pytest tests/private_identity_replay/test_harness.py

Private data placement:
  tests/private_identity_replay/data/<dataset-name>/manifest.json
  (this path is git-ignored -- see continuous-tracking/.gitignore)

For setup instructions, see the public guide:
  docs/development/private-identity-replay.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.private_identity_replay.loader import load_manifest
from tests.private_identity_replay.manifest import ReplayManifest, Split

# Private data lives under tests/private_identity_replay/data/.
# This directory is git-ignored; it is never committed.
_PRIVATE_DATA_ROOT = Path(__file__).parent / "data"

# Synthetic example (always present in the repo).
_EXAMPLE_MANIFEST = Path(__file__).parent / "example" / "manifest.json"


# ---------------------------------------------------------------------------
# Synthetic example tests (always run -- no private data needed).
# ---------------------------------------------------------------------------


class TestExampleManifest:
    """Basic loader tests against the committed synthetic example."""

    def test_example_loads_cleanly(self) -> None:
        manifest = load_manifest(_EXAMPLE_MANIFEST)
        assert manifest.schema_version == "1.0"
        assert manifest.dataset_name == "synthetic-fictional-example"

    def test_example_has_required_identities(self) -> None:
        manifest = load_manifest(_EXAMPLE_MANIFEST)
        ids = {i.identity_id for i in manifest.identities}
        assert "fictional-alice-001" in ids
        assert "fictional-bob-002" in ids

    def test_example_episodes_cover_required_cases(self) -> None:
        manifest = load_manifest(_EXAMPLE_MANIFEST)
        ep_ids = {e.episode_id for e in manifest.episodes}
        required = {
            "ep-train-both-clear-faces",
            "ep-train-profile-facing",
            "ep-train-crossing-occlusion",
            "ep-val-explicit-unknown",
            "ep-val-camera-overlap",
            "ep-test-face-absent-weak-reid",
            "ep-test-operator-correction",
        }
        assert required <= ep_ids

    def test_example_split_disjoint(self) -> None:
        # Loader enforces that no frame sha256 appears in more than one split.
        # load_manifest would raise ManifestValidationError if any frame leaked.
        manifest = load_manifest(_EXAMPLE_MANIFEST)
        # Verify no frame sha256 (non-zero-placeholder) appears in more than one split.
        split_frames: dict[Split, set[str]] = {s: set() for s in Split}
        zero_sha = "0" * 64
        for ep in manifest.episodes:
            for fa in ep.frames:
                if fa.frame_sha256 and fa.frame_sha256 != zero_sha:
                    split_frames[ep.split].add(fa.frame_sha256)
        all_splits = list(split_frames.values())
        for i, a in enumerate(all_splits):
            for b in all_splits[i + 1 :]:
                assert not (a & b), "frame sha256 leaked across splits"


# ---------------------------------------------------------------------------
# Private dataset tests (skip when data absent).
# ---------------------------------------------------------------------------


def _find_private_manifests() -> list[Path]:
    if not _PRIVATE_DATA_ROOT.exists():
        return []
    return sorted(_PRIVATE_DATA_ROOT.rglob("manifest.json"))


_PRIVATE_MANIFESTS = _find_private_manifests()

# When no private data is present, parametrize must still receive at least one
# entry so pytest can collect the class. The sentinel is marked skip so it
# never executes -- the class-level skipif guard fires first on real absence.
_PRIVATE_MANIFESTS_OR_SENTINEL = _PRIVATE_MANIFESTS or [
    pytest.param(
        Path("."),
        marks=pytest.mark.skip(
            reason=(
                f"private replay data not present at {_PRIVATE_DATA_ROOT}; "
                "see docs/development/private-identity-replay.md"
            )
        ),
    )
]

_SKIP_NO_PRIVATE_DATA = pytest.mark.skipif(
    not _PRIVATE_MANIFESTS,
    reason=(
        f"private replay data not present at {_PRIVATE_DATA_ROOT}; "
        "see docs/development/private-identity-replay.md"
    ),
)


@_SKIP_NO_PRIVATE_DATA
class TestPrivateDatasets:
    """Tests that run only when private household replay data is present."""

    @pytest.mark.parametrize("manifest_path", _PRIVATE_MANIFESTS_OR_SENTINEL)
    def test_private_manifest_validates(self, manifest_path: Path) -> None:
        """All private manifests must pass loader validation with hash checks."""
        manifest = load_manifest(manifest_path, require_hashes=True)
        assert isinstance(manifest, ReplayManifest)

    @pytest.mark.parametrize("manifest_path", _PRIVATE_MANIFESTS_OR_SENTINEL)
    def test_private_episodes_have_frames(self, manifest_path: Path) -> None:
        manifest = load_manifest(manifest_path, require_hashes=True)
        for ep in manifest.episodes:
            assert len(ep.frames) > 0, f"episode {ep.episode_id!r} has no frames"

    @pytest.mark.parametrize("manifest_path", _PRIVATE_MANIFESTS_OR_SENTINEL)
    def test_private_split_coverage(self, manifest_path: Path) -> None:
        manifest = load_manifest(manifest_path, require_hashes=True)
        splits_present = {ep.split for ep in manifest.episodes}
        assert Split.test in splits_present, (
            "private dataset must have at least one test-split episode"
        )

    @pytest.mark.parametrize("manifest_path", _PRIVATE_MANIFESTS_OR_SENTINEL)
    def test_private_no_placeholder_sha256(self, manifest_path: Path) -> None:
        """Reject placeholder zero-filled hashes that indicate missing data."""
        manifest = load_manifest(manifest_path, require_hashes=True)
        for ep in manifest.episodes:
            for fa in ep.frames:
                assert fa.frame_sha256 != "0" * 64, (
                    f"episode {ep.episode_id!r} frame {fa.frame_index} has placeholder sha256"
                )
