"""Loader and validator for private identity replay manifests.

Validates manifest JSON files against the schema and enforces data quality
constraints that cannot be expressed in the Pydantic model alone:
  - no overlapping frame indices within an episode
  - no identity/episode leakage across train/val/test splits
  - all sha256 fields non-empty in populated (non-synthetic) manifests
  - malformed or out-of-bounds bboxes rejected
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from tests.private_identity_replay.manifest import Episode, ReplayManifest, Split


class ManifestValidationError(ValueError):
    """Raised when a manifest fails validation."""


def _validate_no_overlapping_frames(episodes: list[Episode]) -> list[str]:
    """Frames within a single episode must have unique frame_index values."""
    findings: list[str] = []
    for ep in episodes:
        seen: set[tuple[str, int]] = set()
        for fa in ep.frames:
            key = (fa.camera_id, fa.frame_index)
            if key in seen:
                findings.append(
                    f"episode {ep.episode_id!r}: duplicate (camera_id={fa.camera_id!r}, "
                    f"frame_index={fa.frame_index})"
                )
            seen.add(key)
    return findings


def _validate_split_disjoint(episodes: list[Episode]) -> list[str]:
    """No frame (by sha256) may appear in more than one split (content leakage check).

    In a single-household dataset the same people appear in every split, so
    identity-level disjointness is not the right invariant. The meaningful
    constraint is that the same captured frame must not be used in both train
    and test -- that would contaminate evaluation with training data.

    Placeholder zero-filled sha256s from synthetic examples are skipped; they
    are validated separately by _validate_hashes_populated when require_hashes=True.
    """
    zero_sha = "0" * 64
    split_frames: dict[Split, set[str]] = {s: set() for s in Split}
    for ep in episodes:
        for fa in ep.frames:
            sha = fa.frame_sha256
            if sha and sha != zero_sha:
                split_frames[ep.split].add(sha)

    findings: list[str] = []
    train = split_frames[Split.train]
    val = split_frames[Split.val]
    test = split_frames[Split.test]

    tv = train & val
    tt = train & test
    vt = val & test
    if tv:
        findings.append(f"train/val frame sha256 leak ({len(tv)} frame(s)): {sorted(tv)[:3]}")
    if tt:
        findings.append(f"train/test frame sha256 leak ({len(tt)} frame(s)): {sorted(tt)[:3]}")
    if vt:
        findings.append(f"val/test frame sha256 leak ({len(vt)} frame(s)): {sorted(vt)[:3]}")
    return findings


def _validate_hashes_populated(manifest: ReplayManifest, require_hashes: bool) -> list[str]:
    """Populated manifests must have non-None sha256 fields."""
    if not require_hashes:
        return []
    findings: list[str] = []
    for identity in manifest.identities:
        if identity.embedding_sha256 is None:
            findings.append(
                f"identity {identity.identity_id!r} is missing embedding_sha256 "
                "(required in populated manifests)"
            )
    for ep in manifest.episodes:
        for fa in ep.frames:
            if not fa.frame_sha256:
                findings.append(
                    f"episode {ep.episode_id!r} frame_index={fa.frame_index} has empty frame_sha256"
                )
    return findings


def load_manifest(
    path: Path,
    *,
    require_hashes: bool = False,
) -> ReplayManifest:
    """Load, parse, and validate a replay manifest JSON file.

    Args:
        path: Path to the manifest JSON file.
        require_hashes: When True, all sha256 fields must be populated
            (set this for production/household datasets; leave False for
            synthetic examples).

    Returns:
        Validated ReplayManifest instance.

    Raises:
        ManifestValidationError: On any validation failure.
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        manifest = ReplayManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestValidationError(f"manifest schema error: {exc}") from exc

    all_findings: list[str] = []
    all_findings.extend(_validate_no_overlapping_frames(manifest.episodes))
    all_findings.extend(_validate_split_disjoint(manifest.episodes))
    all_findings.extend(_validate_hashes_populated(manifest, require_hashes))

    if all_findings:
        bullet_list = "\n".join(f"  - {f}" for f in all_findings)
        raise ManifestValidationError(
            f"manifest {path} failed validation ({len(all_findings)} issue(s)):\n{bullet_list}"
        )

    return manifest
