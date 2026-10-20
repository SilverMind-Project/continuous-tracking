"""Contract name tests: face evidence says which score it carries.

A face score is a raw cosine similarity (a match score), a detector score, or a calibrated
probability; only the last is a confidence. These tests keep the ambiguous ``confidence`` name
from returning on face types and face thresholds:

- ``FaceAnchor``, ``FaceEvidence`` and ``FaceResult`` expose ``match_score`` / ``raw_similarity``,
  never a bare ``confidence`` (``calibrated_confidence`` is a probability and keeps its name).
- No ``.confidence`` attribute access on a face-typed name in ``app/`` (AST walk over accesses
  whose base name ends with ``anchor``, ``fa``, ``face`` or ``evidence``).
- No old threshold or settings name in ``app/`` or ``config/settings.yaml``. The one exception is
  cognitive-companion's ``face_id_min_confidence`` camera field, decoded at a single boundary in
  ``app/main.py``.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from app.domain import FaceAnchor, FaceGateOutcome
from app.inference.evidence import FaceEvidence
from app.inference.face_id_client import FaceResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP = _PROJECT_ROOT / "app"
_SETTINGS = _PROJECT_ROOT / "config" / "settings.yaml"

_FACE_BASE_SUFFIXES = ("anchor", "fa", "face", "evidence")
# Base names that end with a face suffix but are not face evidence: "file:base".
_ATTRIBUTE_ALLOWLIST: set[str] = set()

_OLD_NAMES = (
    "face_id_min_confidence",
    "face_commit_min_confidence",
    "contradiction_face_confidence",
    "duplicate_identity_direct_face_min_confidence",
    "_direct_face_confidence",
    "face_conflict_threshold",
    "DROPPED_MIN_CONFIDENCE",
    "dropped_min_confidence",
)
# The cognitive-companion camera payload still names its field face_id_min_confidence; CTS
# decodes it into FaceIdCameraConfig.min_raw_similarity here and nowhere else.
_CC_BOUNDARY = ("app/main.py", "face_id_min_confidence", 1)


def _field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


@pytest.mark.parametrize("cls", [FaceAnchor, FaceEvidence, FaceResult])
def test_face_types_have_no_bare_confidence(cls: type) -> None:
    assert "confidence" not in _field_names(cls)


def test_face_result_has_one_similarity_field() -> None:
    names = _field_names(FaceResult)
    assert "raw_similarity" in names
    assert "similarity" not in names


def test_anchor_and_evidence_carry_match_score() -> None:
    assert {"match_score", "det_score"} <= _field_names(FaceAnchor)
    assert "match_score" in _field_names(FaceEvidence)


def test_gate_outcome_names_the_similarity_floor() -> None:
    assert FaceGateOutcome.DROPPED_MIN_SIMILARITY.value == "dropped_min_similarity"


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _face_confidence_accesses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(_PROJECT_ROOT).as_posix()
    found: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == "confidence"):
            continue
        base = _base_name(node.value)
        if base is None or not base.lower().endswith(_FACE_BASE_SUFFIXES):
            continue
        if f"{rel}:{base}" in _ATTRIBUTE_ALLOWLIST:
            continue
        found.append(f"{rel}:{node.lineno} {base}.confidence")
    return found


def test_no_confidence_access_on_face_names() -> None:
    violations = [
        hit
        for path in sorted(_APP.rglob("*.py"))
        if "proto" not in path.parts
        for hit in _face_confidence_accesses(path)
    ]
    assert not violations, "face scores are match_score or raw_similarity:\n" + "\n".join(
        violations
    )


def test_old_face_threshold_names_are_gone() -> None:
    boundary_file, boundary_name, boundary_count = _CC_BOUNDARY
    violations: list[str] = []
    for path in [*sorted(_APP.rglob("*.py")), _SETTINGS]:
        if "proto" in path.parts:
            continue
        rel = path.relative_to(_PROJECT_ROOT).as_posix()
        text = path.read_text()
        for name in _OLD_NAMES:
            count = text.count(name)
            if rel == boundary_file and name == boundary_name:
                if count != boundary_count:
                    violations.append(f"{rel}: {name} x{count}, expected x{boundary_count}")
            elif count:
                violations.append(f"{rel}: {name} x{count}")
    assert not violations, "old face vocabulary found:\n" + "\n".join(violations)


def test_face_id_settings_use_similarity_keys() -> None:
    text = _SETTINGS.read_text()
    assert re.search(r"^\s+min_raw_similarity:", text, re.MULTILINE)
    assert re.search(r"^\s+face_commit_min_similarity:", text, re.MULTILINE)
    assert re.search(r"^\s+contradiction_face_similarity:", text, re.MULTILINE)
    assert re.search(r"^\s+duplicate_identity_direct_face_min_similarity:", text, re.MULTILINE)
    assert re.search(r"^\s+face_conflict_min_similarity:", text, re.MULTILINE)
