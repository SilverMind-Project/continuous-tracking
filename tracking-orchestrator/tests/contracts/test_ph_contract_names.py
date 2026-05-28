"""WTR1: Contract name tests — PH-native field names.

Asserts that public Pydantic schemas expose ``ph_id``, not ``global_track_id``
or ``tracklet_id``, except in explicitly approved boundary files.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# -- Approved boundary files that may still reference legacy names -----------
# These files are grandfathered until their respective WTR milestones.
_APPROVED_LEGACY_FILES: set[str] = {
    "app/domain/__init__.py",  # WTR3: Detection, GlobalTrack, Tracklet types
    "app/transport/codec.py",  # WTR9: protobuf decode boundary
    "app/proto/",  # generated protobuf bindings (any file)
}

# -- Forbidden field names in public Pydantic schemas ------------------------
_FORBIDDEN_FIELDS: set[str] = {"global_track_id", "tracklet_id"}

# -- Directories to scan for Pydantic schemas ---------------------------------
_SCHEMA_DIRS: list[str] = ["app/routers"]


def _is_approved(file_path: str) -> bool:
    for approved in _APPROVED_LEGACY_FILES:
        if file_path.startswith(approved) or approved in file_path:
            return True
    return False


def _find_schema_files() -> list[Path]:
    """Find all Python files in schema directories."""
    root = Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for schema_dir in _SCHEMA_DIRS:
        target = root / schema_dir
        if target.exists():
            files.extend(target.rglob("*.py"))
    return files


def _extract_pydantic_field_names(file_path: Path) -> dict[str, list[str]]:
    """Parse a Python file and return {class_name: [field_names]} for Pydantic models."""
    try:
        tree = ast.parse(file_path.read_text())
    except SyntaxError:
        return {}

    models: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Check if it inherits from BaseModel (Pydantic)
        is_pydantic = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseModel":
                is_pydantic = True
                break
            if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
                is_pydantic = True
                break
        if not is_pydantic:
            continue

        field_names: list[str] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_names.append(item.target.id)
        models[node.name] = field_names

    return models


@pytest.mark.parametrize(
    "file_path",
    _find_schema_files(),
    ids=lambda p: str(p.relative_to(p.parents[1])),
)
def test_ph_api_schemas_use_ph_id_not_global_track_id(file_path: Path):
    """PH API response and request models must expose ph_id, not global_track_id."""
    if _is_approved(str(file_path)):
        pytest.skip(f"Approved boundary file: {file_path}")

    models = _extract_pydantic_field_names(file_path)
    violations: list[str] = []
    for class_name, field_names in models.items():
        for forbidden in _FORBIDDEN_FIELDS:
            if forbidden in field_names:
                violations.append(f"  {class_name}.{forbidden}")

    assert not violations, (
        f"{file_path.name}: Pydantic models must not expose legacy field names "
        f"outside approved boundary files. Violations:\n"
        + "\n".join(violations)
    )


def test_correction_request_models_use_ph_id():
    """Correction request models (merge, split, correct) must use ph_id."""
    from app.routers.ph_schemas import (
        BatchCorrectItem,
        CorrectIdentityRequest,
        MergeRequest,
        SplitRequest,
    )

    # MergeRequest uses source_ph_id and target_ph_id
    assert "source_ph_id" in MergeRequest.model_fields
    assert "target_ph_id" in MergeRequest.model_fields
    assert "global_track_id" not in MergeRequest.model_fields
    assert "tracklet_id" not in MergeRequest.model_fields

    # CorrectIdentityRequest uses ph_id via the batch/route path
    assert "global_track_id" not in CorrectIdentityRequest.model_fields
    assert "tracklet_id" not in CorrectIdentityRequest.model_fields

    # SplitRequest uses at_observation_id (not a track id)
    assert "global_track_id" not in SplitRequest.model_fields
    assert "tracklet_id" not in SplitRequest.model_fields

    # BatchCorrectItem uses ph_id
    assert "ph_id" in BatchCorrectItem.model_fields
    assert "global_track_id" not in BatchCorrectItem.model_fields
    assert "tracklet_id" not in BatchCorrectItem.model_fields


def test_ph_api_response_models_use_ph_id():
    """PH API response models (summary, detail) must expose ph_id."""
    from app.routers.ph_schemas import PHDetail, PHSummary

    assert "ph_id" in PHSummary.model_fields
    assert "global_track_id" not in PHSummary.model_fields
    assert "tracklet_id" not in PHSummary.model_fields

    assert "ph_id" in PHDetail.model_fields
    assert "global_track_id" not in PHDetail.model_fields
    assert "tracklet_id" not in PHDetail.model_fields


def test_revision_response_uses_ph_id():
    """RevisionResponse must use ph_id, not global_track_id."""
    from app.routers.ph_schemas import RevisionResponse

    assert "ph_id" in RevisionResponse.model_fields
    assert "global_track_id" not in RevisionResponse.model_fields
    assert "tracklet_id" not in RevisionResponse.model_fields
