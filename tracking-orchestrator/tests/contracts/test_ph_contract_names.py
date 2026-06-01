"""Contract name tests — PH-native field names.

Asserts that public Pydantic schemas expose ``ph_id``, not ``global_track_id``
or ``tracklet_id``, except in explicitly approved boundary files.

R3 state: GlobalTrackRepository, CloseTerminatedStage, and the dead tracking/hints
storage modules are removed. Generated protobuf bindings and public router
schemas must expose PH-native names only.
The transport snapshot (IdentitySnapshot) uses ph_id directly.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.transport.redis_streams import _build_tracking_event_pb

# -- Approved boundary files that may still reference legacy names -----------
# These files are grandfathered until their respective refactoring milestones.
_APPROVED_LEGACY_FILES: set[str] = {
    "app/domain/__init__.py",
}

# -- Forbidden field names in public Pydantic schemas ------------------------
_FORBIDDEN_FIELDS: set[str] = {"global_track_id", "tracklet_id"}

# -- Directories to scan for Pydantic schemas ---------------------------------
_SCHEMA_DIRS: list[str] = ["app/routers"]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_approved(file_path: str) -> bool:
    for approved in _APPROVED_LEGACY_FILES:
        if file_path.startswith(approved) or approved in file_path:
            return True
    return False


def _find_schema_files() -> list[Path]:
    """Find all Python files in schema directories."""
    files: list[Path] = []
    for schema_dir in _SCHEMA_DIRS:
        target = _PROJECT_ROOT / schema_dir
        if not target.exists():
            raise RuntimeError(f"Schema directory does not exist: {target}")
        files.extend(sorted(target.rglob("*.py")))
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
    ids=lambda p: str(p.relative_to(_PROJECT_ROOT)),
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
        f"outside approved boundary files. Violations:\n" + "\n".join(violations)
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


def test_identity_snapshot_proto_uses_ph_id():
    """IdentitySnapshot proto message must expose ph_id, not global_track_id.

    This test encodes the rename contract: the generated _pb2 binding for
    IdentitySnapshot must have a ph_id attribute and must NOT have global_track_id.
    Binary wire format is unchanged (field number 1 is ph_id).
    """
    from app.proto.continuoustracking.v1 import tracking_pb2

    snap = tracking_pb2.IdentitySnapshot(ph_id="ph-test-1", identity_id="alice")
    assert snap.ph_id == "ph-test-1"
    assert not hasattr(snap, "global_track_id"), (
        "IdentitySnapshot must not expose global_track_id (R3 rename); "
        "IdentitySnapshot.global_track_id was renamed to ph_id"
    )


def test_transport_identity_snapshot_sets_ph_id_not_global_track_id() -> None:
    """_build_tracking_event_pb must set ph_id on IdentitySnapshot, not global_track_id."""
    source = inspect.getsource(_build_tracking_event_pb)
    assert "s.ph_id" in source, "transport must set s.ph_id on IdentitySnapshot"
    assert "s.global_track_id" not in source, (
        "transport must not set s.global_track_id on IdentitySnapshot after R3 rename"
    )
