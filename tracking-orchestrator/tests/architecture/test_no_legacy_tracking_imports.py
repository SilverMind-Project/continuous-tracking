"""WTR9: Architecture test — no legacy tracking module imports in product code.

These modules were deleted or deprecated in WTR9. Product code outside
the approved grandfathered list must not import them.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Modules forbidden in product code (outside grandfathered files).
_FORBIDDEN_MODULES = (
    "app.storage.tracking",
    "app.storage.global_track",
    "app.storage.hints",
    "app.tracking.tracker",
    "app.tracking.cross_camera",
    "app.tracking.tracklet_manager",
    "app.tracking.global_track_service",
    "app.tracking.global_track_merger",
    "app.pipeline.stages.local_tracking",
    "app.pipeline.stages.global_tracking",
)

# Files grandfathered to still reference legacy modules during transition.
_GRANDFATHERED = (
    "app/storage/base.py",  # re-exports for backward compat
    "app/storage/__init__.py",
    "tests/",  # tests may reference legacy types
)


def _find_product_files() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    app_dir = root / "app"
    if not app_dir.exists():
        return []
    files = []
    for f in app_dir.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        rel = str(f.relative_to(root))
        if any(rel.startswith(g.replace("tests/", "")) for g in _GRANDFATHERED if g.startswith("tests")):
            continue
        if rel in _GRANDFATHERED:
            continue
        files.append(f)
    return files


def _has_forbidden_import(file_path: Path) -> list[str]:
    violations: list[str] = []
    try:
        tree = ast.parse(file_path.read_text())
    except SyntaxError:
        return violations
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in _FORBIDDEN_MODULES:
                    if alias.name == forbidden or alias.name.startswith(forbidden + "."):
                        violations.append(f"  {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for forbidden in _FORBIDDEN_MODULES:
                    if node.module == forbidden or node.module.startswith(forbidden + "."):
                        violations.append(f"  from {node.module}")
    return violations


def test_no_product_code_imports_legacy_tracking():
    """Product code outside grandfathered files must not import legacy tracking."""
    all_violations: list[str] = []
    for f in _find_product_files():
        v = _has_forbidden_import(f)
        if v:
            rel = str(f.relative_to(f.parents[2]))
            all_violations.append(f"{rel}:\n" + "\n".join(v))

    assert not all_violations, (
        "Product code imports legacy tracking modules that were deleted in WTR9:\n\n"
        + "\n\n".join(all_violations)
    )
