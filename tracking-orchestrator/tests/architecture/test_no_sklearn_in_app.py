"""AST guard proving runtime app/ never imports scikit-learn, joblib, or pickle."""

from __future__ import annotations

import ast
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[2] / "app"
_FORBIDDEN_ROOTS: frozenset[str] = frozenset({"sklearn", "joblib", "pickle", "_pickle"})


def _scan_source(source: str, rel_path: str) -> list[str]:
    tree = ast.parse(source, filename=rel_path)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_ROOTS:
                    violations.append(
                        f"{rel_path}:{node.lineno}: forbidden import {alias.name!r} in app/"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _FORBIDDEN_ROOTS:
                violations.append(
                    f"{rel_path}:{node.lineno}: forbidden import from {node.module!r} in app/"
                )
    return violations


def test_app_never_imports_offline_ml_or_pickle() -> None:
    violations: list[str] = []
    scanned = 0
    for py_file in sorted(_APP_ROOT.rglob("*.py")):
        rel = py_file.relative_to(_APP_ROOT).as_posix()
        if rel.startswith("proto/"):
            continue
        scanned += 1
        violations.extend(_scan_source(py_file.read_text(encoding="utf-8"), rel))
    assert scanned >= 50, f"expected >= 50 app modules scanned, got {scanned}"
    assert not violations, "Forbidden offline/pickle imports in app/:\n" + "\n".join(violations)


def test_no_sklearn_guard_canary() -> None:
    """Non-vacuity canary: synthetic violations must be caught."""
    assert _scan_source("import sklearn.linear_model\n", "tracking/bad.py")
    assert _scan_source("from sklearn.linear_model import LogisticRegression\n", "tracking/bad.py")
    assert _scan_source("import pickle\n", "tracking/bad.py")
    assert _scan_source("import joblib\n", "tracking/bad.py")
    assert not _scan_source("import numpy as np\n", "tracking/ok.py")
