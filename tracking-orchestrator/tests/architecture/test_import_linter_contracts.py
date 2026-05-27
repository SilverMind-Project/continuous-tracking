"""Import-linter contract tests.

Runs ``lint-imports`` as a subprocess and asserts zero violations.
The CLI runs the same contracts declared in pyproject.toml.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_lint_imports_has_zero_violations() -> None:
    # tests/architecture/ -> tests/ -> tracking-orchestrator/
    project_root = Path(__file__).resolve().parent.parent.parent
    lint_imports = project_root / ".venv/bin/lint-imports"
    result = subprocess.run(
        [str(lint_imports)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(project_root),
    )
    assert result.returncode == 0, (
        f"lint-imports failed (exit {result.returncode}):\n{result.stderr}"
    )
    assert "0 broken" in result.stdout, f"lint-imports reported broken contracts:\n{result.stdout}"
