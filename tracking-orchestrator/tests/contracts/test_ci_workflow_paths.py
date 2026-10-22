"""Every repository path the CI workflow names must exist.

The python-test job once loaded ``tracking-orchestrator/migrations/0001_init.sql`` after the
baseline had been renamed ``0001_init.up.sql``; the job could never pass and nothing local
noticed. This test reads ``.github/workflows/ci.yml`` and checks each ``working-directory``
and each file path written in a ``run`` step.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

# A token with a directory separator and a file extension, e.g. "migrations/0001_init.up.sql".
_PATH_TOKEN = re.compile(r"(?<![\w$.])((?:\.{0,2}/)?[\w.-]+(?:/[\w.-]+)+\.[A-Za-z]{1,5})\b")

# Paths created by the job itself rather than committed (virtualenv binaries, build output).
_GENERATED_PREFIXES = (".venv/",)


def _steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for job in workflow.get("jobs", {}).values():
        steps.extend(job.get("steps", []))
    return steps


def _missing_paths(workflow_text: str) -> tuple[list[str], int]:
    """Return (missing paths, number of paths checked) for a workflow document."""
    workflow = yaml.safe_load(workflow_text)
    missing: list[str] = []
    checked = 0
    for step in _steps(workflow):
        workdir = step.get("working-directory")
        base = _REPO_ROOT / workdir if workdir else _REPO_ROOT
        if workdir:
            checked += 1
            if not base.is_dir():
                missing.append(f"working-directory: {workdir}")
        for token in _PATH_TOKEN.findall(step.get("run", "") or ""):
            if token.startswith(_GENERATED_PREFIXES):
                continue
            checked += 1
            if not ((base / token).exists() or (_REPO_ROOT / token).exists()):
                missing.append(token)
    return missing, checked


def test_ci_workflow_names_only_existing_paths() -> None:
    missing, checked = _missing_paths(_WORKFLOW.read_text())

    assert checked >= 3, f"only {checked} paths checked; the workflow parse found too little"
    assert not missing, f"ci.yml names paths that do not exist: {missing}"


def test_checker_rejects_a_renamed_migration_path() -> None:
    """Canary: the exact step that broke CI must be reported."""
    broken = """
jobs:
  python-test:
    steps:
      - name: Run migrations
        run: >-
          psql -h localhost -U cts -d continuous_tracking
          -f tracking-orchestrator/migrations/0001_init.sql
"""
    missing, _ = _missing_paths(broken)

    assert missing == ["tracking-orchestrator/migrations/0001_init.sql"]
