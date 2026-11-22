"""Non-vacuity canary for rf-ingress/pyproject.toml ruff rules."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _PKG_ROOT / "pyproject.toml"
_CANARY = _PKG_ROOT / "tests" / "fixtures" / "lint_canary" / "canary_violations.py"

_EXPECTED_CODES = frozenset(
    {
        "ASYNC230",
        "BLE001",
        "DTZ001",
        "ERA001",
        "G004",
        "I002",
        "PGH003",
        "S608",
        "TID252",
    }
)


def _ruff_codes(config_path: Path) -> set[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(config_path),
            "--output-format=json",
            "--stdin-filename",
            "rf_ingress/_lint_canary.py",
            "-",
        ],
        input=_CANARY.read_text(encoding="utf-8"),
        cwd=_PKG_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    findings = json.loads(result.stdout)
    return {item["code"] for item in findings}


def test_pyproject_enforces_all_guardrail_rules() -> None:
    reported = _ruff_codes(_PYPROJECT)
    missing = _EXPECTED_CODES - reported
    assert not missing, f"pyproject.toml did not report expected ruff codes: {sorted(missing)}"


def test_canary_detects_removed_rule(tmp_path: Path) -> None:
    stripped = _PYPROJECT.read_text(encoding="utf-8").replace('"S608", ', "")
    tmp_config = tmp_path / "pyproject.toml"
    tmp_config.write_text(stripped, encoding="utf-8")

    reported = _ruff_codes(tmp_config)
    assert "S608" not in reported


def test_canary_fixture_is_excluded_from_normal_lint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format=json", str(_PKG_ROOT)],
        cwd=_PKG_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
