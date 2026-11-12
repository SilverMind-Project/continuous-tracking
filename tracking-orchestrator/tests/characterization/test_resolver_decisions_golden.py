"""Golden characterization of identity decisions driven by face evidence.

Freezes what the world tracker and identity resolver decide, with the production configuration
from ``config/settings.yaml``, over every frame-replay fixture plus a synthetic two-person,
two-camera scenario that exercises each face-evidence path whose names a vocabulary refactor
touches (``tests/characterization/_resolver_golden.py``). A refactor must leave the golden file
byte-identical.

Exact posterior ties (prior-only frames over two enrolled identities) are broken by identity id
(``PosteriorDist.ranked``), so the record does not depend on ``PYTHONHASHSEED``; the
reproducibility test runs the scenarios under two different hash seeds to keep it that way. The
scenarios run in a subprocess so each run starts from a fresh interpreter.

Regenerate only for an intended behavior change:
``CTS_REGENERATE_GOLDEN=1 pytest tests/characterization/test_resolver_decisions_golden.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN = _ROOT / "tests" / "fixtures" / "characterization" / "resolver_decisions_golden.json"


def _run_scenarios(hash_seed: str | None = None) -> str:
    env = {**os.environ, "PERSON_ID_SERVICE_URL": "http://pid.test"}
    env.pop("PYTHONHASHSEED", None)
    if hash_seed is not None:
        env["PYTHONHASHSEED"] = hash_seed
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "record.json"
        subprocess.run(
            [sys.executable, "-m", "tests.characterization._resolver_golden", str(out)],
            cwd=_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return out.read_text()


def test_resolver_decisions_match_golden() -> None:
    actual = _run_scenarios()
    if os.environ.get("CTS_REGENERATE_GOLDEN") == "1":
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(actual + "\n")
    assert actual == _GOLDEN.read_text().rstrip("\n"), (
        "resolver decisions changed; a refactor must not change them"
    )


def test_golden_is_reproducible_and_not_vacuous() -> None:
    first = _run_scenarios(hash_seed="1")
    assert first == _run_scenarios(hash_seed="2")
    synthetic: list[dict[str, Any]] = json.loads(first)["synthetic_face_paths"]
    identities = {
        s["identity_id"] for step in synthetic for s in step["snapshots"] if s["identity_id"]
    }
    authorities = {s["authority"] for step in synthetic for s in step["snapshots"]}
    assert identities == {"amma", "grandma"}
    assert "direct_face" in authorities
