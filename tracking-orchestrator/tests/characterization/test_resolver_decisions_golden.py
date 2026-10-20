"""Golden characterization of identity decisions driven by face evidence.

Freezes what the world tracker and identity resolver decide, with the production configuration
from ``config/settings.yaml``, over every frame-replay fixture plus a synthetic two-person,
two-camera scenario that exercises each face-evidence path whose names a vocabulary refactor
touches (``tests/characterization/_resolver_golden.py``). A refactor must leave the golden file
byte-identical.

Known pre-existing nondeterminism, frozen rather than fixed here: when the posterior is an exact
tie (prior-only frames over two enrolled identities), ``PosteriorDist.top_identity`` breaks the
tie by dict insertion order, which follows set iteration in ``combine_posteriors`` and therefore
``PYTHONHASHSEED``. Only the provenance field ``inferred_identity_id`` of uncommitted frames is
affected; committed identities are not. The scenarios therefore run in a subprocess with
``PYTHONHASHSEED=0``.

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


def _run_scenarios(hash_seed: str = "0") -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PERSON_ID_SERVICE_URL": "http://pid.test"}
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
    first = _run_scenarios()
    assert first == _run_scenarios()
    synthetic: list[dict[str, Any]] = json.loads(first)["synthetic_face_paths"]
    identities = {
        s["identity_id"] for step in synthetic for s in step["snapshots"] if s["identity_id"]
    }
    authorities = {s["authority"] for step in synthetic for s in step["snapshots"]}
    assert identities == {"amma", "grandma"}
    assert "direct_face" in authorities
