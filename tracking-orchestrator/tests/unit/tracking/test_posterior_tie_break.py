"""Exact posterior ties resolve the same way in every process.

Before the tie-break was made explicit, ``PosteriorDist.top_identity`` returned whichever tied
identity came first in dict order, and ``combine_posteriors`` built that order by iterating a set,
so the winner of a tie followed ``PYTHONHASHSEED``. The subprocess test runs the same tie under
three hash seeds, as the resolver golden runner does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.domain import PosteriorDist
from app.tracking.identity.posterior import combine_posteriors

_ROOT = Path(__file__).resolve().parents[3]

# A prior-only frame over two enrolled identities: the prior is uniform, face and ReID are empty,
# so the two residents tie exactly. Identity ids are chosen so their set iteration order differs
# across hash seeds.
_TIE_SCRIPT = """
import json, sys
from app.domain import PosteriorDist
from app.tracking.identity.posterior import combine_posteriors
prior = PosteriorDist({"zeta-resident": 0.5, "alpha-resident": 0.5, "UNKNOWN": 0.05})
combined = combine_posteriors(prior, PosteriorDist({}), PosteriorDist({}))
(top_id, _), margin = combined.top_with_margin()
json.dump({"top_identity": combined.top_identity()[0], "top_with_margin": top_id,
           "margin": margin, "keys": list(combined.distribution)}, sys.stdout)
"""


def _run_tie(hash_seed: str) -> dict[str, object]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run(
        [sys.executable, "-c", _TIE_SCRIPT],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, object] = json.loads(out.stdout)
    return result


def test_exact_tie_resolves_identically_under_every_hash_seed() -> None:
    results = [_run_tie(seed) for seed in ("0", "1", "2")]

    assert results[0] == results[1] == results[2]
    assert results[0]["top_identity"] == "alpha-resident"
    assert results[0]["top_with_margin"] == "alpha-resident"
    assert results[0]["margin"] == 0.0


def test_unknown_loses_an_exact_tie() -> None:
    dist = PosteriorDist({"UNKNOWN": 0.5, "resident": 0.5})

    assert dist.top_identity()[0] == "resident"
    assert dist.top_with_margin() == (("resident", 0.5), 0.0)


def test_ranked_orders_by_probability_then_identity_with_unknown_last() -> None:
    dist = PosteriorDist({"b": 0.25, "UNKNOWN": 0.25, "a": 0.25, "c": 0.25 + 1e-9})

    assert [identity for identity, _ in dist.ranked()] == ["c", "a", "b", "UNKNOWN"]


def test_combined_keys_are_sorted_regardless_of_input_order() -> None:
    prior = PosteriorDist({"zeta": 0.4, "alpha": 0.4, "UNKNOWN": 0.2})
    face = PosteriorDist({"mid": 0.6, "UNKNOWN": 0.4})

    combined = combine_posteriors(prior, face, PosteriorDist({}))

    assert list(combined.distribution) == sorted(combined.distribution)
