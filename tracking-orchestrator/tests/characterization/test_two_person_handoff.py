"""M00 sanitized production characterization for label exchange and PH handoff."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parents[1] / "fixtures/identity_integrity/two_person_handoff.json"


def _scenario(name: str) -> dict[str, object]:
    data = json.loads(_FIXTURE.read_text())
    return next(item for item in data["scenarios"] if item["name"] == name)


@pytest.mark.xfail(
    strict=True,
    reason="M03 replaces this sanitized baseline with a positive reviewed two-person replay",
)
def test_two_visible_people_keep_their_truth_labels() -> None:
    scenario = _scenario("two_visible_people_labels_exchanged")
    truth = scenario["truth"]
    actual = scenario["current_baseline"]
    assert isinstance(truth, dict)
    assert isinstance(actual, dict)
    assert {
        detection_id: observation["effective_identity_id"]
        for detection_id, observation in actual.items()
    } == truth


@pytest.mark.xfail(
    strict=True,
    reason="M03 and M06 remove this xfail with handoff continuity and correction boundaries",
)
def test_one_physical_person_has_one_ph_or_a_confirmed_handoff_boundary() -> None:
    scenario = _scenario("one_physical_person_handed_between_two_phs")
    baseline = scenario["current_baseline"]
    assert isinstance(baseline, dict)
    assert (
        baseline["ph_count"] == baseline["physical_person_count"]
        or baseline["correction_boundary_available"]
    )
