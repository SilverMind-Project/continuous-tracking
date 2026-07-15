"""M00 characterization for mismatched face and gallery seed labels.

M04 delivered the fix: the confirmed baseline bug was that gallery seeding
never checked ``face_anchor.person_id == identity_id``. That path
(``WorldTracker._seed_multiview_gallery``) is deleted; creation now runs
through ``evaluate_candidate``'s unconditional ``identity_mismatch`` gate.
This is no longer an xfail -- it is a strict positive assertion that the
scenario from the fixture produces no candidate.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.domain import FaceAnchor
from app.tracking.identity.candidate_eligibility import CandidatePolicy, evaluate_candidate
from app.tracking.orientation import OrientationBin

_FIXTURE = (
    Path(__file__).parents[1] / "fixtures/identity_integrity/gallery_seed_identity_mismatch.json"
)


def test_recognized_face_must_match_gallery_seed_identity() -> None:
    data = json.loads(_FIXTURE.read_text())
    datetime.fromisoformat(data["captured_at"])  # sanity: fixture timestamp parses

    face_anchor = FaceAnchor(
        person_id=data["direct_face_identity_id"],
        confidence=0.95,
        recognition_state=data["recognition_state"],
        calibrated_confidence=0.95,
    )

    result = evaluate_candidate(
        committed_identity_id=data["resolved_identity_id"],
        face_anchor=face_anchor,
        embedding=data["body_embedding"],
        quality=data["quality"],
        orientation=OrientationBin[data["orientation"]],
        orientation_confidence=data["orientation_confidence"],
        cfg=CandidatePolicy(),
    )

    assert result.eligible is False
    assert result.reason == "identity_mismatch"
