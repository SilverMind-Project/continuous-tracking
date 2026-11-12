"""Replay guardrail for the "calibrated" face evidence policy: no authoritative swap, no wrong name.

Replays the two-person, resident-plus-stranger, cross-camera handoff, and uncalibrated-home
fixtures through the real ``WorldTracker`` and ``IdentityResolver`` with the production settings,
once under "legacy" and once under "calibrated". A fixture calibration artifact (a fixed monotone
map from raw similarity to calibrated confidence for the enrolled residents) stands in for the
household artifact. Each fixture also runs with every recorded face lowered to a 0.5 raw
similarity, below the legacy 0.6 floor: legacy never sees those faces, calibrated does.

These replays build ``FaceAnchor``s directly, so they exercise the resolver's weighting and the
evidence rules but not the face stage's emission policy, which the stage-level tests in
``tests/unit/pipeline/test_face_gate_outcome_policy.py`` prove. Unknown rates are printed for
the rollout record (``pytest -s``).
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.domain import FaceAnchor, FaceEvidencePolicy, Identity, WorldObservation
from app.main import _build_resolver_config, _build_world_tracker_config
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver
from app.tracking.world.tracker import WorldTracker
from tests.integration._identity_replay import (
    DecisionRow,
    EvaluationReport,
    build_records,
    decision_row_from_snapshot,
    evaluate,
)
from tests.integration._replay import load_fixture, load_truth

_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _ROOT / "config" / "settings.yaml"
_REPLAYS = _ROOT / "tests" / "fixtures" / "frame_replays"
_ROOM = {"living_room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}
_FIXTURES = (
    "two_people_one_room",
    "resident_plus_stranger",
    "cross_camera_handoff",
    "uncalibrated_two_people_home",
)
_STRANGER = "stranger"
_ENROLLED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_LEGACY_FLOOR = 0.6
_SUB_FLOOR_SIMILARITY = 0.5

# Fixture calibration artifact: (raw similarity, calibrated confidence) knots of a monotone map,
# shaped like an isotonic fit on a household holdout.
_ARTIFACT_KNOTS = (
    (0.30, 0.02),
    (0.45, 0.35),
    (0.55, 0.70),
    (0.65, 0.90),
    (0.80, 0.97),
    (1.0, 0.99),
)


def _artifact(similarity: float) -> float:
    xs, ys = zip(*_ARTIFACT_KNOTS, strict=True)
    return float(np.interp(similarity, xs, ys))


@dataclass(frozen=True)
class _ReplayResult:
    report: EvaluationReport
    wrong_name_commits: int
    stranger_named: int
    committed_frames: int


def _faces_for_policy(
    step: list[WorldObservation], policy: FaceEvidencePolicy, similarity: float | None
) -> list[WorldObservation]:
    """Apply the replay variant and what each policy would emit to one step's observations."""
    out: list[WorldObservation] = []
    for obs in step:
        fa = obs.face_anchor
        if fa is None:
            out.append(obs)
            continue
        if similarity is not None:
            fa = replace(fa, match_score=similarity, similarity=similarity)
        if policy == "legacy":
            emitted: FaceAnchor | None = fa if fa.match_score >= _LEGACY_FLOOR else None
        else:
            emitted = replace(fa, calibrated_confidence=_artifact(fa.match_score))
        out.append(replace(obs, face_anchor=emitted))
    return out


async def _replay(name: str, policy: FaceEvidencePolicy, similarity: float | None) -> _ReplayResult:
    os.environ.setdefault("PERSON_ID_SERVICE_URL", "http://person-id.test")
    settings = Settings(_SETTINGS_PATH)
    settings.reload()
    truth = load_truth(_REPLAYS / f"{name}.truth.json")
    residents = [p for p in truth["persons"] if p != _STRANGER]

    gallery = InMemoryGalleryRepository()
    for person in residents:
        await gallery.upsert_identity(
            Identity(identity_id=person, display_name=person, enrolled_at=_ENROLLED_AT)
        )
    config = replace(_build_resolver_config(settings), face_evidence_policy=policy)
    tracker = WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=_build_world_tracker_config(settings),
        identity_resolver=IdentityResolver(gallery_repo=gallery, config=config),
        gallery_repo=gallery,
    )

    detection_truth: dict[str, str] = truth["detection_truth"]
    bodies_by_ph: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[DecisionRow] = []
    committed: list[tuple[str, str]] = []
    for raw_step in load_fixture(_REPLAYS / f"{name}.bin"):
        if not raw_step:
            continue
        step = _faces_for_policy(raw_step, policy, similarity)
        result = await tracker.step(
            observations=step,
            now=max(obs.captured_at for obs in step),
            room_polygons=_ROOM,
            face_anchors=[obs.face_anchor for obs in step if obs.face_anchor is not None],
        )
        for det_id, ph_id in result.det_to_ph.items():
            if det_id in detection_truth:
                bodies_by_ph[ph_id][detection_truth[det_id]] += 1
        rows.extend(decision_row_from_snapshot(s) for s in result.snapshots)
        committed.extend(
            (d.ph_id, d.identity_id) for d in result.identity_decisions if d.identity_id
        )

    report = evaluate(await build_records(rows, InMemoryIdentityCorrectionRepository()))
    wrong = 0
    stranger_named = 0
    for ph_id, identity_id in committed:
        body = bodies_by_ph[ph_id].most_common(1)[0][0] if bodies_by_ph[ph_id] else None
        if body == _STRANGER:
            stranger_named += 1
        elif body is not None and body != identity_id:
            wrong += 1
    return _ReplayResult(report, wrong, stranger_named, len(committed))


@pytest.mark.integration
@pytest.mark.parametrize("similarity", [None, _SUB_FLOOR_SIMILARITY], ids=["recorded", "sub_floor"])
@pytest.mark.parametrize("name", _FIXTURES)
async def test_calibrated_policy_replays_without_swaps_or_wrong_names(
    name: str, similarity: float | None
) -> None:
    legacy = await _replay(name, "legacy", similarity)
    calibrated = await _replay(name, "calibrated", similarity)

    for policy, result in (("legacy", legacy), ("calibrated", calibrated)):
        print(
            f"{name}[{'recorded' if similarity is None else similarity}] {policy}: "
            f"frames={result.report.frames} unknown_rate={result.report.unknown_rate:.3f} "
            f"authoritative_frames={result.report.authoritative_frames} "
            f"phs={result.report.distinct_phs} committed={result.committed_frames}"
        )
    # Release gate: no authoritative identity swap, no resident named on the wrong body, no
    # stranger given a resident's name, and never two live PHs holding one identity.
    assert calibrated.report.swap_count == 0, calibrated.report.authoritative_swaps
    assert calibrated.wrong_name_commits == 0
    assert calibrated.stranger_named == 0
    assert calibrated.report.duplicate_active_frames == 0
    # Non-vacuity: the recorded faces map to calibrated confidences of 0.94 and above, so the
    # calibrated replay must reach ArcFace authority.
    if similarity is None:
        assert calibrated.report.authoritative_frames > 0
