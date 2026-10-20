"""Scenario runner for the resolver-decisions golden (see test_resolver_decisions_golden.py).

Run as ``python -m tests.characterization._resolver_golden OUT.json`` from
``tracking-orchestrator/``; it writes the scenario record as canonical JSON to ``OUT.json`` (stdout
is left to logging). The test runs it in a subprocess with a fixed ``PYTHONHASHSEED`` because
identity ties are broken by dict order today (see that test's docstring), and the golden must be
reproducible.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import Settings
from app.domain import (
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    Identity,
    OrientationBin,
    WorldFrameSnapshot,
    WorldObservation,
)
from app.main import _build_resolver_config, _build_world_tracker_config
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver
from app.tracking.world.tracker import WorldTracker, WorldTrackerResult
from tests.integration._replay import load_fixture

_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _ROOT / "config" / "settings.yaml"
_REPLAYS = _ROOT / "tests" / "fixtures" / "frame_replays"
_IDENTITIES = ("amma", "grandma")
_ROOM = {"living_room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}
_BASE = datetime(2026, 6, 23, 9, 0, 0, tzinfo=UTC)


def _round(value: float | None, places: int = 6) -> float | None:
    return None if value is None else round(float(value), places)


async def _tracker() -> WorldTracker:
    os.environ.setdefault("PERSON_ID_SERVICE_URL", "http://person-id.test")
    settings = Settings(_SETTINGS_PATH)
    settings.reload()
    gallery = InMemoryGalleryRepository()
    for identity in _IDENTITIES:
        await gallery.upsert_identity(
            Identity(identity_id=identity, display_name=identity, enrolled_at=_BASE)
        )
    resolver = IdentityResolver(gallery_repo=gallery, config=_build_resolver_config(settings))
    return WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=_build_world_tracker_config(settings),
        identity_resolver=resolver,
        gallery_repo=gallery,
    )


def _snapshot_key(snap: WorldFrameSnapshot) -> tuple[Any, ...]:
    bbox = snap.bbox
    return (
        snap.camera_id,
        snap.frame_index,
        round(snap.floor_x_m, 4),
        round(snap.floor_y_m, 4),
        (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) if bbox else (),
    )


class _Recorder:
    """Turns tracker results into a stable, PH-id-independent record."""

    def __init__(self) -> None:
        self._ph_ordinals: dict[str, int] = {}
        self.steps: list[dict[str, Any]] = []

    def _ordinal(self, ph_id: str) -> int:
        return self._ph_ordinals.setdefault(ph_id, len(self._ph_ordinals))

    def record(self, result: WorldTrackerResult) -> None:
        snapshots = sorted(result.snapshots, key=_snapshot_key)
        for snap in snapshots:
            self._ordinal(snap.ph_id)
        evidence_by_ph = {}
        for snap in snapshots:
            evidence = json.loads(snap.evidence_json or "{}")
            evidence_by_ph[self._ordinal(snap.ph_id)] = evidence
        self.steps.append(
            {
                "snapshots": [
                    {
                        "ph": self._ordinal(s.ph_id),
                        "camera_id": s.camera_id,
                        "identity_id": s.identity_id,
                        "identity_confidence": _round(s.identity_confidence),
                        "posterior_entropy": _round(s.posterior_entropy),
                        "direct_face_evidence": s.direct_face_evidence,
                        "inferred_identity_id": s.inferred_identity_id,
                        "effective_identity_id": s.effective_identity_id,
                        "authority": s.authority,
                        "decision_source": s.decision_source,
                        "conflict": s.conflict,
                        "evidence": evidence_by_ph[self._ordinal(s.ph_id)],
                    }
                    for s in snapshots
                ],
                "decisions": sorted(
                    (
                        {
                            "ph": self._ordinal(d.ph_id),
                            "identity_id": d.identity_id,
                            "authority": str(d.authority),
                            "decision_source": d.decision_source,
                            "reason": d.reason,
                            "revises_previous": d.revises_previous,
                            "evidence_backed": d.evidence_backed,
                            "posterior": {
                                k: _round(v) for k, v in sorted(d.posterior.distribution.items())
                            },
                        }
                        for d in result.identity_decisions
                        if d.ph_id in self._ph_ordinals
                    ),
                    key=lambda row: (row["ph"], str(row["identity_id"])),
                ),
                "revisions": len(result.revisions),
                "revived": len(result.revived_ph_ids),
            }
        )


async def _run_replay(path: Path) -> list[dict[str, Any]]:
    tracker = await _tracker()
    recorder = _Recorder()
    for step in load_fixture(path):
        if not step:
            continue
        now = max(obs.captured_at for obs in step)
        result = await tracker.step(
            observations=step,
            now=now,
            room_polygons=_ROOM,
            face_anchors=[obs.face_anchor for obs in step if obs.face_anchor is not None],
        )
        recorder.record(result)
    return recorder.steps


# ---------------------------------------------------------------------------
# Synthetic face-evidence scenario
# ---------------------------------------------------------------------------

_AMMA_BODY = [1.0 if i < 384 else 0.0 for i in range(768)]
_GRANDMA_BODY = [0.0 if i < 384 else 1.0 for i in range(768)]

# (person face shown, recognition_state, raw similarity, calibrated confidence, origin) per frame.
# Each row exercises a threshold the refactor renames: 0.70 commit, 0.70 contradiction, 0.70
# face-conflict gate, 0.90 duplicate-guard bypass, 0.80 calibrated authority.
_FACE_SCRIPT: list[tuple[str | None, str, float, float | None, str]] = [
    ("self", "recognized", 0.95, 0.93, "native"),
    ("self", "recognized", 0.72, None, "native"),
    ("self", "recognized", 0.65, None, "native"),
    ("self", "candidate", 0.32, None, "native"),
    (None, "unrecognized", 0.10, None, "native"),
    ("other", "recognized", 0.74, None, "native"),
    ("other", "recognized", 0.66, None, "native"),
    ("self", "recognized", 0.91, 0.85, "native"),
    ("self", "recognized", 0.80, None, "cc_assertion"),
    (None, "none", 0.0, None, "native"),
]


def _synthetic_anchor(
    person: str, other: str, det_id: str, cam: str, captured: datetime, row_index: int
) -> FaceAnchor | None:
    who, state, similarity, calibrated, origin = _FACE_SCRIPT[row_index % len(_FACE_SCRIPT)]
    if state == "none":
        return None
    person_id = "unknown" if who is None else (person if who == "self" else other)
    score = 0.88 if state == "unrecognized" else similarity
    return FaceAnchor(
        person_id=person_id,
        match_score=score,
        quality=0.9,
        detection_id=det_id,
        camera_id=cam,
        captured_at=captured,
        recognition_state=state,
        similarity=similarity,
        yaw_deg=10.0 + 5.0 * (row_index % 4),
        calibrated_confidence=calibrated,
        origin=origin,
    )


def _synthetic_obs(
    cam: str, k: int, x_m: float, body: list[float], person: str, other: str, face_row: int
) -> WorldObservation:
    captured = _BASE + timedelta(seconds=0.5 * k)
    det_id = f"{person}-{cam}-{k}"
    return WorldObservation(
        camera_id=cam,
        frame_index=k,
        captured_at=captured,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=5000, calibrated=True),
        bbox=BoundingBox(x_min=int(x_m * 100), y_min=100, x_max=int(x_m * 100) + 50, y_max=300),
        embedding=body,
        detection_confidence=0.92,
        detection_id=det_id,
        quality=0.85,
        orientation=OrientationBin.FRONT,
        orientation_confidence=0.9,
        face_anchor=_synthetic_anchor(person, other, det_id, cam, captured, face_row),
    )


async def _run_synthetic() -> list[dict[str, Any]]:
    tracker = await _tracker()
    recorder = _Recorder()
    for k in range(40):
        # Grandma leaves the view for frames 18 to 23, then re-enters (revival path).
        grandma_visible = not 18 <= k < 24
        amma_x = 5.0 + 0.1 * k
        grandma_x = 11.0 - 0.1 * k
        obs: list[WorldObservation] = []
        for cam_index, cam in enumerate(("cam-1", "cam-2")):
            obs.append(_synthetic_obs(cam, k, amma_x, _AMMA_BODY, "amma", "grandma", k + cam_index))
            if grandma_visible:
                obs.append(
                    _synthetic_obs(
                        cam, k, grandma_x, _GRANDMA_BODY, "grandma", "amma", k + 3 + cam_index
                    )
                )
        result = await tracker.step(
            observations=obs,
            now=_BASE + timedelta(seconds=0.5 * k),
            room_polygons=_ROOM,
            face_anchors=[o.face_anchor for o in obs if o.face_anchor is not None],
        )
        recorder.record(result)
    return recorder.steps


async def _all_scenarios() -> dict[str, Any]:
    scenarios: dict[str, Any] = {"synthetic_face_paths": await _run_synthetic()}
    for path in sorted(_REPLAYS.glob("*.bin")):
        scenarios[path.stem] = await _run_replay(path)
    return scenarios


def main() -> None:
    """Write the scenario record as canonical JSON to the path given as the only argument."""
    record = asyncio.run(_all_scenarios())
    Path(sys.argv[1]).write_text(json.dumps(record, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
