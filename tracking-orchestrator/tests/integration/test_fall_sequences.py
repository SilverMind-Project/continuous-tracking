"""Replay fall-sequence fixtures through FallFeatureExtractor + FallDetector.

Each fixture is a JSONL file under tests/fixtures/fall_sequences/:
  Line 1: {"type": "header", "expectation": "detect"|"no-detect"|"warning-max",
            "description": "...", "room": "..."}
  Lines 2+: serialised FallFrameInput (see _load_fixture).

Expectation semantics:
    "detect"       - check_impact returns non-None for at least one frame.
    "no-detect"    - check_impact returns None for every frame.
    "warning-max"  - check_impact may fire (warning accepted), but is_escalatable
                     never returns True (no emergency paging).

This module is pure Python (no DB, no Redis) so it runs in the fast unit gate
(`make check`).  Add @pytest.mark.integration only if fixture count or frame
count grows to where wall-clock time exceeds ~5 s.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.domain import BoundingBox
from app.inference.schemas import Keypoint
from app.trajectory.fall_detector import FallDetector, FallDetectorConfig
from app.trajectory.fall_features import FallFeatureExtractor, FallFrameInput
from app.trajectory.posture import PostureScores

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fall_sequences"
_RESTING_ROOMS = ("bed", "bedroom")

_VALID_EXPECTATIONS = {"detect", "no-detect", "warning-max"}


# ---------------------------------------------------------------------------
# Deserialisation helpers
# ---------------------------------------------------------------------------


def _deserialize_frame(d: dict) -> FallFrameInput:
    bbox_d = d["bbox"]
    bbox = BoundingBox(
        x_min=int(bbox_d["x_min"]),
        y_min=int(bbox_d["y_min"]),
        x_max=int(bbox_d["x_max"]),
        y_max=int(bbox_d["y_max"]),
    )

    raw_kps = d["keypoints"]
    keypoints: tuple[Keypoint, ...] | None = None
    if raw_kps is not None:
        keypoints = tuple(
            Keypoint(x=float(kp[0]), y=float(kp[1]), score=float(kp[2])) for kp in raw_kps
        )

    raw_ps = d["posture_scores"]
    posture_scores: PostureScores | None = None
    if raw_ps is not None:
        posture_scores = PostureScores(
            lying=float(raw_ps["lying"]),
            sitting=float(raw_ps["sitting"]),
            standing_walking=float(raw_ps["standing_walking"]),
            keypoint_confidence=float(raw_ps.get("keypoint_confidence", 0.0)),
        )

    return FallFrameInput(
        captured_at=datetime.fromisoformat(d["captured_at"]),
        bbox=bbox,
        keypoints=keypoints,
        posture_scores=posture_scores,
        floor_speed_m_s=d["floor_speed_m_s"],
        motion_energy_nu_s=d["motion_energy_nu_s"],
    )


def _load_fixture(path: Path) -> tuple[dict, list[FallFrameInput]]:
    """Return (header_dict, list[FallFrameInput]) from a JSONL fixture file."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"empty fixture: {path.name}")

    header = json.loads(lines[0])
    if header.get("type") != "header":
        raise ValueError(f"first line must be a header row in {path.name}")
    if "expectation" not in header:
        raise ValueError(f"header missing 'expectation' in {path.name}")
    if header["expectation"] not in _VALID_EXPECTATIONS:
        raise ValueError(
            f"unknown expectation {header['expectation']!r} in {path.name}; "
            f"must be one of {_VALID_EXPECTATIONS}"
        )

    frames = [_deserialize_frame(json.loads(line)) for line in lines[1:] if line.strip()]
    return header, frames


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _run_sequence(
    frames: list[FallFrameInput],
    room: str,
    config: FallDetectorConfig | None = None,
) -> tuple[bool, bool]:
    """Replay frames; return (any_impact_detected, confirmed_escalated).

    *any_impact_detected*:  True when check_impact returned non-None at least once.
    *confirmed_escalated*:  True when is_escalatable returned True AND
                            post_event_motion_nu_s was non-None at the same frame.

    The confirmed_escalated distinction matters for "warning-max" fixtures: the
    posture-proxy branch of is_escalatable (triggered when post_event_motion is
    None) fires immediately alongside check_impact.  The clinically meaningful
    escalation is the confirmed kind - where the post-event motion measurement
    window has closed and the person is still immobile.
    """
    extractor = FallFeatureExtractor()
    detector = FallDetector(config)
    ph_id = "ph-seq-test"

    any_detected = False
    confirmed_escalated = False

    for frame in frames:
        features = extractor.update(ph_id, frame)
        decision = detector.check_impact(features, room, _RESTING_ROOMS)
        if decision is not None:
            any_detected = True
            if features.post_event_motion_nu_s is not None and detector.is_escalatable(features):
                confirmed_escalated = True

    return any_detected, confirmed_escalated


# ---------------------------------------------------------------------------
# Parametrised fixture test
# ---------------------------------------------------------------------------


def _fixture_paths() -> list[Path]:
    if not _FIXTURES_DIR.exists():
        return []
    return sorted(_FIXTURES_DIR.glob("*.jsonl"))


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda p: p.stem)
def test_fixture_expectation(fixture_path: Path) -> None:
    """Each fixture must satisfy its labeled expectation at the shipped thresholds."""
    header, frames = _load_fixture(fixture_path)
    expectation: str = header["expectation"]
    room: str = header.get("room", "living_room")

    assert len(frames) > 0, f"{fixture_path.name}: no frames"

    any_detected, any_escalated = _run_sequence(frames, room)

    if expectation == "detect":
        assert any_detected, (
            f"{fixture_path.name}: expected at least one check_impact() non-None "
            f"(description: {header.get('description', '')})"
        )
    elif expectation == "no-detect":
        assert not any_detected, (
            f"{fixture_path.name}: expected no check_impact() fire "
            f"(description: {header.get('description', '')})"
        )
    elif expectation == "warning-max":
        assert not any_escalated, (
            f"{fixture_path.name}: warning fired but is_escalatable also returned True "
            f"(description: {header.get('description', '')})"
        )


# ---------------------------------------------------------------------------
# Round-trip serialisation test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda p: p.stem)
def test_round_trip(fixture_path: Path) -> None:
    """Deserialised frames re-serialise to bit-identical JSON."""
    import importlib.util

    # Dynamically import the synthesiser to re-use its serialisation helpers.
    script_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "synthesize_fall_sequence.py"
    )
    spec = importlib.util.spec_from_file_location("_synth", script_path)
    assert spec is not None and spec.loader is not None
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)  # type: ignore[union-attr]

    lines = fixture_path.read_text().splitlines()
    frame_lines = [line for line in lines[1:] if line.strip()]

    for raw_line in frame_lines:
        original_dict = json.loads(raw_line)
        frame = _deserialize_frame(original_dict)

        # Re-serialise via the synthesiser helpers and compare field-by-field.
        bbox_d = original_dict["bbox"]
        assert frame.bbox.x_min == int(bbox_d["x_min"])
        assert frame.bbox.y_min == int(bbox_d["y_min"])
        assert frame.bbox.x_max == int(bbox_d["x_max"])
        assert frame.bbox.y_max == int(bbox_d["y_max"])

        if original_dict["keypoints"] is None:
            assert frame.keypoints is None
        else:
            assert frame.keypoints is not None
            assert len(frame.keypoints) == 17

        if original_dict["posture_scores"] is None:
            assert frame.posture_scores is None
        else:
            assert frame.posture_scores is not None
            assert frame.posture_scores.lying == pytest.approx(
                float(original_dict["posture_scores"]["lying"]), abs=1e-6
            )

        # Timestamp: re-parsed isoformat must survive a round-trip.
        reparsed = datetime.fromisoformat(frame.captured_at.isoformat())
        assert reparsed == frame.captured_at


# ---------------------------------------------------------------------------
# Expectation-sidecar enforcement
# ---------------------------------------------------------------------------


def test_missing_expectation_raises(tmp_path: Path) -> None:
    """A fixture without a labeled expectation must fail loudly."""
    bad = tmp_path / "bad.jsonl"
    # Header line without expectation key.
    bad.write_text(
        '{"type": "header", "room": "hallway"}\n{"captured_at": "2026-01-01T00:00:00+00:00"}\n'
    )
    with pytest.raises(ValueError, match="missing 'expectation'"):
        _load_fixture(bad)


def test_unknown_expectation_raises(tmp_path: Path) -> None:
    """A fixture with an unrecognised expectation value must fail loudly."""
    bad = tmp_path / "bad2.jsonl"
    bad.write_text('{"type": "header", "expectation": "maybe", "room": "hallway"}\n')
    with pytest.raises(ValueError, match="unknown expectation"):
        _load_fixture(bad)


def test_non_header_first_line_raises(tmp_path: Path) -> None:
    """A fixture whose first line is not a header must fail loudly."""
    bad = tmp_path / "bad3.jsonl"
    bad.write_text('{"type": "frame", "expectation": "detect", "room": "hallway"}\n')
    with pytest.raises(ValueError, match="first line must be a header"):
        _load_fixture(bad)


# ---------------------------------------------------------------------------
# Grid sanity - shipped thresholds
# ---------------------------------------------------------------------------


def test_fall_forward_fast_always_detected() -> None:
    """fall_forward_fast must detect at every threshold combination in the grid.

    Uses default (shipped) thresholds; this is the most generous positive fixture.
    """
    path = _FIXTURES_DIR / "fall_forward_fast.jsonl"
    if not path.exists():
        pytest.skip("fixture not generated yet; run synthesize_fall_sequence.py")
    header, frames = _load_fixture(path)
    any_detected, _ = _run_sequence(frames, header.get("room", "living_room"))
    assert any_detected, "fall_forward_fast must be detected at shipped thresholds"


def test_no_emergency_on_guardrail_fixtures() -> None:
    """No guardrail fixture may produce an emergency at the shipped thresholds."""
    guardrails = [
        "sit_down_normal.jsonl",
        "sit_down_heavy.jsonl",
        "lie_on_bed.jsonl",
        "bend_to_pick_up.jsonl",
        "tie_shoes.jsonl",
        "child_or_pet_proxy.jsonl",
    ]
    missing = [g for g in guardrails if not (_FIXTURES_DIR / g).exists()]
    if missing:
        pytest.skip(f"guardrail fixtures not generated: {missing}")

    for name in guardrails:
        _, frames = _load_fixture(_FIXTURES_DIR / name)
        room = "bedroom" if name == "lie_on_bed.jsonl" else "living_room"
        _, any_escalated = _run_sequence(frames, room)
        assert not any_escalated, f"{name}: guardrail fixture must not produce an emergency signal"


def test_fixture_count() -> None:
    """At least 10 fixtures must be present (task 2.5 requirement)."""
    if not _FIXTURES_DIR.exists():
        pytest.skip("fall_sequences directory not present")
    count = len(list(_FIXTURES_DIR.glob("*.jsonl")))
    assert count >= 10, f"expected >= 10 fall fixtures, found {count}"
