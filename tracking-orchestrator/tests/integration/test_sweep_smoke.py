"""Sweep smoke test and determinism proof.

T1: 2-point grid on 1 fixture completes and emits well-formed CSV (< 30 s).
T2: Same grid + fixtures twice produces identical CSV (seeded; any nondeterminism
    in the tracker is a finding worth filing as a separate issue).

Marked not-integration so the default `make check` runs them.  If they become
slow (> 15 s) reclassify with @pytest.mark.integration.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pytest

# Allow importing scripts/ from within tests/
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "frame_replays"


def _smoke_fixture_path() -> Path:
    """Return the path to the one fixture used in the smoke test."""
    p = FIXTURES_DIR / "two_rooms_two_people.bin"
    if not p.exists():
        pytest.skip("Fixture missing; run scripts/synthesize_replay_fixture.py first")
    return p


@pytest.mark.asyncio
async def test_sweep_smoke_completes_and_csv_well_formed(tmp_path: Path) -> None:
    """T1: 2-point grid on 1 fixture emits well-formed CSV in < 30 s."""
    from sweep_association import sweep

    _smoke_fixture_path()  # skip if missing

    # Inline a tiny 2-point smoke grid (no YAML file dependency for speed).
    import yaml

    smoke_spec = {
        "seed": 42,
        "params": {
            "gate_chi2": [5.99, 9.21],
            "observation_noise_m": [0.15, 0.25],
        },
    }
    grid_file = tmp_path / "smoke.yaml"
    grid_file.write_text(yaml.dump(smoke_spec))

    t0 = time.perf_counter()
    await sweep(grid_file, tmp_path / "out")
    elapsed = time.perf_counter() - t0

    assert elapsed < 30.0, f"Smoke sweep took {elapsed:.1f}s; must complete in < 30 s"

    csv_path = tmp_path / "out" / "sweep_results.csv"
    assert csv_path.exists(), "sweep_results.csv not written"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0, "CSV is empty"
    required_cols = {
        "fixture",
        "gate_chi2",
        "observation_noise_m",
        "identity_preservation",
        "phantom_rate",
        "fragmentation",
        "identity_contamination",
    }
    assert required_cols <= set(rows[0].keys()), (
        f"CSV missing columns: {required_cols - set(rows[0].keys())}"
    )

    # Spot-check: all numeric fields are parseable floats or ints
    for row in rows:
        float(row["identity_preservation"])
        float(row["phantom_rate"])
        float(row["fragmentation"])
        int(float(row["identity_contamination"]))


@pytest.mark.asyncio
async def test_sweep_determinism(tmp_path: Path) -> None:
    """T2: Same seed + fixtures produces identical CSV twice."""
    import yaml

    _smoke_fixture_path()

    smoke_spec = {
        "seed": 42,
        "params": {
            "gate_chi2": [5.99, 9.21],
        },
    }
    grid_file = tmp_path / "det_smoke.yaml"
    grid_file.write_text(yaml.dump(smoke_spec))

    from sweep_association import sweep

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    await sweep(grid_file, out1)
    await sweep(grid_file, out2)

    csv1 = (out1 / "sweep_results.csv").read_text()
    csv2 = (out2 / "sweep_results.csv").read_text()

    assert csv1 == csv2, (
        "Sweep is non-deterministic: two runs with the same seed produced "
        "different CSVs.  This is itself a finding worth filing.\n"
        f"Run 1:\n{csv1[:500]}\n\nRun 2:\n{csv2[:500]}"
    )


@pytest.mark.asyncio
async def test_loader_round_trip_with_orientation(tmp_path: Path) -> None:
    """T3: Loader preserves orientation and orientation_confidence fields."""
    import json
    import struct

    from app.domain import OrientationBin

    # Write a minimal fixture with orientation set to FRONT (0) and confidence 0.85
    obs = {
        "camera_id": "cam-test",
        "frame_index": 0,
        "captured_at_iso": "2026-05-28T09:00:00+00:00",
        "floor_x_mm": 3000,
        "floor_y_mm": 5000,
        "embedding": [1.0, 0.0, 0.0],
        "detection_confidence": 0.9,
        "bbox": {"x_min": 100, "y_min": 100, "x_max": 300, "y_max": 400},
        "detection_id": "det-rt-0",
        "quality": 0.5,
        "calibrated": True,
        "orientation": int(OrientationBin.FRONT),
        "orientation_confidence": 0.85,
    }
    bin_path = tmp_path / "rt_test.bin"
    with bin_path.open("wb") as f:
        data = json.dumps([obs]).encode()
        f.write(struct.pack(">I", len(data)))
        f.write(data)

    from tests.integration._replay import load_fixture

    steps = load_fixture(bin_path)
    assert len(steps) == 1
    assert len(steps[0]) == 1
    loaded_obs = steps[0][0]
    assert loaded_obs.orientation == OrientationBin.FRONT
    assert loaded_obs.orientation_confidence == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_loader_legacy_fixture_defaults(tmp_path: Path) -> None:
    """T4: Legacy fixture without orientation/calibrated fields loads with defaults."""
    import json
    import struct

    from app.domain import OrientationBin

    # Legacy format: no orientation, no calibrated
    obs = {
        "camera_id": "cam-legacy",
        "frame_index": 0,
        "captured_at_iso": "2026-05-28T09:00:00+00:00",
        "floor_x_mm": 1000,
        "floor_y_mm": 2000,
        "embedding": [0.5, 0.5, 0.0],
        "detection_confidence": 0.8,
        "bbox": {"x_min": 50, "y_min": 50, "x_max": 200, "y_max": 300},
        "detection_id": "det-legacy-0",
        "quality": 0.4,
    }
    bin_path = tmp_path / "legacy.bin"
    with bin_path.open("wb") as f:
        data = json.dumps([obs]).encode()
        f.write(struct.pack(">I", len(data)))
        f.write(data)

    from tests.integration._replay import load_fixture

    steps = load_fixture(bin_path)
    loaded = steps[0][0]
    # calibrated defaults to True for backward compatibility
    assert loaded.floor_point.calibrated is True
    # orientation defaults to UNKNOWN
    assert loaded.orientation == OrientationBin.UNKNOWN
    assert loaded.orientation_confidence == pytest.approx(0.0)
