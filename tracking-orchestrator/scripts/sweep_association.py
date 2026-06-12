"""Association parameter sweep driver for WorldTracker.

Replays all fixture binaries through WorldTracker.step across a Cartesian grid
of WorldTrackerConfig fields, scores each (fixture, params) combination against
the ground-truth sidecar, and emits a ranked CSV plus a markdown summary.

Usage::

    cd tracking-orchestrator
    uv run python scripts/sweep_association.py --grid scripts/grids/default.yaml --out /tmp/sweep

    # Smoke: tiny grid, one fixture, quick check
    uv run python scripts/sweep_association.py \\
        --grid scripts/grids/smoke.yaml --out /tmp/sweep-smoke

Output files:
    <out>/sweep_results.csv  -- one row per (fixture, param_combo)
    <out>/sweep_summary.md   -- ranked admissible/inadmissible table + conclusion

Dev notes:
    - Pure InMemory repos; no Docker, no Postgres, no Triton.
    - asyncio.Semaphore caps concurrent replays (default 8) to bound memory.
    - Seeded via ``seed`` key in grid YAML for full determinism.
    - Import paths assume CWD is tracking-orchestrator/ or the venv is active.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import random
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

# Allow running from the repo root or from tracking-orchestrator/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository  # noqa: E402
from app.tracking.world.config import WorldTrackerConfig  # noqa: E402
from app.tracking.world.tracker import WorldTracker  # noqa: E402
from scripts.replay_metrics import (  # noqa: E402
    FrameRecord,
    RunMetrics,
    SweepRunResult,
    aggregate_metrics,
    format_summary_table,
    score_run,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "frame_replays"
BASE_TIME = datetime(2026, 5, 28, 9, 0, 0, tzinfo=UTC)
FRAME_INTERVAL_S = 0.5


# ── Fixture loading ───────────────────────────────────────────────────────


def _load_bin(path: Path) -> list[list[dict]]:
    """Load a length-prefixed JSON fixture binary."""
    import struct

    steps: list[list[dict]] = []
    with path.open("rb") as f:
        while chunk := f.read(4):
            length = struct.unpack(">I", chunk)[0]
            data = f.read(length)
            steps.append(json.loads(data))
    return steps


def _load_truth(bin_path: Path) -> dict:
    truth_path = bin_path.with_suffix(".truth.json")
    if not truth_path.exists():
        return {"persons": [], "detection_truth": {}, "events": []}
    with truth_path.open() as f:
        return json.load(f)


def _frames_to_world_obs(raw_steps: list[list[dict]]):  # type: ignore[return]
    """Convert raw JSON steps to WorldObservation lists (avoids importing _replay)."""
    from app.domain import BoundingBox, FaceAnchor, FloorPoint, OrientationBin, WorldObservation

    steps = []
    for step_obs in raw_steps:
        frame: list[WorldObservation] = []
        for o in step_obs:
            bbox_d = o["bbox"]
            face_anchor: FaceAnchor | None = None
            fa_data = o.get("face_anchor")
            if fa_data is not None:
                from datetime import UTC

                face_anchor = FaceAnchor(
                    person_id=fa_data["person_id"],
                    confidence=float(fa_data["confidence"]),
                    quality=float(fa_data.get("quality", 1.0)),
                    detection_id=fa_data.get("detection_id", ""),
                    camera_id=fa_data.get("camera_id", ""),
                    captured_at=(
                        datetime.fromisoformat(fa_data["captured_at_iso"])
                        if fa_data.get("captured_at_iso")
                        else datetime(2026, 1, 1, tzinfo=UTC)
                    ),
                )

            ori_raw = o.get("orientation", OrientationBin.UNKNOWN)
            try:
                orientation = OrientationBin(int(ori_raw))
            except (ValueError, KeyError):
                orientation = OrientationBin.UNKNOWN

            frame.append(
                WorldObservation(
                    camera_id=o["camera_id"],
                    frame_index=o["frame_index"],
                    captured_at=datetime.fromisoformat(o["captured_at_iso"]),
                    floor_point=FloorPoint(
                        x_mm=o["floor_x_mm"],
                        y_mm=o["floor_y_mm"],
                        calibrated=o.get("calibrated", True),
                    ),
                    bbox=BoundingBox(
                        x_min=bbox_d["x_min"],
                        y_min=bbox_d["y_min"],
                        x_max=bbox_d["x_max"],
                        y_max=bbox_d["y_max"],
                    ),
                    embedding=o["embedding"],
                    detection_confidence=o["detection_confidence"],
                    detection_id=o.get("detection_id", ""),
                    quality=float(o.get("quality", 0.5)),
                    face_anchor=face_anchor,
                    orientation=orientation,
                    orientation_confidence=float(o.get("orientation_confidence", 0.0)),
                )
            )
        steps.append(frame)
    return steps


# ── Single replay ─────────────────────────────────────────────────────────

_ROOM_POLYGONS = {"living_room": [(0.0, 0.0), (25.0, 0.0), (25.0, 25.0), (0.0, 25.0)]}


async def _run_one(
    fixture_name: str,
    world_obs_steps: list,
    cfg: WorldTrackerConfig,
) -> SweepRunResult:
    """Replay one fixture through WorldTracker with the given config."""
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    run = SweepRunResult(fixture_name=fixture_name)

    for step_idx, frame_obs in enumerate(world_obs_steps):
        now = BASE_TIME + timedelta(seconds=step_idx * FRAME_INTERVAL_S)
        result = await tracker.step(
            observations=frame_obs,
            now=now,
            room_polygons=_ROOM_POLYGONS,
        )
        ph_identities = {
            ph.ph_id: ph.current_identity_id for ph in result.updated_phs if ph.closed_at is None
        }
        run.frames.append(
            FrameRecord(
                step=step_idx,
                det_to_ph=dict(result.det_to_ph),
                ph_identities=ph_identities,
            )
        )

    return run


# ── Grid expansion ────────────────────────────────────────────────────────


def _expand_grid(params: dict) -> list[dict]:
    """Expand a param dict of {name: [values]} into a Cartesian product list."""
    keys = list(params.keys())
    value_lists = [params[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combo = dict(zip(keys, values, strict=True))
        # Filter illegal weight combinations: alpha_geo + alpha_app must be <= 1.0.
        geo = float(combo.get("alpha_geo", WorldTrackerConfig.alpha_geo))
        app = float(combo.get("alpha_app", WorldTrackerConfig.alpha_app))
        if geo + app > 1.0 + 1e-9:
            continue
        combos.append(combo)
    return combos


def _combo_to_config(combo: dict) -> WorldTrackerConfig:
    """Build a WorldTrackerConfig from a param combo dict."""
    cfg = WorldTrackerConfig()
    for k, v in combo.items():
        cfg = replace(cfg, **{k: v})
    return cfg


# ── Main sweep ────────────────────────────────────────────────────────────


async def sweep(
    grid_path: Path,
    out_dir: Path,
    *,
    max_concurrent: int = 8,
) -> list[dict]:
    """Run the full sweep and return all result rows."""
    with grid_path.open() as f:
        grid_spec = yaml.safe_load(f)

    seed = int(grid_spec.get("seed", 42))
    random.seed(seed)

    param_combos = _expand_grid(grid_spec.get("params", {}))
    print(f"Grid: {len(param_combos)} param combinations")

    # Discover fixtures
    fixture_paths = sorted(FIXTURES_DIR.glob("*.bin"))
    print(f"Fixtures: {[p.stem for p in fixture_paths]}")

    # Pre-load all fixtures (pure JSON, fast)
    loaded: list[tuple[str, list, dict]] = []
    for fp in fixture_paths:
        raw = _load_bin(fp)
        world_obs = _frames_to_world_obs(raw)
        truth = _load_truth(fp)
        loaded.append((fp.stem, world_obs, truth))

    total_runs = len(param_combos) * len(loaded)
    print(f"Total replay runs: {total_runs}")

    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(max_concurrent)
    t0 = time.perf_counter()

    async def _bounded_run(
        fixture_name: str,
        world_obs: list,
        cfg: WorldTrackerConfig,
        truth: dict,
        combo: dict,
    ) -> dict:
        async with sem:
            run = await _run_one(fixture_name, world_obs, cfg)
        metrics = score_run(run, truth)
        return {
            "fixture": fixture_name,
            "params": combo,
            **aggregate_metrics([metrics]),
            **{
                f"m_{k}": getattr(metrics, k)
                for k in [
                    "identity_preservation",
                    "phantom_rate",
                    "fragmentation",
                    "identity_contamination",
                ]
            },
        }

    tasks = []
    for combo in param_combos:
        cfg = _combo_to_config(combo)
        for fixture_name, world_obs, truth in loaded:
            tasks.append(_bounded_run(fixture_name, world_obs, cfg, truth, combo))

    rows = await asyncio.gather(*tasks)
    rows = list(rows)

    elapsed = time.perf_counter() - t0
    print(f"Completed {len(rows)} runs in {elapsed:.1f}s")

    # Aggregate per param_combo across fixtures
    combo_agg: dict[str, list[dict]] = {}
    for row in rows:
        key = json.dumps(row["params"], sort_keys=True)
        combo_agg.setdefault(key, []).append(row)

    agg_rows: list[dict] = []
    for _, fixture_rows in combo_agg.items():
        combo = fixture_rows[0]["params"]
        per_fixture_metrics = []
        for r in fixture_rows:
            per_fixture_metrics.append(
                RunMetrics(
                    fixture_name=r["fixture"],
                    identity_preservation=r["m_identity_preservation"],
                    phantom_rate=r["m_phantom_rate"],
                    fragmentation=r["m_fragmentation"],
                    identity_contamination=int(r["m_identity_contamination"]),
                    person_count=0,
                )
            )
        agg = aggregate_metrics(per_fixture_metrics)
        agg_rows.append({"params": combo, **agg})

    # Write CSV (per-fixture rows)
    csv_path = out_dir / "sweep_results.csv"
    if rows:
        flat_rows = []
        for row in rows:
            flat = {"fixture": row["fixture"]}
            flat.update(row["params"])
            flat["identity_preservation"] = row["m_identity_preservation"]
            flat["phantom_rate"] = row["m_phantom_rate"]
            flat["fragmentation"] = row["m_fragmentation"]
            flat["identity_contamination"] = row["m_identity_contamination"]
            flat_rows.append(flat)

        with csv_path.open("w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"Wrote {csv_path}")

    # Write markdown summary
    summary_path = out_dir / "sweep_summary.md"
    with summary_path.open("w") as mdf:
        mdf.write("# Association Sweep Summary\n\n")
        mdf.write(f"- **Grid**: `{grid_path.name}`\n")
        mdf.write(f"- **Fixtures**: {len(loaded)}\n")
        mdf.write(f"- **Combinations**: {len(param_combos)}\n")
        mdf.write(f"- **Total runs**: {total_runs}\n")
        mdf.write(f"- **Elapsed**: {elapsed:.1f}s\n")
        mdf.write(f"- **Seed**: {seed}\n\n")
        mdf.write(format_summary_table(agg_rows))
        mdf.write("\n")
    print(f"Wrote {summary_path}")

    return agg_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="WorldTracker association parameter sweep")
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path(__file__).parent / "grids" / "default.yaml",
        help="Path to grid YAML spec",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/sweep"),
        help="Output directory for CSV + markdown",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent async replays",
    )
    args = parser.parse_args()

    asyncio.run(sweep(args.grid, args.out, max_concurrent=args.concurrency))
    return 0


if __name__ == "__main__":
    sys.exit(main())
