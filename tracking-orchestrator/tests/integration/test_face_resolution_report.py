"""scripts/face_resolution_report.py aggregates face_observation_stats correctly in SQL.

Marked @pytest.mark.integration: skipped by make check, included by make ci.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import asyncpg
import pytest

from app.domain import FaceGateOutcome
from app.storage.postgres.face_stats_repo import PostgresFaceObservationStatsRepository
from tests.fixtures.face_stats import T0, face_stat

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import face_resolution_report as report_script  # noqa: E402


def _row(offset_s: float, camera_id: str, interocular_px: float | None, **changes: object):
    return replace(face_stat(offset_s, camera_id), interocular_px=interocular_px, **changes)


_ROWS = [
    _row(0, "cam-1", 10.0),
    _row(1, "cam-1", 20.0),
    _row(2, "cam-1", 30.0),
    _row(3, "cam-1", 50.0),
    _row(4, "cam-1", None, recognition_state="unrecognized", best_candidate_id=None),
    _row(
        5,
        "cam-1",
        18.0,
        raw_similarity=0.52,
        gate_outcome=FaceGateOutcome.DROPPED_MIN_SIMILARITY,
    ),
    _row(6, "cam-2", 44.0, best_candidate_id="resident-beta"),
    _row(7200, "cam-1", 99.0),  # outside the window
]


@pytest.mark.integration
async def test_report_bands_percentiles_and_gate_drops(db_pool: asyncpg.Pool) -> None:
    await PostgresFaceObservationStatsRepository(db_pool).append_many(_ROWS)

    async with db_pool.acquire() as conn:
        report = await report_script.build_report(conn, T0, T0 + timedelta(hours=1), None)

    cam1 = report["cameras"]["cam-1"]
    assert cam1["faces"] == 6
    assert cam1["interocular_px"]["measured"] == 5
    assert cam1["interocular_px"]["p50"] == 20.0
    assert cam1["interocular_band_share"] == {
        "lt_16px": 0.2,
        "16_to_24px": 0.4,
        "24_to_40px": 0.2,
        "ge_40px": 0.2,
    }
    assert cam1["dropped_min_similarity"]["faces"] == 1
    assert cam1["dropped_min_similarity"]["p50"] == 0.52
    assert cam1["raw_similarity_by_state"]["unrecognized"]["faces"] == 1
    assert cam1["recognized_by_candidate"] == {"resident-alpha": 5}
    assert report["cameras"]["cam-2"]["recognized_by_candidate"] == {"resident-beta": 1}
    assert "Camera cam-1: 6 faces" in report_script.render_table(report)


@pytest.mark.integration
async def test_report_camera_filter_and_empty_window(db_pool: asyncpg.Pool) -> None:
    await PostgresFaceObservationStatsRepository(db_pool).append_many(_ROWS)

    async with db_pool.acquire() as conn:
        only_cam2 = await report_script.build_report(conn, T0, T0 + timedelta(hours=1), "cam-2")
        empty = await report_script.build_report(
            conn, T0 - timedelta(days=2), T0 - timedelta(days=1), None
        )

    assert list(only_cam2["cameras"]) == ["cam-2"]
    assert empty["cameras"] == {}
    assert "No face observations" in report_script.render_table(empty)
