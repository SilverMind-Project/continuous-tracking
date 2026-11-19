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
    _row(2, "cam-1", 30.0, quality_score=0.15),
    _row(3, "cam-1", 50.0, quality_score=0.8, calibrated_confidence=0.9),
    _row(4, "cam-1", None, recognition_state="unrecognized", best_candidate_id=None),
    _row(
        5,
        "cam-1",
        18.0,
        raw_similarity=0.52,
        gate_outcome=FaceGateOutcome.DROPPED_MIN_SIMILARITY,
    ),
    _row(
        6,
        "cam-2",
        44.0,
        best_candidate_id="resident-beta",
        quality_score=None,
        quality_version=None,
    ),
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
    # Fixture default quality is 0.42; one row each in the lowest and highest bands.
    assert cam1["quality_bands"] == {
        "lt_0.2": {"faces": 1, "interocular_p50": 30.0, "raw_similarity_p50": 0.47},
        "0.4_to_0.6": {"faces": 4, "interocular_p50": 18.0, "raw_similarity_p50": 0.47},
        "ge_0.6": {"faces": 1, "interocular_p50": 50.0, "raw_similarity_p50": 0.47},
    }
    assert report["cameras"]["cam-2"]["quality_bands"] == {}
    # Recognized faces with a measured interocular distance, per band (the 18 px face dropped by
    # the similarity floor is still recognized); only the 50 px face carries 0.9.
    assert cam1["calibrated_recognition_by_band"] == {
        "lt_16px": {"recognized": 1, "share_calibrated_ge_0.8": 0.0},
        "16_to_24px": {"recognized": 2, "share_calibrated_ge_0.8": 0.0},
        "24_to_40px": {"recognized": 1, "share_calibrated_ge_0.8": 0.0},
        "ge_40px": {"recognized": 1, "share_calibrated_ge_0.8": 1.0},
    }
    assert report["cameras"]["cam-2"]["recognized_by_candidate"] == {"resident-beta": 1}
    table = report_script.render_table(report)
    assert "Camera cam-1: 6 faces" in table
    assert "lt_0.2 n=1" in table


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


@pytest.mark.integration
async def test_report_counts_faces_only_the_calibrated_policy_would_emit(
    db_pool: asyncpg.Pool,
) -> None:
    shadow_rows = [
        _row(
            10, "cam-3", 20.0, raw_similarity=0.44, gate_outcome=FaceGateOutcome.EMITTED_SHADOW_ONLY
        ),
        _row(
            11, "cam-3", 22.0, raw_similarity=0.56, gate_outcome=FaceGateOutcome.EMITTED_SHADOW_ONLY
        ),
        _row(12, "cam-3", 30.0),
    ]
    await PostgresFaceObservationStatsRepository(db_pool).append_many(shadow_rows)

    async with db_pool.acquire() as conn:
        report = await report_script.build_report(conn, T0, T0 + timedelta(hours=1), "cam-3")

    cam3 = report["cameras"]["cam-3"]
    assert cam3["emitted_shadow_only"]["faces"] == 2
    assert cam3["emitted_shadow_only"]["p50"] == 0.5
    assert cam3["dropped_min_similarity"] == {"faces": 0}
    assert "recognized faces only the calibrated policy would emit: 2" in (
        report_script.render_table(report)
    )


@pytest.mark.integration
async def test_report_aggregates_recognized_share_by_pitch_band(db_pool: asyncpg.Pool) -> None:
    pitch_rows = [
        _row(
            20,
            "cam-pitch",
            26.0,
            pitch_deg=-25.0,
            recognition_state="recognized",
            raw_similarity=0.55,
        ),
        _row(
            21,
            "cam-pitch",
            26.0,
            pitch_deg=-22.0,
            recognition_state="candidate",
            raw_similarity=0.32,
        ),
        _row(
            22,
            "cam-pitch",
            26.0,
            pitch_deg=-10.0,
            recognition_state="recognized",
            raw_similarity=0.60,
        ),
        _row(
            23,
            "cam-pitch",
            26.0,
            pitch_deg=12.0,
            recognition_state="recognized",
            raw_similarity=0.58,
        ),
        _row(
            24,
            "cam-pitch",
            26.0,
            pitch_deg=24.0,
            recognition_state="unrecognized",
            raw_similarity=0.18,
        ),
        _row(
            25,
            "cam-pitch",
            26.0,
            pitch_deg=28.0,
            recognition_state="recognized",
            raw_similarity=0.52,
        ),
    ]
    await PostgresFaceObservationStatsRepository(db_pool).append_many(pitch_rows)

    async with db_pool.acquire() as conn:
        report = await report_script.build_report(conn, T0, T0 + timedelta(hours=1), "cam-pitch")

    bands = report["cameras"]["cam-pitch"]["pitch_bands"]
    assert list(bands.keys()) == ["lt_-20deg", "-20_to_0deg", "0_to_20deg", "ge_20deg"]
    assert bands["lt_-20deg"]["faces"] == 2
    assert bands["lt_-20deg"]["recognized"] == 1
    assert bands["lt_-20deg"]["recognized_share"] == 0.5
    assert bands["-20_to_0deg"]["recognized_share"] == 1.0
    assert bands["0_to_20deg"]["recognized_share"] == 1.0
    assert bands["ge_20deg"]["faces"] == 2
    assert bands["ge_20deg"]["recognized"] == 1
    assert bands["ge_20deg"]["recognized_share"] == 0.5
    table = report_script.render_table(report)
    assert "pitch bands:" in table
    assert "ge_20deg n=2 recognized=1 (0.5)" in table
