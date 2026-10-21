"""The stage trusts calibrated confidence only from a single-frame calibration on its scorer.

Anything else (a template profile, another face-quality scorer version, no quality at all) drops
the calibrated confidence on the anchor and on the measurement row alike.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.inference.face_id_client import FaceResult
from tests.pipeline.test_face_identity_quality import _READING, _T, _face, _run


def _calibrated(**changes: object) -> FaceResult:
    face = replace(
        _face("recognized", _READING),
        calibrated_confidence=0.88,
        calibration_status="ready",
        calibration_profile="single_frame",
    )
    return replace(face, **changes)  # type: ignore[arg-type]


async def _calibrated_values(face: FaceResult) -> tuple[float | None, float | None]:
    ctx, repo = await _run(face)
    [anchor] = ctx.face_anchors
    [row] = await repo.list_between(_T, _T.replace(minute=1))
    return anchor.calibrated_confidence, row.calibrated_confidence


@pytest.mark.asyncio
async def test_single_frame_calibration_on_the_expected_scorer_is_trusted() -> None:
    assert await _calibrated_values(_calibrated()) == (0.88, 0.88)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"calibration_profile": "template"},
        {"calibration_profile": ""},
        {"quality": replace(_READING, version="heuristic-v2")},
        {"quality": None},
        {"calibration_status": "degraded_incompatible"},
    ],
    ids=["template_profile", "no_profile", "other_scorer", "unscored", "degraded"],
)
async def test_any_other_calibration_is_dropped_everywhere(changes: dict[str, object]) -> None:
    assert await _calibrated_values(_calibrated(**changes)) == (None, None)
