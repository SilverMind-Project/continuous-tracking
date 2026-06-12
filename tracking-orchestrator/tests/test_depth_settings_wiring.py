"""Regression tests for depth slow-path dependency construction."""

from __future__ import annotations

from typing import NoReturn

from app.config import Settings
from app.main import _build_posture_components
from app.pipeline.frame_pipeline import PipelineDependencies
from app.trajectory.posture_strategy import RTMPosePostureStrategy


class _FailIfUsedTritonClient:
    async def is_model_ready(self, model_name: str) -> NoReturn:
        raise AssertionError(f"Triton must not be queried for disabled depth model {model_name}")


async def test_disabled_depth_slow_path_constructs_no_depth_dependency() -> None:
    settings = Settings.from_dict(
        {
            "triton": {"depth_model": "depth-anything-v2"},
            "pipeline": {
                "posture": {
                    "depth_slow_path_enabled": False,
                    "depth_slow_path_min_interval_s": 15.0,
                    "depth_slow_path_max_age_s": 60.0,
                }
            },
        }
    )

    components = await _build_posture_components(settings, _FailIfUsedTritonClient())
    dependencies = PipelineDependencies(posture_strategy=components.posture_strategy)

    assert components.depth_estimator is None
    assert isinstance(dependencies.posture_strategy, RTMPosePostureStrategy)
