"""Invariant 3: DepthEstimator is implemented but intentionally not wired
into the live frame pipeline. It is used only during homography auto-calibration.
M7 will wire it into the posture slow-path."""

from __future__ import annotations

import inspect

from app.pipeline.frame_pipeline import FrameProcessingPipeline


def test_depth_estimator_not_in_frame_pipeline_init_signature() -> None:
    sig = inspect.signature(FrameProcessingPipeline.__init__)
    params = set(sig.parameters.keys())
    assert "depth_estimator" not in params, (
        f"depth_estimator found in FrameProcessingPipeline.__init__ params: {params}"
    )
    assert "depth" not in params, (
        f"'depth' found in FrameProcessingPipeline.__init__ params: {params}"
    )
