from app.pipeline.frame_pipeline import PipelineConfig


def test_pipeline_config_no_dead_fields() -> None:
    config = PipelineConfig()
    assert not hasattr(config, "identity_committer_enabled"), (
        "identity_committer_enabled was removed as dead code"
    )
    assert not hasattr(config, "detector_confidence"), (
        "detector_confidence was moved to PersonDetector constructor; remove from PipelineConfig"
    )


def test_pipeline_config_defaults_are_sane() -> None:
    config = PipelineConfig()
    assert 0 < config.tracker_min_frames_to_publish <= 10
    assert 0 < config.identity_commit_window_s <= 60
    assert 0 < config.tracker_dedup_iou_threshold <= 1.0
