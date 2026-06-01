from app.pipeline.frame_pipeline import PipelineConfig


def test_pipeline_config_no_dead_fields() -> None:
    config = PipelineConfig()
    assert not hasattr(config, "identity_committer_enabled"), (
        "identity_committer_enabled was removed as dead code"
    )
    assert not hasattr(config, "detector_confidence"), (
        "detector_confidence was moved to PersonDetector constructor"
    )
    # removed as unused (resolver/world-tracker own their thresholds).
    assert not hasattr(config, "tracker_min_frames_to_publish")
    assert not hasattr(config, "tracker_dedup_iou_threshold")
    assert not hasattr(config, "identity_commit_window_s")
    assert not hasattr(config, "identity_high_confidence_face_threshold")
    assert not hasattr(config, "gallery_identity_backfill_delay_s")


def test_pipeline_config_defaults_are_sane() -> None:
    config = PipelineConfig()
    assert 0 < config.detection_iou_dedup_threshold <= 1.0
    assert config.shutdown_timeout >= 1.0
    assert config.max_concurrent_frames >= 1
