"""Settings audit guards for explicitly loaded orchestrator config."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.config import Settings
from app.main import (
    _build_face_id_config,
    _build_resolver_config,
    _build_sampler_config,
    _build_signal_config,
    _build_world_tracker_config,
)
from app.pipeline.frame_pipeline import FaceIdConfig, PipelineConfig, SignalConfig
from app.sampling.keyframe_sampler import SamplerConfig
from app.tracking.identity_resolver import ResolverConfig
from app.tracking.world.config import WorldTrackerConfig

SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"

ENV_FOR_SETTINGS = {
    "PERSON_ID_SERVICE_URL": "http://person-id.test",
}

INACTIVE_RESERVED = {
    "triton.depth_enabled",
    "triton.depth_model",
    "pipeline.posture.depth_slow_path_enabled",
    "pipeline.posture.depth_slow_path_min_interval_s",
    "pipeline.posture.depth_slow_path_max_age_s",
}


def _settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key, value in ENV_FOR_SETTINGS.items():
        monkeypatch.setenv(key, value)
    settings = Settings(SETTINGS_PATH)
    settings.reload()
    return settings


def _yaml() -> dict[str, Any]:
    with open(SETTINGS_PATH) as f:
        loaded = yaml.safe_load(f) or {}
    assert isinstance(loaded, dict)
    return loaded


def test_settings_loads_full_config(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)

    resolver = _build_resolver_config(settings)
    world_tracker = _build_world_tracker_config(settings)
    signal = _build_signal_config(settings)
    sampler = _build_sampler_config(settings)
    face_id = _build_face_id_config(settings, camera_configs={})

    assert isinstance(resolver, ResolverConfig)
    assert isinstance(world_tracker, WorldTrackerConfig)
    assert isinstance(signal, SignalConfig)
    assert isinstance(sampler, SamplerConfig)
    assert isinstance(face_id, FaceIdConfig)


def test_settings_values_match_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    data = _yaml()

    signal = _build_signal_config(settings)
    face_id = _build_face_id_config(settings, camera_configs={})
    resolver = _build_resolver_config(settings)

    assert signal.absence_threshold_minutes == data["signal"]["absence_threshold_minutes"] == 90
    assert SignalConfig().absence_threshold_minutes == 90
    assert (
        settings.as_float("pipeline.identity.high_confidence_face_threshold")
        == data["pipeline"]["identity"]["high_confidence_face_threshold"]
        == pytest.approx(0.85)
    )
    assert PipelineConfig().identity_high_confidence_face_threshold == pytest.approx(0.85)
    assert face_id.min_confidence == data["face_id"]["min_confidence"] == pytest.approx(0.6)
    assert FaceIdConfig().min_confidence == pytest.approx(0.6)
    assert resolver.cross_gt_face_propagation_threshold == pytest.approx(0.65)


def test_no_dead_settings() -> None:
    data = _yaml()

    removed_world_tracker_key = "evidence" + "_window_s"
    assert removed_world_tracker_key not in data["world_tracker"]
    assert {
        "triton.depth_enabled",
        "triton.depth_model",
        "pipeline.posture.depth_slow_path_enabled",
        "pipeline.posture.depth_slow_path_min_interval_s",
        "pipeline.posture.depth_slow_path_max_age_s",
    } == INACTIVE_RESERVED

    builder_by_section = {
        "resolver": _build_resolver_config,
        "world_tracker": _build_world_tracker_config,
        "signal": _build_signal_config,
        "sampler": _build_sampler_config,
        "face_id": _build_face_id_config,
    }
    for section, builder in builder_by_section.items():
        source = inspect.getsource(builder)
        section_data = data[section]
        assert isinstance(section_data, dict)
        for key in section_data:
            assert f'"{key}"' in source or f'"{section}.{key}"' in source

    dataclass_fields = {
        ResolverConfig: _build_resolver_config,
        WorldTrackerConfig: _build_world_tracker_config,
        SignalConfig: _build_signal_config,
        SamplerConfig: _build_sampler_config,
        FaceIdConfig: _build_face_id_config,
    }
    externally_supplied = {
        (FaceIdConfig, "camera_configs"),
        (SignalConfig, "timezone"),
    }
    for config_cls, builder in dataclass_fields.items():
        source = inspect.getsource(builder)
        for field in fields(config_cls):
            if (config_cls, field.name) in externally_supplied:
                continue
            assert f"{field.name}=" in source

    pipeline_source = inspect.getsource(PipelineConfig)
    main_source = Path("app/main.py").read_text()
    assert "min_keyframe_detection_confidence" in pipeline_source
    assert '"pipeline.min_keyframe_detection_confidence"' in main_source
