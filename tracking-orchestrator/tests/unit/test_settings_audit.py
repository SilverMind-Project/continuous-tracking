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
from app.pipeline.frame_pipeline import FaceIdConfig, PipelineConfig
from app.sampling.keyframe_sampler import SamplerConfig
from app.tracking.identity_resolver import ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.trajectory.dementia_signals import SignalConfig

SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"

ENV_FOR_SETTINGS = {
    "PERSON_ID_SERVICE_URL": "http://person-id.test",
}

INACTIVE_RESERVED = {
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
    world_tracker = _build_world_tracker_config(settings)

    assert signal.absence_threshold_minutes == data["signal"]["absence_threshold_minutes"] == 90
    assert SignalConfig().absence_threshold_minutes == 90
    assert signal.min_baseline_n == data["signal"]["min_baseline_n"] == 5
    assert signal.cooldown_minutes == data["signal"]["cooldown_minutes"] == 60
    assert signal.onset_consecutive_windows == data["signal"]["onset_consecutive_windows"] == 2
    assert signal.resting_rooms == tuple(data["signal"]["resting_rooms"])
    assert signal.sundowning_z_threshold == pytest.approx(data["signal"]["sundowning_z_threshold"])
    assert signal.bathroom_z_threshold == pytest.approx(data["signal"]["bathroom_z_threshold"])
    assert signal.bathroom_z_threshold_night == pytest.approx(
        data["signal"]["bathroom_z_threshold_night"]
    )
    assert signal.pacing_min_obs_density == pytest.approx(data["signal"]["pacing_min_obs_density"])
    assert face_id.min_confidence == data["face_id"]["min_confidence"] == pytest.approx(0.6)
    assert FaceIdConfig().min_confidence == pytest.approx(0.6)
    assert resolver.cross_gt_face_propagation_threshold == pytest.approx(0.72)
    assert world_tracker.dedup_residual_coeff_k == pytest.approx(1.0)
    assert world_tracker.dedup_max_distance_ceiling_m == pytest.approx(1.5)
    assert world_tracker.zupt_speed_enter_m_s == pytest.approx(0.12)
    assert world_tracker.zupt_speed_exit_m_s == pytest.approx(0.20)
    assert world_tracker.zupt_consecutive_frames == 5
    assert world_tracker.zupt_velocity_sigma_m_s == pytest.approx(0.05)


def test_signal_config_defaults_match_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    loaded = _build_signal_config(settings)
    defaults = SignalConfig()

    for field in fields(SignalConfig):
        assert getattr(defaults, field.name) == getattr(loaded, field.name), field.name


def test_no_dead_settings() -> None:
    data = _yaml()

    removed_world_tracker_key = "evidence" + "_window_s"
    assert removed_world_tracker_key not in data["world_tracker"]
    assert {
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
            if section == "signal" and key in {"enabled", "interval_s"}:
                continue
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
        (SignalConfig, "tz_name"),
        # sampling gate, default-off, no settings.yaml key.
        (ResolverConfig, "coherence_shadow_sample_rate"),
    }
    for config_cls, builder in dataclass_fields.items():
        source = inspect.getsource(builder)
        for field in fields(config_cls):
            if (config_cls, field.name) in externally_supplied:
                continue
            assert f"{field.name}=" in source

    signal_source = inspect.getsource(_build_signal_config)
    for key in data["signal.agitation"]:
        assert f'"{key}"' in signal_source

    pipeline_source = inspect.getsource(PipelineConfig)
    main_source = Path("app/main.py").read_text()
    assert "min_keyframe_detection_confidence" in pipeline_source
    assert '"pipeline.min_keyframe_detection_confidence"' in main_source
    assert '"signal.enabled"' in main_source
    assert '"signal.interval_s"' in main_source
