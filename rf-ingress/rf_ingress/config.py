"""Pydantic v2 settings models and YAML loader with ``${ENV}`` interpolation.

Defaulted environment interpolation (``${VAR:-default}``) is intentionally
disallowed so runtime tunables stay literal in ``settings.yaml`` and required
secrets/URLs fail fast at startup when missing.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-[^}]*)?\}")


class ConfigInterpolationError(ValueError):
    """Raised when an environment variable reference cannot be resolved."""


def _interpolate_value(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_part = match.group(2)
            if default_part is not None:
                raise ConfigInterpolationError(
                    "Defaulted env interpolation is disallowed in settings.yaml: "
                    f"${{{var_name}{default_part}}}"
                )
            if var_name not in env or env[var_name] == "":
                raise ConfigInterpolationError(
                    f"Required environment variable '{var_name}' is not set"
                )
            return env[var_name]

        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_value(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(item, env) for item in value]
    return value


class ServiceSettings(BaseModel):
    """HTTP health/metrics server and runtime lifecycle settings."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8095, ge=1, le=65535)
    log_level: str = "INFO"
    capability_heartbeat_interval_s: float = Field(default=60.0, gt=0.0)


class RedisSettings(BaseModel):
    """Redis Streams connection and stream retention settings."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    fixes_stream: str = Field(default="wearable.fixes", min_length=1)
    capabilities_stream: str = Field(default="wearable.capabilities", min_length=1)
    retention_hours: float = Field(default=72.0, gt=0.0, le=72.0)


class SiteConfigSettings(BaseModel):
    """Cognitive Companion site-config polling settings."""

    model_config = ConfigDict(extra="forbid")

    cc_api_url: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    poll_interval_s: float = Field(default=60.0, gt=0.0)
    request_timeout_s: float = Field(default=10.0, gt=0.0)


class SolverSettings(BaseModel):
    """Epoch builder and multilateration solver parameters."""

    model_config = ConfigDict(extra="forbid")

    epoch_window_s: float = Field(default=0.25, gt=0.0)
    # Background expiry tick; null means min(0.05, epoch_window_s / 4).
    epoch_flush_interval_s: float | None = Field(default=None, gt=0.0, le=1.0)
    assumed_tag_height_m: float = Field(default=1.0, ge=0.0, le=3.5)
    tag_height_sigma_m: float = Field(default=0.3, ge=0.0, le=2.0)
    min_vertical_spread_m: float = Field(default=0.5, ge=0.0)
    max_condition: float = Field(default=1000.0, gt=1.0)
    quality_scale_m: float = Field(default=0.5, gt=0.0)
    max_sigma_m: float | None = Field(default=None, gt=0.0)
    min_los_confidence: float = Field(default=0.15, gt=0.0, le=1.0)


class ClockSettings(BaseModel):
    """Clock-skew monitoring and burst clock-offset estimation settings."""

    model_config = ConfigDict(extra="forbid")

    skew_window_s: float = Field(default=60.0, gt=0.0)
    skew_warn_threshold_s: float = Field(default=0.5, gt=0.0)
    burst_window_size: int = Field(default=20, ge=1, le=500)
    burst_max_sample_age_s: float = Field(default=1800.0, gt=0.0)
    burst_step_reset_threshold_s: float = Field(default=0.75, gt=0.0)


class AdapterConfig(BaseModel):
    """Base adapter configuration block validated before registry dispatch.

    Adapter-specific keys are allowed here and validated strictly by each
    adapter's own Pydantic config model inside its module.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=64)
    payload_profile: str | None = None
    ntp_synced: bool = True
    latency_s: float = Field(default=0.0, ge=0.0)

    def extra_fields(self) -> dict[str, Any]:
        """Return adapter-specific configuration fields beyond the base keys."""
        return dict(self.model_extra or {})


class RfIngressSettings(BaseModel):
    """Top-level rf-ingress configuration."""

    model_config = ConfigDict(extra="forbid")

    service: ServiceSettings = Field(default_factory=ServiceSettings)
    redis: RedisSettings
    site_config: SiteConfigSettings
    solver: SolverSettings = Field(default_factory=SolverSettings)
    clock: ClockSettings = Field(default_factory=ClockSettings)
    adapters: list[AdapterConfig] = Field(default_factory=list)

    @field_validator("adapters")
    @classmethod
    def _unique_adapter_ids(cls, adapters: list[AdapterConfig]) -> list[AdapterConfig]:
        seen: set[str] = set()
        for adapter in adapters:
            if adapter.id in seen:
                raise ValueError(f"Duplicate adapter id in configuration: {adapter.id!r}")
            seen.add(adapter.id)
        return adapters


def load_settings(
    path: Path | str,
    *,
    env: Mapping[str, str] | None = None,
) -> RfIngressSettings:
    """Load, interpolate, and validate an ``rf-ingress`` YAML settings file."""
    raw_text = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Settings file {path} must contain a YAML mapping")
    resolved = _interpolate_value(parsed, os.environ if env is None else env)
    return RfIngressSettings.model_validate(resolved)
