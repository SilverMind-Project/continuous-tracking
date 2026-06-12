"""Configuration loader for the tracking orchestrator.

Reads a single YAML file with explicit ``${ENV_VAR}`` interpolation for
deployment-provided values, exposed as a dot-notation dict. Defaults live as
literal values in ``settings.yaml`` so tunables do not become hidden env-var
override surfaces.

Usage::

    from app.config import settings

    redis_url = settings.as_str("redis.url")
    timeout = settings.as_int("redis.ack_ttl_seconds")

The config file is resolved from ``ORCHESTRATOR_CONFIG_PATH``, or defaults
to ``config/settings.yaml`` relative to the project root.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

__all__ = ["SettingNotFoundError", "Settings", "SettingsSection", "settings"]

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
_CONFIG_PATH = os.environ.get("ORCHESTRATOR_CONFIG_PATH", str(_DEFAULT_CONFIG))
_MISSING = object()


class SettingNotFoundError(KeyError):
    """Raised when a required config value is missing."""

    def __init__(self, dotted_key: str) -> None:
        super().__init__(f"Required setting not found: {dotted_key}")
        self.dotted_key = dotted_key


def _as_int(value: object, key: str) -> int:
    try:
        return int(value)  # type: ignore[call-overload,no-any-return]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Setting {key} must be an integer, got {value!r}") from exc


def _as_float(value: object, key: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Setting {key} must be a float, got {value!r}") from exc


def _as_bool(value: object, key: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Setting {key} must be a boolean, got {value!r}")


class SettingsSection:
    """Typed access to a required mapping-valued config section."""

    def __init__(self, prefix: str, data: Mapping[str, Any], env: Mapping[str, str]) -> None:
        self._prefix = prefix
        self._data = data
        self._env = env

    def require(self, key: str) -> Any:
        if key not in self._data:
            raise SettingNotFoundError(self._qualify(key))
        return _interpolate(self._data[key], self._env)

    def as_str(self, key: str) -> str:
        return str(self.require(key))

    def as_int(self, key: str) -> int:
        return _as_int(self.require(key), self._qualify(key))

    def as_float(self, key: str) -> float:
        return _as_float(self.require(key), self._qualify(key))

    def as_bool(self, key: str) -> bool:
        return _as_bool(self.require(key), self._qualify(key))

    def _qualify(self, key: str) -> str:
        return f"{self._prefix}.{key}"


def _resolve_string(value: str, env: Mapping[str, str]) -> str:
    """Replace explicit ``${VAR}`` placeholders.

    Missing or empty variables are errors. Optional integrations should use a
    literal empty string in settings.yaml rather than an env fallback.
    """
    result: list[str] = []
    i = 0
    while i < len(value):
        if value[i : i + 2] == "${":
            depth = 1
            j = i + 2
            while j < len(value) and depth > 0:
                if value[j : j + 2] == "${":
                    depth += 1
                    j += 2
                    continue
                if value[j] == "}":
                    depth -= 1
                j += 1
            content = value[i + 2 : j - 1]
            if ":-" in content:
                raise ValueError(
                    "Env fallback interpolation is disabled; use a literal "
                    f"settings.yaml value instead: {content!r}"
                )
            val = env.get(content, "")
            if not val:
                raise ValueError(f"Required environment variable is not set: {content}")
            result.append(val)
            i = j
        else:
            result.append(value[i])
            i += 1
    return "".join(result)


def _interpolate(value: Any, env: Mapping[str, str]) -> Any:
    """Recursively replace explicit ``${VAR}`` placeholders."""
    if isinstance(value, str):
        return _resolve_string(value, env)
    if isinstance(value, dict):
        return {k: _interpolate(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, env) for v in value]
    return value


class Settings:
    """YAML-backed settings with dot-notation access and explicit env interpolation.

    Tests can bypass the filesystem with :meth:`from_dict`::

        s = Settings.from_dict({"redis": {"url": "redis://localhost:6379/0"}})
        assert s.as_str("redis.url") == "redis://localhost:6379/0"
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path: Path = Path(path) if path else Path(_CONFIG_PATH)
        self._env: Mapping[str, str] = os.environ
        self._data: dict[str, Any] = {}
        self._loaded: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Settings:
        inst = cls.__new__(cls)
        inst._path = Path(_CONFIG_PATH)
        inst._env = os.environ
        inst._data = dict(data)
        inst._loaded = True
        return inst

    def reload(self, path: Path | str | None = None) -> None:
        """(Re-)load config from *path* (or the current one)."""
        if path is not None:
            self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        with open(self._path) as f:
            raw = yaml.safe_load(f) or {}
        self._data = raw
        self._loaded = True

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a nested value using dot notation.

        Returns *default* if any segment is missing or traverses through
        a non-dict value.
        """
        value = self._lookup(dotted_key, _MISSING)
        if value is _MISSING:
            return default
        return _interpolate(value, self._env)

    def require(self, dotted_key: str) -> Any:
        """Retrieve a required nested value using dot notation.

        Missing keys raise :class:`SettingNotFoundError`. Empty strings are
        valid only when written literally in ``settings.yaml``.
        """
        value = self._lookup(dotted_key, _MISSING)
        if value is _MISSING:
            raise SettingNotFoundError(dotted_key)
        return _interpolate(value, self._env)

    def as_str(self, dotted_key: str) -> str:
        return str(self.require(dotted_key))

    def as_int(self, dotted_key: str) -> int:
        return _as_int(self.require(dotted_key), dotted_key)

    def as_float(self, dotted_key: str) -> float:
        return _as_float(self.require(dotted_key), dotted_key)

    def as_bool(self, dotted_key: str) -> bool:
        return _as_bool(self.require(dotted_key), dotted_key)

    def section(self, dotted_key: str) -> SettingsSection:
        """Retrieve a required mapping-valued config section."""
        self._ensure_loaded()
        if dotted_key in self._data:
            value = _interpolate(self._data[dotted_key], self._env)
        else:
            value = self.require(dotted_key)
        if not isinstance(value, dict):
            raise TypeError(f"Setting section must be a mapping: {dotted_key}")
        return SettingsSection(dotted_key, value, self._env)

    def _lookup(self, dotted_key: str, default: Any) -> Any:
        self._ensure_loaded()
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def raw(self) -> dict[str, Any]:
        """Return the full interpolated config dict (for debugging)."""
        self._ensure_loaded()
        return cast(dict[str, Any], _interpolate(self._data, self._env))

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()


#: Module-level settings singleton.
settings: Settings = Settings()
