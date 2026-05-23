"""Configuration loader for the tracking orchestrator.

Reads a single YAML file with ``${ENV_VAR}`` and ``${ENV_VAR:-default}``
interpolation, exposed as a dot-notation dict. Runtime configuration should
use :meth:`Settings.require` so defaults live in ``settings.yaml`` rather than
being duplicated at call sites.

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
from typing import Any

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
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Setting {key} must be an integer, got {value!r}") from exc


def _as_float(value: object, key: str) -> float:
    try:
        return float(value)
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

    def __init__(self, prefix: str, data: Mapping[str, Any]) -> None:
        self._prefix = prefix
        self._data = data

    def require(self, key: str) -> Any:
        if key not in self._data:
            raise SettingNotFoundError(self._qualify(key))
        return self._data[key]

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
    """Replace ``${VAR}`` and ``${VAR:-default}`` placeholders.

    Brace depth is counted so nested ``${}`` in fallback values are matched
    correctly.  Fallback values are themselves resolved recursively.
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
                var_name, fallback = content.split(":-", 1)
                val = env.get(var_name, "")
                result.append(val if val else _resolve_string(fallback, env))
            else:
                result.append(env.get(content, ""))
            i = j
        else:
            result.append(value[i])
            i += 1
    return "".join(result)


def _interpolate(value: Any, env: Mapping[str, str]) -> Any:
    """Recursively replace ``${VAR}`` and ``${VAR:-default}`` placeholders.

    A bare ``${VAR}`` with a missing variable becomes an empty string.
    ``${VAR:-fallback}`` uses the fallback when VAR is missing or empty.
    """
    if isinstance(value, str):
        return _resolve_string(value, env)
    if isinstance(value, dict):
        return {k: _interpolate(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, env) for v in value]
    return value


class Settings:
    """YAML-backed settings with dot-notation access and env-var interpolation.

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
        self._data = _interpolate(raw, self._env)
        self._loaded = True

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a nested value using dot notation.

        Returns *default* if any segment is missing or traverses through
        a non-dict value.
        """
        return self._lookup(dotted_key, default)

    def require(self, dotted_key: str) -> Any:
        """Retrieve a required nested value using dot notation.

        Missing keys raise :class:`SettingNotFoundError`. Empty strings are
        valid values because several optional integrations are represented in
        ``settings.yaml`` as ``${ENV_VAR:-}``.
        """
        value = self._lookup(dotted_key, _MISSING)
        if value is _MISSING:
            raise SettingNotFoundError(dotted_key)
        return value

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
        value = self.require(dotted_key)
        if not isinstance(value, dict):
            raise TypeError(f"Setting section must be a mapping: {dotted_key}")
        return SettingsSection(dotted_key, value)

    def _lookup(self, dotted_key: str, default: Any) -> Any:
        self._ensure_loaded()
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def raw(self) -> dict[str, Any]:
        """Return the full merged config dict (for debugging)."""
        self._ensure_loaded()
        return self._data

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()


#: Module-level settings singleton.
settings: Settings = Settings()
