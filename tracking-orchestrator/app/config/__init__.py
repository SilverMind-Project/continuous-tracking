"""Configuration loader for the tracking orchestrator.

Reads a single YAML file with ``${ENV_VAR}`` and ``${ENV_VAR:-default}``
interpolation, exposed as a dot-notation dict.

Usage::

    from app.config import settings

    redis_url = settings.get("redis.url")
    face_id = settings.get("face_id.url", "")

The config file is resolved from ``ORCHESTRATOR_CONFIG_PATH``, or defaults
to ``config/settings.yaml`` relative to the project root.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

__all__ = ["SettingNotFoundError", "Settings", "settings"]

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
_CONFIG_PATH = os.environ.get("ORCHESTRATOR_CONFIG_PATH", str(_DEFAULT_CONFIG))


class SettingNotFoundError(KeyError):
    """Raised when a required config value is missing."""

    def __init__(self, dotted_key: str) -> None:
        super().__init__(f"Required setting not found: {dotted_key}")
        self.dotted_key = dotted_key


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
        assert s.get("redis.url") == "redis://localhost:6379/0"
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
            try:
                self.reload()
            except FileNotFoundError:
                # Config file is optional; fall back to empty + env vars.
                self._data = {}
                self._loaded = True


#: Module-level settings singleton.
settings: Settings = Settings()
