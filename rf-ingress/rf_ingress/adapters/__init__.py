"""RF localization adapters, protocols, and registry."""

from __future__ import annotations

from rf_ingress.adapters.base import (
    AdapterDeps,
    AdapterHealth,
    AnyAdapter,
    BaseAdapter,
    PositionAdapter,
    RangeAdapter,
    RoomAdapter,
)
from rf_ingress.adapters.registry import AdapterFactory, AdapterRegistry, build_default_registry

__all__ = [
    "AdapterDeps",
    "AdapterFactory",
    "AdapterHealth",
    "AdapterRegistry",
    "AnyAdapter",
    "BaseAdapter",
    "PositionAdapter",
    "RangeAdapter",
    "RoomAdapter",
    "build_default_registry",
]
