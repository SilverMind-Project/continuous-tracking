"""Configuration-driven adapter registry for rf-ingress.

Maps each adapter ``kind`` discriminator to its factory callable. Unknown
``kind`` or ``payload_profile`` values fail startup immediately with a clear
:class:`ValueError`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from rf_ingress.adapters.base import AdapterDeps, AnyAdapter
from rf_ingress.adapters.simulator import build_simulator_adapter
from rf_ingress.config import AdapterConfig

AdapterFactory = Callable[[AdapterConfig, AdapterDeps], AnyAdapter]


class AdapterRegistry:
    """Registry resolving ``AdapterConfig.kind`` to concrete adapter instances."""

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, kind: str, factory: AdapterFactory) -> None:
        """Register *factory* under discriminator *kind*."""
        if not kind:
            raise ValueError("Adapter kind must be non-empty")
        if kind in self._factories:
            raise ValueError(f"Adapter kind already registered: {kind!r}")
        self._factories[kind] = factory

    def registered_kinds(self) -> tuple[str, ...]:
        """Return all registered adapter kinds in sorted order."""
        return tuple(sorted(self._factories.keys()))

    def build(self, config: AdapterConfig, deps: AdapterDeps) -> AnyAdapter:
        """Instantiate one adapter from *config* or raise :class:`ValueError`."""
        factory = self._factories.get(config.kind)
        if factory is None:
            known = ", ".join(self.registered_kinds()) or "<none>"
            raise ValueError(
                f"Unknown adapter kind {config.kind!r} for adapter {config.id!r}; "
                f"registered kinds: {known}"
            )
        return factory(config, deps)

    def build_all(
        self,
        configs: Iterable[AdapterConfig],
        deps: AdapterDeps,
    ) -> list[AnyAdapter]:
        """Instantiate all configured adapters, failing fast on any invalid config."""
        return [self.build(cfg, deps) for cfg in configs]


def build_default_registry() -> AdapterRegistry:
    """Create the default registry with the hardware-free built-in adapters."""
    registry = AdapterRegistry()
    registry.register("simulator", build_simulator_adapter)
    return registry
