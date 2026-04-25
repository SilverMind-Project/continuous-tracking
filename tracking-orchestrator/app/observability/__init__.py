"""Observability primitives for the tracking orchestrator (M10).

Centralises Prometheus metrics so producers / consumers / pipeline stages
record against one canonical registry.  The metrics shape mirrors the
contract in phase-1 §1.9 and phase-3 §3.19.

Importers should reach for the metrics submodule explicitly so the
``metrics`` instance can be swapped (e.g. by tests using a fresh
``CollectorRegistry``)::

    from ..observability import metrics

    metrics.metrics.tracking_events_published_total.labels(camera_id=cid).inc()
"""

from __future__ import annotations

from . import metrics

__all__ = ["metrics"]
