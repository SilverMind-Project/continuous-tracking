"""FastAPI health and Prometheus metrics endpoints for rf-ingress."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from rf_ingress.domain import ensure_utc
from rf_ingress.pipeline import RfIngressRuntime


class AdapterHealthOut(BaseModel):
    """Per-adapter connectivity and drop counters returned on ``GET /health``."""

    adapter_id: str
    connected: bool
    last_report_age_s: float | None
    malformed_reports: int
    unregistered_reports: int


class SiteConfigHealthOut(BaseModel):
    """Site-config synchronization state returned on ``GET /health``."""

    loaded: bool
    version: str
    floor_plan_id: str | None
    age_s: float | None
    node_count: int
    device_count: int
    binding_count: int


class PublisherHealthOut(BaseModel):
    """Publisher state returned on ``GET /health``."""

    healthy: bool


class HealthOut(BaseModel):
    """Top-level response schema for ``GET /health``."""

    status: Literal["ok", "degraded"]
    adapters: list[AdapterHealthOut]
    site_config: SiteConfigHealthOut
    publisher: PublisherHealthOut
    node_last_seen_age_s: dict[str, float]
    tag_last_seen_age_s: dict[str, float] = Field(default_factory=dict)
    tag_battery_fraction: dict[str, float] = Field(default_factory=dict)


def create_health_app(
    runtime: RfIngressRuntime,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    lifespan: object | None = None,
) -> FastAPI:
    """Create the FastAPI app serving ``/health`` and ``/metrics``."""
    app = FastAPI(
        title="rf-ingress",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,  # type: ignore[arg-type]
    )

    @app.get("/health", response_model=HealthOut)
    async def health() -> HealthOut:
        now = ensure_utc(clock())
        snap = runtime.health_snapshot()
        adapters_out = [
            AdapterHealthOut(
                adapter_id=a.adapter_id,
                connected=a.connected,
                last_report_age_s=(
                    max(0.0, (now - ensure_utc(a.last_report_at)).total_seconds())
                    if a.last_report_at is not None
                    else None
                ),
                malformed_reports=a.malformed_reports,
                unregistered_reports=a.unregistered_reports,
            )
            for a in snap.adapters
        ]
        all_connected = all(a.connected for a in snap.adapters)
        overall: Literal["ok", "degraded"] = (
            "ok" if (snap.publisher_healthy and all_connected) else "degraded"
        )
        return HealthOut(
            status=overall,
            adapters=adapters_out,
            site_config=SiteConfigHealthOut(
                loaded=snap.site_config_loaded,
                version=snap.site_config_version,
                floor_plan_id=snap.site_config_floor_plan_id,
                age_s=snap.site_config_age_s,
                node_count=snap.node_count,
                device_count=snap.device_count,
                binding_count=snap.binding_count,
            ),
            publisher=PublisherHealthOut(healthy=snap.publisher_healthy),
            node_last_seen_age_s=snap.node_last_seen_age_s,
            tag_last_seen_age_s=snap.tag_last_seen_age_s,
            tag_battery_fraction=snap.tag_battery_fraction,
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
