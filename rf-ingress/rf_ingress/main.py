"""Composition root for rf-ingress: config, adapters, solver, publisher, FastAPI app."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI
from redis.asyncio import Redis

from rf_ingress.adapters.base import AdapterDeps
from rf_ingress.adapters.registry import AdapterRegistry, build_default_registry
from rf_ingress.config import RfIngressSettings, load_settings
from rf_ingress.health import create_health_app
from rf_ingress.observability import configure_logging
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import RedisTagFixPublisher
from rf_ingress.site_config import HttpSiteConfigClient
from rf_ingress.solver.multilateration import MultilaterationConfig

logger = structlog.get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def build_runtime(
    settings: RfIngressSettings,
    *,
    registry: AdapterRegistry | None = None,
) -> RfIngressRuntime:
    """Wire all production dependencies explicitly from *settings*."""
    configure_logging(settings.service.log_level)
    adapter_registry = registry if registry is not None else build_default_registry()

    site_config_client = HttpSiteConfigClient(
        cc_api_url=settings.site_config.cc_api_url,
        api_key=settings.site_config.api_key,
        timeout_s=settings.site_config.request_timeout_s,
    )
    redis_client = Redis.from_url(settings.redis.url, decode_responses=False)
    publisher = RedisTagFixPublisher(
        redis_client,
        fixes_stream=settings.redis.fixes_stream,
        capabilities_stream=settings.redis.capabilities_stream,
        retention_hours=settings.redis.retention_hours,
    )
    deps = AdapterDeps(site_config=site_config_client)
    adapters = adapter_registry.build_all(settings.adapters, deps)

    solver_cfg = MultilaterationConfig(
        assumed_tag_height_m=settings.solver.assumed_tag_height_m,
        tag_height_sigma_m=settings.solver.tag_height_sigma_m,
        min_vertical_spread_m=settings.solver.min_vertical_spread_m,
        max_condition=settings.solver.max_condition,
        quality_scale_m=settings.solver.quality_scale_m,
        max_sigma_m=settings.solver.max_sigma_m,
    )
    return RfIngressRuntime(
        adapters=adapters,
        site_config=site_config_client,
        publisher=publisher,
        solver_config=solver_cfg,
        epoch_window_s=settings.solver.epoch_window_s,
        site_config_poll_interval_s=settings.site_config.poll_interval_s,
        capability_heartbeat_interval_s=settings.service.capability_heartbeat_interval_s,
        skew_window_s=settings.clock.skew_window_s,
        skew_warn_threshold_s=settings.clock.skew_warn_threshold_s,
        burst_window_size=settings.clock.burst_window_size,
    )


def create_app(config_path: Path | str | None = None) -> FastAPI:
    """FastAPI application factory for uvicorn and production deployment."""
    resolved_path = Path(
        config_path
        if config_path is not None
        else os.environ.get("RF_INGRESS_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    settings = load_settings(resolved_path)
    runtime = build_runtime(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("rf_ingress_starting", config_path=str(resolved_path))
        await runtime.start()
        try:
            yield
        finally:
            logger.info("rf_ingress_stopping")
            await runtime.stop()

    return create_health_app(runtime, lifespan=lifespan)


def main() -> None:
    """Run the rf-ingress HTTP health/metrics server and background loops."""
    resolved_path = Path(os.environ.get("RF_INGRESS_CONFIG", str(DEFAULT_CONFIG_PATH)))
    settings = load_settings(resolved_path)
    app = create_app(resolved_path)
    uvicorn.run(
        app,
        host=settings.service.host,
        port=settings.service.port,
        log_level=settings.service.log_level.lower(),
    )


if __name__ == "__main__":
    main()
