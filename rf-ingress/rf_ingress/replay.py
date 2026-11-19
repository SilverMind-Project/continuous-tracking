"""Time-limited raw adapter report recorder and JSONL replay CLI.

Usage:
    python -m rf_ingress.replay record --adapter <id> --minutes <1..60> --out <file.jsonl>
    python -m rf_ingress.replay play <file.jsonl>

Recordings capture raw internal adapter reports (``RangeMeasurement``,
``PositionReport``, ``RoomReport``), never solved ``TagFix`` messages, and are
hard-capped at 60 minutes per run. Recordings are private artifacts and are
git-ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rf_ingress.adapters.base import (
    AdapterDeps,
    AnyAdapter,
    PositionAdapter,
    RangeAdapter,
    RoomAdapter,
)
from rf_ingress.adapters.registry import build_default_registry
from rf_ingress.adapters.simulator import SimulatorAdapter
from rf_ingress.config import load_settings
from rf_ingress.domain import PositionReport, RangeMeasurement, RoomReport
from rf_ingress.pipeline import RfIngressRuntime
from rf_ingress.publish import InMemoryTagFixPublisher
from rf_ingress.site_config import (
    HttpSiteConfigClient,
    InMemorySiteConfigProvider,
    SiteConfigProvider,
)

MAX_RECORD_MINUTES = 60.0


def serialize_report_jsonl(report: RangeMeasurement | PositionReport | RoomReport) -> str:
    """Serialize one raw internal adapter report into a canonical JSONL line."""
    if isinstance(report, RangeMeasurement):
        payload: dict[str, object] = {
            "report_type": "range",
            "tag_id": report.tag_id,
            "anchor_id": report.anchor_id,
            "distance_m": report.distance_m,
            "variance_m2": report.variance_m2,
            "technology": report.technology.value,
            "measured_at": report.measured_at.astimezone(UTC).isoformat(),
            "los_confidence": report.los_confidence,
            "azimuth_deg": report.azimuth_deg,
            "azimuth_variance_deg2": report.azimuth_variance_deg2,
            "elevation_deg": report.elevation_deg,
            "azimuth_reference_frame": report.azimuth_reference_frame,
            "wear_state": report.wear_state.value,
            "motion_state": report.motion_state.value,
            "battery_fraction": report.battery_fraction,
            "epoch_seq": report.epoch_seq,
            "sent_at": (
                report.sent_at.astimezone(UTC).isoformat() if report.sent_at is not None else None
            ),
        }
    elif isinstance(report, PositionReport):
        payload = {
            "report_type": "position",
            "tag_id": report.tag_id,
            "measured_at": report.measured_at.astimezone(UTC).isoformat(),
            "x_native_m": report.x_native_m,
            "y_native_m": report.y_native_m,
            "z_native_m": report.z_native_m,
            "cov_native_m2": list(report.cov_native_m2)
            if report.cov_native_m2 is not None
            else None,
            "wear_state": report.wear_state.value,
            "motion_state": report.motion_state.value,
            "battery_fraction": report.battery_fraction,
            "epoch_seq": report.epoch_seq,
            "sent_at": (
                report.sent_at.astimezone(UTC).isoformat() if report.sent_at is not None else None
            ),
        }
    else:
        payload = {
            "report_type": "room",
            "tag_id": report.tag_id,
            "measured_at": report.measured_at.astimezone(UTC).isoformat(),
            "room_id": report.room_id,
            "probability": report.probability,
            "wear_state": report.wear_state.value,
            "motion_state": report.motion_state.value,
            "battery_fraction": report.battery_fraction,
            "epoch_seq": report.epoch_seq,
            "sent_at": (
                report.sent_at.astimezone(UTC).isoformat() if report.sent_at is not None else None
            ),
            "anchor_id": report.anchor_id,
            "distance_m": report.distance_m,
            "anchors_used": report.anchors_used,
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


async def record_adapter_to_jsonl(
    adapter: AnyAdapter,
    *,
    out_path: Path,
    max_minutes: float = 30.0,
) -> int:
    """Capture raw adapter reports to *out_path* for at most *max_minutes* (<= 60)."""
    if max_minutes <= 0.0 or max_minutes > MAX_RECORD_MINUTES:
        raise ValueError(f"max_minutes must be in (0, {MAX_RECORD_MINUTES}]")

    deadline = datetime.now(UTC) + timedelta(minutes=max_minutes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with out_path.open("w", encoding="utf-8") as fh:
        if isinstance(adapter, RangeAdapter):
            async for report in adapter.ranges():
                if datetime.now(UTC) >= deadline:
                    break
                fh.write(serialize_report_jsonl(report) + "\n")
                count += 1
        if isinstance(adapter, PositionAdapter):
            async for pos_report in adapter.positions():
                if datetime.now(UTC) >= deadline:
                    break
                fh.write(serialize_report_jsonl(pos_report) + "\n")
                count += 1
        if isinstance(adapter, RoomAdapter):
            async for room_report in adapter.rooms():
                if datetime.now(UTC) >= deadline:
                    break
                fh.write(serialize_report_jsonl(room_report) + "\n")
                count += 1
    return count


async def replay_jsonl_file(
    replay_path: Path,
    *,
    site_config: SiteConfigProvider,
    adapter_id: str = "sim-replay",
) -> InMemoryTagFixPublisher:
    """Replay *replay_path* through :class:`SimulatorAdapter` and return published fixes."""
    deps = AdapterDeps(site_config=site_config)
    adapter = SimulatorAdapter(
        adapter_id=adapter_id,
        deps=deps,
        replay_path=replay_path,
        realtime=False,
    )
    publisher = InMemoryTagFixPublisher()
    runtime = RfIngressRuntime(
        adapters=[adapter],
        site_config=site_config,
        publisher=publisher,
    )
    await runtime.run_once()
    await adapter.close()
    return publisher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rf_ingress.replay",
        description="Record raw RF adapter reports to JSONL or replay them through the simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record raw reports from a configured adapter (<= 60 min)")
    rec.add_argument("--adapter", required=True, help="Configured adapter id to record from")
    rec.add_argument(
        "--minutes",
        type=float,
        default=30.0,
        help="Recording duration in minutes (hard cap: 60)",
    )
    rec.add_argument("--out", type=Path, required=True, help="Output JSONL file path")
    rec.add_argument(
        "--settings",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to rf-ingress settings.yaml",
    )

    play = sub.add_parser("play", help="Replay a recorded JSONL file through the simulator adapter")
    play.add_argument("file", type=Path, help="Input JSONL recording path")
    play.add_argument("--adapter-id", default="sim-replay", help="Adapter id to tag replayed fixes")
    play.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Optional settings.yaml for polling site config from Cognitive Companion",
    )
    return parser


async def _run_cli(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "record":
        if args.minutes <= 0.0 or args.minutes > MAX_RECORD_MINUTES:
            parser.error(f"--minutes must be between 0 and {int(MAX_RECORD_MINUTES)}")
        settings = load_settings(args.settings)
        site_client = HttpSiteConfigClient(
            cc_api_url=settings.site_config.cc_api_url,
            api_key=settings.site_config.api_key,
            timeout_s=settings.site_config.request_timeout_s,
        )
        await site_client.refresh()
        registry = build_default_registry()
        deps = AdapterDeps(site_config=site_client)
        adapters = {a.adapter_id: a for a in registry.build_all(settings.adapters, deps)}
        target = adapters.get(args.adapter)
        if target is None:
            await site_client.close()
            sys.stderr.write(f"Unknown adapter id: {args.adapter}\n")
            return 2
        try:
            recorded = await record_adapter_to_jsonl(
                target,
                out_path=args.out,
                max_minutes=args.minutes,
            )
            sys.stdout.write(f"Recorded {recorded} reports to {args.out}\n")
        finally:
            await target.close()
            await site_client.close()
        return 0

    if args.command == "play":
        provider: SiteConfigProvider
        if args.settings is not None:
            settings = load_settings(args.settings)
            client = HttpSiteConfigClient(
                cc_api_url=settings.site_config.cc_api_url,
                api_key=settings.site_config.api_key,
                timeout_s=settings.site_config.request_timeout_s,
            )
            await client.refresh()
            provider = client
        else:
            provider = InMemorySiteConfigProvider()
        try:
            pub = await replay_jsonl_file(
                args.file,
                site_config=provider,
                adapter_id=args.adapter_id,
            )
            sys.stdout.write(
                f"Replayed {args.file}: published {len(pub.fixes)} TagFix message(s)\n"
            )
        finally:
            await provider.close()
        return 0

    return 2


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m rf_ingress.replay``."""
    return asyncio.run(_run_cli(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
