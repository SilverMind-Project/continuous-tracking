#!/usr/bin/env python3
"""Per-camera face resolution report from continuous_tracking.face_observation_stats.

Answers, per camera: how many pixels sit between the eyes of the faces the cameras see, what raw
similarity recognized faces score, and how many recognized faces the face-ID gate discarded
(the ``dropped_min_confidence`` outcome). Use it to size and place cameras for face recognition.

Read-only. All aggregation runs in SQL. The share of person crops that yielded any face is not in
this table (rows exist only for faces); read it from the ``cts_face_queries_total`` metric.

Rows recorded since face crops were padded (upward and sideways around the person box) carry the
padded crop size in ``crop_width_px`` / ``crop_height_px``; earlier rows carry the tight person
box. Face sizes and interocular distances are frame pixels in both cases and remain comparable.

Usage:
    DATABASE_URL=postgresql://... python scripts/face_resolution_report.py --since 2026-10-20
    ... --until 2026-10-27 --camera cam-01 --format json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import asyncpg

# Interocular bands in pixels. Recognition is unreliable below about 16 px, marginal to 24 px,
# good to 40 px; above that the ArcFace 112x112 template (about 35 px between the eyes) is
# downsampled and accuracy saturates.
_BANDS_PX = ((None, 16.0), (16.0, 24.0), (24.0, 40.0), (40.0, None))

_SQL_INTEROCULAR = """
SELECT camera_id,
       count(*)                                                        AS faces,
       count(interocular_px)                                           AS measured,
       percentile_cont(0.1) WITHIN GROUP (ORDER BY interocular_px)     AS p10,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY interocular_px)     AS p50,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY interocular_px)     AS p90,
       count(*) FILTER (WHERE interocular_px < 16)                     AS band_lt16,
       count(*) FILTER (WHERE interocular_px >= 16 AND interocular_px < 24) AS band_16_24,
       count(*) FILTER (WHERE interocular_px >= 24 AND interocular_px < 40) AS band_24_40,
       count(*) FILTER (WHERE interocular_px >= 40)                    AS band_ge40
FROM continuous_tracking.face_observation_stats
WHERE observed_at >= $1 AND observed_at < $2 AND ($3::text IS NULL OR camera_id = $3)
GROUP BY camera_id
ORDER BY camera_id
"""

_SQL_SIMILARITY_BY_STATE = """
SELECT camera_id, recognition_state, count(*) AS faces,
       percentile_cont(0.1) WITHIN GROUP (ORDER BY raw_similarity) AS p10,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY raw_similarity) AS p50,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY raw_similarity) AS p90
FROM continuous_tracking.face_observation_stats
WHERE observed_at >= $1 AND observed_at < $2 AND ($3::text IS NULL OR camera_id = $3)
GROUP BY camera_id, recognition_state
ORDER BY camera_id, recognition_state
"""

_SQL_GATE_DROPS = """
SELECT camera_id, count(*) AS faces,
       percentile_cont(0.1) WITHIN GROUP (ORDER BY raw_similarity) AS p10,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY raw_similarity) AS p50,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY raw_similarity) AS p90
FROM continuous_tracking.face_observation_stats
WHERE observed_at >= $1 AND observed_at < $2 AND ($3::text IS NULL OR camera_id = $3)
  AND gate_outcome = 'dropped_min_confidence'
GROUP BY camera_id
ORDER BY camera_id
"""

_SQL_RECOGNIZED_BY_CANDIDATE = """
SELECT camera_id, best_candidate_id, count(*) AS faces
FROM continuous_tracking.face_observation_stats
WHERE observed_at >= $1 AND observed_at < $2 AND ($3::text IS NULL OR camera_id = $3)
  AND recognition_state = 'recognized'
GROUP BY camera_id, best_candidate_id
ORDER BY camera_id, faces DESC, best_candidate_id
"""


def _percentiles(row: asyncpg.Record) -> dict[str, float | None]:
    return {
        key: None if row[key] is None else round(float(row[key]), 3)
        for key in ("p10", "p50", "p90")
    }


async def build_report(
    conn: asyncpg.Connection, since: datetime, until: datetime, camera_id: str | None
) -> dict[str, Any]:
    """Return a JSON-serializable report keyed by camera id."""
    args = (since, until, camera_id)
    cameras: dict[str, dict[str, Any]] = {}

    def camera(cid: str) -> dict[str, Any]:
        return cameras.setdefault(
            cid,
            {
                "faces": 0,
                "interocular_px": None,
                "interocular_band_share": None,
                "raw_similarity_by_state": {},
                "dropped_min_confidence": {"faces": 0},
                "recognized_by_candidate": {},
            },
        )

    for row in await conn.fetch(_SQL_INTEROCULAR, *args):
        entry = camera(row["camera_id"])
        entry["faces"] = row["faces"]
        entry["interocular_px"] = {"measured": row["measured"], **_percentiles(row)}
        measured = row["measured"]
        bands = (row["band_lt16"], row["band_16_24"], row["band_24_40"], row["band_ge40"])
        entry["interocular_band_share"] = {
            _band_label(low, high): (round(count / measured, 3) if measured else None)
            for (low, high), count in zip(_BANDS_PX, bands, strict=True)
        }

    for row in await conn.fetch(_SQL_SIMILARITY_BY_STATE, *args):
        camera(row["camera_id"])["raw_similarity_by_state"][row["recognition_state"]] = {
            "faces": row["faces"],
            **_percentiles(row),
        }

    for row in await conn.fetch(_SQL_GATE_DROPS, *args):
        camera(row["camera_id"])["dropped_min_confidence"] = {
            "faces": row["faces"],
            **_percentiles(row),
        }

    for row in await conn.fetch(_SQL_RECOGNIZED_BY_CANDIDATE, *args):
        candidate = row["best_candidate_id"] or "(none)"
        camera(row["camera_id"])["recognized_by_candidate"][candidate] = row["faces"]

    return {
        "since": since.isoformat(),
        "until": until.isoformat(),
        "camera_filter": camera_id,
        "cameras": cameras,
    }


def _band_label(low: float | None, high: float | None) -> str:
    if low is None:
        return f"lt_{high:g}px"
    if high is None:
        return f"ge_{low:g}px"
    return f"{low:g}_to_{high:g}px"


def render_table(report: dict[str, Any]) -> str:
    """Human-readable rendering of :func:`build_report` output."""
    lines = [f"Face resolution report {report['since']} to {report['until']}"]
    if not report["cameras"]:
        lines.append("No face observations in this window.")
    for camera_id, entry in report["cameras"].items():
        lines.append("")
        lines.append(f"Camera {camera_id}: {entry['faces']} faces")
        iod = entry["interocular_px"]
        if iod:
            lines.append(
                f"  interocular px (n={iod['measured']}): "
                f"p10={iod['p10']} p50={iod['p50']} p90={iod['p90']}"
            )
            shares = entry["interocular_band_share"] or {}
            lines.append(
                "  band share: " + ", ".join(f"{band}={share}" for band, share in shares.items())
            )
        for state, stats in entry["raw_similarity_by_state"].items():
            lines.append(
                f"  raw similarity [{state}] (n={stats['faces']}): "
                f"p10={stats['p10']} p50={stats['p50']} p90={stats['p90']}"
            )
        drops = entry["dropped_min_confidence"]
        lines.append(
            f"  recognized faces dropped by the min-similarity gate: {drops['faces']}"
            + (
                f" (raw similarity p10={drops['p10']} p50={drops['p50']} p90={drops['p90']})"
                if drops["faces"]
                else ""
            )
        )
        if entry["recognized_by_candidate"]:
            lines.append(
                "  recognized faces by identity: "
                + ", ".join(f"{cid}={n}" for cid, n in entry["recognized_by_candidate"].items())
            )
    return "\n".join(lines)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _dsn_from_env() -> str:
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise SystemExit("DATABASE_URL environment variable is not set")
    return raw.replace("postgresql+asyncpg://", "postgresql://")


async def _main(args: argparse.Namespace) -> int:
    until = _parse_time(args.until) if args.until else datetime.now(UTC)
    conn = await asyncpg.connect(_dsn_from_env())
    try:
        report = await build_report(conn, _parse_time(args.since), until, args.camera)
    finally:
        await conn.close()
    output = json.dumps(report, indent=2) if args.format == "json" else render_table(report)
    sys.stdout.write(output + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", required=True, help="ISO date or datetime (UTC if no zone)")
    parser.add_argument("--until", default=None, help="ISO date or datetime; default now")
    parser.add_argument("--camera", default=None, help="Restrict to one camera id")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
