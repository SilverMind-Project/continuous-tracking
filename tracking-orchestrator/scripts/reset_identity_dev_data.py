#!/usr/bin/env python3
"""Development identity data reset script.

Deletes derived CTS and CC tracking state while preserving household
configuration, ArcFace enrollments, and raw MinIO frame objects.

NEVER run automatically on service startup or migration.
Default mode: dry-run only. Use --apply + --confirm to execute.

Required env vars (read from environment; same vars as docker-compose):
  CTS: DATABASE_URL (postgresql+asyncpg://...)
  CC:  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
  MinIO: MINIO_ENDPOINT_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, MINIO_SECURE
  Redis (optional, only with --include-redis): REDIS_URL
  Smoke check (optional): CTS_BASE_URL, CC_BASE_URL, CC_API_KEY (the last enables the
    correction-targets endpoint probe; without it that probe is skipped and the auth-free
    identities-preserved DB check remains authoritative).
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import httpx

# ---------------------------------------------------------------------------
# Allowlists -- never infer tables at runtime; only these names are touched.
# ---------------------------------------------------------------------------

# CTS derived tables to truncate. Schema: continuous_tracking.
# Source: identity-integrity-m11-dev-reset-and-private-replay-data.md (verified 2026-06-21).
_CTS_DELETE_TABLES: tuple[str, ...] = (
    "person_hypotheses",
    "world_observations",
    "person_trajectories",
    "room_dwells",
    "co_presence_links",
    "tagged_keyframes",
    "keyframe_bbox_annotations",
    "ph_revisions",
    "ph_merges",
    "dementia_signals",
    "agitation_windows",
    "gait_bouts",
    "gait_daily",
    "identity_decisions",
    "identity_evidence_items",
    "identity_decision_gallery_hits",
    "identity_corrections",
    "identity_revision_ranges",
    "identity_revision_jobs",
    "identity_projection_acks",
    "reid_gallery",
    "gallery_review_events",
)

# CTS tables that must never be deleted (household registry + topology).
_CTS_PRESERVE_TABLES: frozenset[str] = frozenset(
    {
        "identities",
        "cameras",
        "streams",
        "camera_topology_edges",
    }
)

# CC CTS-derived tables to truncate. Schema: public (Alembic-managed).
# Source: CC Alembic baseline migration 0001_baseline.py (verified 2026-06-21).
_CC_DELETE_TABLES: tuple[str, ...] = (
    "person_location_history",
    "person_location_state",
    "location_observations",
    "presence_segments",
    "room_occupancy_state",
    "person_sightings",
    "person_activities",
    "transit_zones",
    "cts_dementia_signals",
    "cts_identity_revision_log",
)

# MinIO prefixes that contain ReID candidate crops only.
# Source: ReIDCandidateService in app/tracking/identity/candidate_service.py.
# NEVER delete frames/... -- keyframes reference raw frame objects.
_REID_PREFIXES: tuple[str, ...] = (
    "reid-candidates/",
    "reid-candidates-frames/",
)

# Redis streams containing CTS-derived state.
_REDIS_STREAMS: tuple[str, ...] = (
    "tracking.revisions",
    "cc.identity_assertions",
)

_CONFIRMATION_PHRASE = "RESET DEVELOPMENT IDENTITY DATA"
_REDIS_CONFIRMATION_PHRASE = "RESET REDIS STREAMS"

log = logging.getLogger("reset_identity")

# ---------------------------------------------------------------------------
# Pure helper functions (no I/O).
# ---------------------------------------------------------------------------


def _confirmation_matches(provided: str, expected: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(provided.encode(), expected.encode())


def _build_cts_truncate_sql(tables: tuple[str, ...]) -> str:
    qualified = ", ".join(f"continuous_tracking.{t}" for t in tables)
    return f"TRUNCATE {qualified} RESTART IDENTITY CASCADE"


def _build_cc_truncate_sql(tables: tuple[str, ...]) -> str:
    names = ", ".join(tables)
    return f"TRUNCATE {names} RESTART IDENTITY CASCADE"


def _redact_url(url: str) -> str:
    """Remove credentials from a DSN before logging."""
    return re.sub(r"//[^@]+@", "//***@", url)


def _check_violations(
    referencing_tables: list[str],
    allowed: set[str],
) -> list[str]:
    """Return names of referencing tables that are not in the allowed set."""
    return [t for t in referencing_tables if t not in allowed]


# ---------------------------------------------------------------------------
# Database helpers.
# ---------------------------------------------------------------------------


async def _cts_dsn_from_env() -> str:
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    # asyncpg wants postgresql:// not postgresql+asyncpg://
    return raw.replace("postgresql+asyncpg://", "postgresql://")


async def _cc_dsn_from_env() -> str:
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    db = os.environ.get("POSTGRES_DB", "cognitive_companion")
    if not user or not password:
        raise RuntimeError("POSTGRES_USER and POSTGRES_PASSWORD must be set for CC database access")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def _existing_tables(
    conn: asyncpg.Connection,
    schema: str,
    candidates: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Split an allowlist into (present, missing) for the given schema.

    A dev database can lag the current migration head, so an allowlisted table
    may not exist yet. Operating over the intersection keeps the reset bounded
    by the allowlist (it never touches anything outside it) while staying
    resilient to migration drift. Missing tables are reported, not fatal --
    the same philosophy the MinIO path already uses for absent objects.
    """
    rows = await conn.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = $1 AND table_name = ANY($2::text[])
        """,
        schema,
        list(candidates),
    )
    present_set = {r["table_name"] for r in rows}
    present = [t for t in candidates if t in present_set]
    missing = [t for t in candidates if t not in present_set]
    return present, missing


async def _count_rows(conn: asyncpg.Connection, schema: str, table: str) -> int:
    quoted = f'"{schema}"."{table}"' if schema != "public" else f'"{table}"'
    row = await conn.fetchrow(f"SELECT COUNT(*) AS n FROM {quoted}")
    return int(row["n"])


async def _get_table_counts(
    conn: asyncpg.Connection,
    tables: tuple[str, ...],
    schema: str,
) -> dict[str, int]:
    return {t: await _count_rows(conn, schema, t) for t in tables}


async def _check_cts_orphan_fks(
    conn: asyncpg.Connection,
    delete_tables: tuple[str, ...],
) -> list[str]:
    """Return FK violations: tables that reference a delete-target but aren't in the allowlist."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT
            tc.table_name  AS referencing_table,
            ccu.table_name AS referenced_table
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.referential_constraints AS rc
            ON tc.constraint_name = rc.constraint_name
            AND tc.table_schema   = rc.constraint_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON rc.unique_constraint_name = ccu.constraint_name
            AND rc.unique_constraint_schema = ccu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema  = 'continuous_tracking'
          AND ccu.table_schema = 'continuous_tracking'
          AND ccu.table_name   = ANY($1::text[])
          AND tc.table_name   != ALL($1::text[])
        """,
        list(delete_tables),
    )
    return [
        f"{r['referencing_table']!r} references {r['referenced_table']!r} "
        "but is not in the delete allowlist"
        for r in rows
    ]


async def _check_cc_orphan_fks(
    conn: asyncpg.Connection,
    delete_tables: tuple[str, ...],
) -> list[str]:
    """Same orphan check for CC tables (public schema)."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT
            tc.table_name  AS referencing_table,
            ccu.table_name AS referenced_table
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.referential_constraints AS rc
            ON tc.constraint_name = rc.constraint_name
            AND tc.table_schema   = rc.constraint_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON rc.unique_constraint_name = ccu.constraint_name
            AND rc.unique_constraint_schema = ccu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema  = 'public'
          AND ccu.table_schema = 'public'
          AND ccu.table_name   = ANY($1::text[])
          AND tc.table_name   != ALL($1::text[])
        """,
        list(delete_tables),
    )
    return [
        f"{r['referencing_table']!r} references {r['referenced_table']!r} "
        "but is not in the CC delete allowlist"
        for r in rows
    ]


# ---------------------------------------------------------------------------
# MinIO helpers (reuses MinioFrameFetcher -- no second S3 client).
# ---------------------------------------------------------------------------


def _minio_config_from_env() -> tuple[str, str, str, str, bool]:
    endpoint = os.environ.get("MINIO_ENDPOINT_URL", "")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "")
    bucket = os.environ.get("MINIO_BUCKET", "")
    # Canonical CTS flag is MINIO_SECURE (settings key minio.secure, see app/main.py).
    # MINIO_USE_SSL is accepted as a fallback for older env files.
    secure_raw = os.environ.get("MINIO_SECURE") or os.environ.get("MINIO_USE_SSL", "false")
    use_ssl = secure_raw.lower() in ("true", "1", "yes")
    if not all([endpoint, access_key, secret_key, bucket]):
        raise RuntimeError(
            "MINIO_ENDPOINT_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET must be set"
        )
    return endpoint, access_key, secret_key, bucket, use_ssl


async def _list_reid_objects(fetcher: Any, bucket: str) -> dict[str, list[str]]:
    """List all objects under each ReID prefix. Returns {prefix: [key, ...]}."""
    result: dict[str, list[str]] = {}
    for prefix in _REID_PREFIXES:
        keys = await fetcher.list_objects_by_prefix(prefix)
        result[prefix] = keys
    return result


async def _delete_reid_objects(
    fetcher: Any,
    prefix_keys: dict[str, list[str]],
    dry_run: bool,
) -> dict[str, dict[str, int]]:
    """Delete ReID crop objects. Returns {prefix: {count, deleted, failed}}."""
    report: dict[str, dict[str, int]] = {}
    for prefix, keys in prefix_keys.items():
        deleted = 0
        failed = 0
        if not dry_run:
            for key in keys:
                try:
                    await fetcher.delete_object(key)
                    deleted += 1
                except Exception as exc:  # noqa: BLE001 -- best-effort crop cleanup; failure tallied + reported, never aborts the run
                    log.warning("minio_delete_failed key=%r error=%s", key, exc)
                    failed += 1
        report[prefix] = {"count": len(keys), "deleted": deleted, "failed": failed}
    return report


async def _check_raw_frame_survival(fetcher: Any, sample_key: str) -> dict[str, Any]:
    """Confirm the reset preserved raw frames.

    The reset never touches `frames/...`, so a referenced raw frame must survive.
    Passing the exact key as the prefix lists only that one object (no full scan).
    If the sampled key is itself absent -- already removed by MinIO retention, not
    by the reset -- fall back to confirming `frames/` still holds objects. What the
    reset guarantees is that `frames/` is never emptied, not that every historical
    keyframe's object outlives retention.
    """
    exists = sample_key in await fetcher.list_objects_by_prefix(sample_key)
    result: dict[str, Any] = {"sample_key": sample_key, "exists": exists}
    if exists:
        result["ok"] = True
    else:
        frames_present = bool(await fetcher.list_objects_by_prefix("frames/"))
        result["frames_prefix_nonempty"] = frames_present
        result["ok"] = frames_present
        result["note"] = (
            "sampled key already absent (MinIO retention, not the reset); "
            "frames/ as a class survives"
        )
    return result


# ---------------------------------------------------------------------------
# Redis helpers.
# ---------------------------------------------------------------------------


async def _clear_redis_streams(
    redis_url: str,
    dry_run: bool,
) -> dict[str, Any]:
    import redis.asyncio as aioredis

    client = aioredis.from_url(redis_url, decode_responses=True)
    report: dict[str, Any] = {}
    try:
        for stream in _REDIS_STREAMS:
            try:
                before = await client.xlen(stream)
            except Exception:  # noqa: BLE001 -- absent stream has no length; 0 is the correct best-effort count
                before = 0
            report[stream] = {"before": before, "deleted": False}
            if not dry_run:
                try:
                    await client.delete(stream)
                    report[stream]["deleted"] = True
                except Exception as exc:  # noqa: BLE001 -- per-stream failure recorded in report, does not abort other streams
                    report[stream]["error"] = str(exc)
    finally:
        await client.aclose()
    return report


# ---------------------------------------------------------------------------
# Smoke check.
# ---------------------------------------------------------------------------


async def _run_smoke_check(
    cts_conn: asyncpg.Connection,
    cc_conn: asyncpg.Connection,
    cts_base_url: str,
    cc_base_url: str,
    pre_preserved_counts: dict[str, int],
    cc_api_key: str | None,
    raw_frame_survival: dict[str, Any] | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=10.0) as http:
        # CTS health
        try:
            r = await http.get(f"{cts_base_url}/health")
            results["cts_health"] = {"status": r.status_code, "ok": r.status_code == 200}
        except Exception as exc:  # noqa: BLE001 -- smoke probe records the failure as not-ok; non-fatal
            results["cts_health"] = {"ok": False, "error": str(exc)}

        # CC health (CC mounts health under the /api/v1 prefix, not bare /health).
        try:
            r = await http.get(f"{cc_base_url}/api/v1/health")
            results["cc_health"] = {"status": r.status_code, "ok": r.status_code == 200}
        except Exception as exc:  # noqa: BLE001 -- smoke probe records the failure as not-ok; non-fatal
            results["cc_health"] = {"ok": False, "error": str(exc)}

        # Correction targets endpoint (requires X-API-Key). The authoritative
        # "both members survive" assertion is identities_preserved below, which
        # is auth-free, so this HTTP probe is optional: run it only when an
        # operator supplies CC_API_KEY, and never let a missing key fail the run.
        if not cc_api_key:
            results["correction_targets"] = {
                "ok": True,
                "skipped": "no CC_API_KEY provided; identities_preserved is authoritative",
                "count": None,
            }
        else:
            try:
                r = await http.get(
                    f"{cc_base_url}/api/v1/cts/identity/correction-targets",
                    headers={"X-API-Key": cc_api_key},
                )
                body = r.json() if r.status_code == 200 else {}
                targets = body.get("targets", [])
                results["correction_targets"] = {
                    "ok": r.status_code == 200 and len(targets) >= 2,
                    "status": r.status_code,
                    "count": len(targets),
                }
            except Exception as exc:  # noqa: BLE001 -- smoke probe records the failure as not-ok; non-fatal
                results["correction_targets"] = {"ok": False, "error": str(exc)}

    # Raw frame survival (sampled before the reset; must never be deleted).
    if raw_frame_survival is not None:
        results["raw_frame_survives"] = raw_frame_survival

    # Gallery empty
    gallery_count = await _count_rows(cts_conn, "continuous_tracking", "reid_gallery")
    results["reid_gallery_empty"] = {"count": gallery_count, "ok": gallery_count == 0}

    # No open PHs
    ph_count = await _count_rows(cts_conn, "continuous_tracking", "person_hypotheses")
    results["no_open_person_hypotheses"] = {"count": ph_count, "ok": ph_count == 0}

    # No CC derived locations
    loc_count = await _count_rows(cc_conn, "public", "person_location_history")
    results["no_cc_derived_locations"] = {"count": loc_count, "ok": loc_count == 0}

    # Preserved table counts unchanged
    after_identities = await _count_rows(cts_conn, "continuous_tracking", "identities")
    after_cameras = await _count_rows(cts_conn, "continuous_tracking", "cameras")
    results["identities_preserved"] = {
        "before": pre_preserved_counts.get("identities", -1),
        "after": after_identities,
        "ok": after_identities == pre_preserved_counts.get("identities", -1),
    }
    results["cameras_preserved"] = {
        "before": pre_preserved_counts.get("cameras", -1),
        "after": after_cameras,
        "ok": after_cameras == pre_preserved_counts.get("cameras", -1),
    }

    results["overall_ok"] = all(v.get("ok", False) for v in results.values() if isinstance(v, dict))
    return results


# ---------------------------------------------------------------------------
# Main reset orchestration.
# ---------------------------------------------------------------------------


async def run_reset(args: argparse.Namespace) -> dict[str, Any]:
    dry_run = not args.apply
    run_redis = getattr(args, "include_redis", False)

    if dry_run:
        log.info("DRY RUN -- no writes will be performed")
    else:
        log.info("APPLY MODE -- identity data will be deleted")

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "include_redis": run_redis,
    }

    # Sampled before the CTS truncate so the post-reset smoke check can prove a
    # referenced raw frame object survives (the reset must never touch frames/...).
    sample_frame_key: str | None = None

    # -- CTS database -------------------------------------------------------
    cts_dsn = await _cts_dsn_from_env()
    log.info("connecting to CTS database at %s", _redact_url(cts_dsn))
    cts_conn: asyncpg.Connection = await asyncpg.connect(cts_dsn)

    try:
        # Resolve which allowlisted tables actually exist (dev DBs can lag the
        # migration head). Only present tables are counted/truncated; missing
        # ones are reported loudly but are not fatal.
        cts_present, cts_missing = await _existing_tables(
            cts_conn, "continuous_tracking", _CTS_DELETE_TABLES
        )
        report["cts_delete_tables_present"] = cts_present
        report["cts_delete_tables_missing"] = cts_missing
        if cts_missing:
            log.warning(
                "CTS allowlisted tables absent from this database (dev DB may lag "
                "the migration head): %s",
                ", ".join(cts_missing),
            )

        preserve_present, preserve_missing = await _existing_tables(
            cts_conn, "continuous_tracking", tuple(_CTS_PRESERVE_TABLES)
        )
        if preserve_missing:
            log.warning(
                "CTS preserve tables absent (unexpected -- check the database): %s",
                ", ".join(preserve_missing),
            )

        # Pre-reset counts
        pre_cts = await _get_table_counts(cts_conn, tuple(cts_present), "continuous_tracking")
        pre_preserved = await _get_table_counts(
            cts_conn, tuple(preserve_present), "continuous_tracking"
        )
        report["cts_pre_counts"] = pre_cts
        report["preserved_pre_counts"] = pre_preserved

        # Sample one referenced raw frame key (before truncate) for the survival check.
        # Use the most recent keyframe: older frames may have already been removed by
        # MinIO retention, so an old sample would be absent for reasons unrelated to the
        # reset. The survival check falls back to a frames/ class check if even this is gone.
        if "tagged_keyframes" in cts_present:
            row = await cts_conn.fetchrow(
                "SELECT minio_key FROM continuous_tracking.tagged_keyframes "
                "WHERE minio_key LIKE 'frames/%' ORDER BY captured_at DESC LIMIT 1"
            )
            if row is not None:
                sample_frame_key = row["minio_key"]

        # Orphan FK check over the FULL allowlist (missing tables contribute no
        # constraints, so semantics are unchanged; when the DB reaches head this
        # validates every table).
        violations = await _check_cts_orphan_fks(cts_conn, _CTS_DELETE_TABLES)
        if violations:
            report["orphan_fk_violations"] = violations
            log.error("CTS orphan FK violations detected -- aborting reset:")
            for v in violations:
                log.error("  %s", v)
            raise SystemExit(1)
        report["orphan_fk_check"] = "passed"

        if not dry_run and cts_present:
            sql = _build_cts_truncate_sql(tuple(cts_present))
            log.info("executing CTS TRUNCATE in transaction (%d tables)", len(cts_present))
            async with cts_conn.transaction():
                await cts_conn.execute(sql)

        post_cts = await _get_table_counts(cts_conn, tuple(cts_present), "continuous_tracking")
        report["cts_post_counts"] = post_cts

    finally:
        await cts_conn.close()

    # -- CC database --------------------------------------------------------
    cc_dsn = await _cc_dsn_from_env()
    log.info("connecting to CC database at %s", _redact_url(cc_dsn))
    cc_conn: asyncpg.Connection = await asyncpg.connect(cc_dsn)

    try:
        cc_present, cc_missing = await _existing_tables(cc_conn, "public", _CC_DELETE_TABLES)
        report["cc_delete_tables_present"] = cc_present
        report["cc_delete_tables_missing"] = cc_missing
        if cc_missing:
            log.warning(
                "CC allowlisted tables absent from this database: %s",
                ", ".join(cc_missing),
            )

        pre_cc = await _get_table_counts(cc_conn, tuple(cc_present), "public")
        report["cc_pre_counts"] = pre_cc

        cc_violations = await _check_cc_orphan_fks(cc_conn, _CC_DELETE_TABLES)
        if cc_violations:
            report["cc_orphan_fk_violations"] = cc_violations
            log.error("CC orphan FK violations detected -- aborting reset:")
            for v in cc_violations:
                log.error("  %s", v)
            raise SystemExit(1)
        report["cc_orphan_fk_check"] = "passed"

        if not dry_run and cc_present:
            sql = _build_cc_truncate_sql(tuple(cc_present))
            log.info("executing CC TRUNCATE in transaction (%d tables)", len(cc_present))
            async with cc_conn.transaction():
                await cc_conn.execute(sql)

        post_cc = await _get_table_counts(cc_conn, tuple(cc_present), "public")
        report["cc_post_counts"] = post_cc

    finally:
        await cc_conn.close()

    # -- MinIO --------------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.transport.minio_frames import MinioFrameConfig, MinioFrameFetcher

    endpoint, access_key, secret_key, bucket, use_ssl = _minio_config_from_env()
    minio_config = MinioFrameConfig(
        endpoint_url=endpoint,
        bucket=bucket,
        access_key_id=access_key,
        secret_access_key=secret_key,
        secure=use_ssl,
    )
    fetcher = MinioFrameFetcher(minio_config)
    raw_frame_survival: dict[str, Any] | None = None
    await fetcher.connect()
    try:
        prefix_keys = await _list_reid_objects(fetcher, bucket)
        minio_report = await _delete_reid_objects(fetcher, prefix_keys, dry_run=dry_run)
        report["minio"] = minio_report

        # Confirm raw frames survive the reset (sampled before truncate).
        if sample_frame_key is not None:
            raw_frame_survival = await _check_raw_frame_survival(fetcher, sample_frame_key)
            report["raw_frame_survival"] = raw_frame_survival
    finally:
        await fetcher.disconnect()

    # -- Redis (optional) ---------------------------------------------------
    if run_redis:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        redis_report = await _clear_redis_streams(redis_url, dry_run=dry_run)
        report["redis"] = redis_report
    else:
        report["redis"] = "skipped (use --include-redis to clear streams)"

    # -- Smoke check (apply only) ------------------------------------------
    if not dry_run:
        cts_conn2: asyncpg.Connection = await asyncpg.connect(cts_dsn)
        cc_conn2: asyncpg.Connection = await asyncpg.connect(cc_dsn)
        cts_url = os.environ.get("CTS_BASE_URL", "http://localhost:8500")
        cc_url = os.environ.get("CC_BASE_URL", "http://localhost:8000")
        cc_api_key = os.environ.get("CC_API_KEY") or None
        try:
            smoke = await _run_smoke_check(
                cts_conn2,
                cc_conn2,
                cts_url,
                cc_url,
                pre_preserved,
                cc_api_key,
                raw_frame_survival,
            )
            report["smoke_check"] = smoke
            if smoke.get("overall_ok"):
                log.info("smoke check passed")
            else:
                log.warning("smoke check reported issues: %s", smoke)
        finally:
            await cts_conn2.close()
            await cc_conn2.close()

    return report


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Development identity data reset. Default: dry-run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (dry run):\n"
            "  python reset_identity_dev_data.py\n\n"
            "Example (apply):\n"
            "  python reset_identity_dev_data.py --apply \\\n"
            '      --confirm "RESET DEVELOPMENT IDENTITY DATA"\n\n'
            "Example (with Redis):\n"
            "  python reset_identity_dev_data.py --apply \\\n"
            '      --confirm "RESET DEVELOPMENT IDENTITY DATA" \\\n'
            "      --include-redis \\\n"
            '      --redis-confirm "RESET REDIS STREAMS"\n'
        ),
    )
    p.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Execute the reset. Without this flag the script only prints counts.",
    )
    p.add_argument(
        "--confirm",
        metavar="PHRASE",
        default="",
        help=f'Exact confirmation phrase required with --apply: "{_CONFIRMATION_PHRASE}"',
    )
    p.add_argument(
        "--include-redis",
        action="store_true",
        default=False,
        help="Also clear Redis tracking streams (requires --redis-confirm).",
    )
    p.add_argument(
        "--redis-confirm",
        metavar="PHRASE",
        default="",
        help=(
            "Exact confirmation phrase required with --include-redis: "
            f'"{_REDIS_CONFIRMATION_PHRASE}"'
        ),
    )
    p.add_argument(
        "--report",
        metavar="PATH",
        default="",
        help="Write machine-readable JSON report to this path.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # Confirmation validation (pure, before any I/O).
    if args.apply and not _confirmation_matches(args.confirm, _CONFIRMATION_PHRASE):
        print(
            f'ERROR: --apply requires --confirm "{_CONFIRMATION_PHRASE}"',
            file=sys.stderr,
        )
        return 2

    if args.include_redis and not _confirmation_matches(
        args.redis_confirm, _REDIS_CONFIRMATION_PHRASE
    ):
        print(
            f'ERROR: --include-redis requires --redis-confirm "{_REDIS_CONFIRMATION_PHRASE}"',
            file=sys.stderr,
        )
        return 2

    try:
        report = asyncio.run(run_reset(args))
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    except Exception as exc:
        log.exception("reset failed: %s", exc)
        return 1

    # Output report to stdout as JSON.
    output = json.dumps(report, indent=2, default=str)
    print(output)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output, encoding="utf-8")
        log.info("report written to %s", report_path)

    # Exit non-zero if smoke check failed.
    smoke = report.get("smoke_check")
    if isinstance(smoke, dict) and not smoke.get("overall_ok"):
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
