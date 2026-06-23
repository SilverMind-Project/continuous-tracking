"""Tests for the development identity data reset script.

All tests that verify pure logic run without any database or Docker dependency.
Integration tests (live DB, idempotency, preserved-count equivalence) require
the integration marker and are skipped by default.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Import the module under test.  The script lives in scripts/ and is not part
# of the app package, so we add the parent to sys.path in conftest.py or here.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import reset_identity_dev_data as rsd  # noqa: E402

# ---------------------------------------------------------------------------
# Pure-function tests (no I/O).
# ---------------------------------------------------------------------------


class TestConfirmationPhrase:
    def test_correct_phrase_accepted(self) -> None:
        assert rsd._confirmation_matches(
            "RESET DEVELOPMENT IDENTITY DATA",
            rsd._CONFIRMATION_PHRASE,
        )

    def test_wrong_phrase_rejected(self) -> None:
        assert not rsd._confirmation_matches("wrong phrase", rsd._CONFIRMATION_PHRASE)

    def test_empty_phrase_rejected(self) -> None:
        assert not rsd._confirmation_matches("", rsd._CONFIRMATION_PHRASE)

    def test_partial_phrase_rejected(self) -> None:
        assert not rsd._confirmation_matches("RESET DEVELOPMENT", rsd._CONFIRMATION_PHRASE)

    def test_extra_whitespace_rejected(self) -> None:
        assert not rsd._confirmation_matches(
            "RESET DEVELOPMENT IDENTITY DATA ",
            rsd._CONFIRMATION_PHRASE,
        )

    def test_redis_phrase_correct(self) -> None:
        assert rsd._confirmation_matches(
            "RESET REDIS STREAMS",
            rsd._REDIS_CONFIRMATION_PHRASE,
        )


class TestAllowlists:
    def test_identities_not_in_delete_list(self) -> None:
        assert "identities" not in rsd._CTS_DELETE_TABLES

    def test_cameras_not_in_delete_list(self) -> None:
        assert "cameras" not in rsd._CTS_DELETE_TABLES

    def test_streams_not_in_delete_list(self) -> None:
        assert "streams" not in rsd._CTS_DELETE_TABLES

    def test_camera_topology_edges_not_in_delete_list(self) -> None:
        assert "camera_topology_edges" not in rsd._CTS_DELETE_TABLES

    def test_no_overlap_between_delete_and_preserve(self) -> None:
        overlap = set(rsd._CTS_DELETE_TABLES) & rsd._CTS_PRESERVE_TABLES
        assert not overlap, f"delete/preserve overlap: {overlap}"

    def test_all_expected_cts_tables_present(self) -> None:
        # Verified set from M11 spec implementation-reality table.
        expected = {
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
        }
        assert set(rsd._CTS_DELETE_TABLES) == expected

    def test_all_expected_cc_tables_present(self) -> None:
        expected = {
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
        }
        assert set(rsd._CC_DELETE_TABLES) == expected

    def test_reid_prefixes_only_dedicated_crops(self) -> None:
        for prefix in rsd._REID_PREFIXES:
            assert not prefix.startswith("frames/"), (
                f"prefix {prefix!r} would delete raw frame objects -- forbidden"
            )
        assert "reid-candidates/" in rsd._REID_PREFIXES
        assert "reid-candidates-frames/" in rsd._REID_PREFIXES


class TestBuildTruncateSql:
    def test_cts_sql_schema_qualified(self) -> None:
        sql = rsd._build_cts_truncate_sql(("person_hypotheses", "reid_gallery"))
        assert "continuous_tracking.person_hypotheses" in sql
        assert "continuous_tracking.reid_gallery" in sql
        assert "RESTART IDENTITY CASCADE" in sql
        assert "TRUNCATE" in sql

    def test_cc_sql_unqualified(self) -> None:
        sql = rsd._build_cc_truncate_sql(("person_location_history",))
        assert "person_location_history" in sql
        assert "continuous_tracking." not in sql
        assert "RESTART IDENTITY CASCADE" in sql

    def test_cts_sql_contains_all_delete_tables(self) -> None:
        sql = rsd._build_cts_truncate_sql(rsd._CTS_DELETE_TABLES)
        for table in rsd._CTS_DELETE_TABLES:
            assert f"continuous_tracking.{table}" in sql


class TestRedactUrl:
    def test_credentials_removed(self) -> None:
        url = "postgresql://user:s3cr3t@postgres:5432/mydb"
        redacted = rsd._redact_url(url)
        assert "s3cr3t" not in redacted
        assert "***" in redacted
        assert "postgres:5432/mydb" in redacted

    def test_url_without_credentials_unchanged(self) -> None:
        url = "postgresql://postgres:5432/mydb"
        redacted = rsd._redact_url(url)
        # No credentials present, URL shape preserved.
        assert "postgres:5432/mydb" in redacted


class TestViolationCheck:
    def test_empty_violations_when_all_in_allowlist(self) -> None:
        allowed = {"table_a", "table_b"}
        assert rsd._check_violations(["table_a", "table_b"], allowed) == []

    def test_violation_when_table_not_in_allowlist(self) -> None:
        allowed = {"table_a"}
        violations = rsd._check_violations(["table_a", "table_x"], allowed)
        assert "table_x" in violations

    def test_multiple_violations_reported(self) -> None:
        allowed = {"table_a"}
        violations = rsd._check_violations(["table_x", "table_y"], allowed)
        assert set(violations) == {"table_x", "table_y"}


# ---------------------------------------------------------------------------
# Argument parsing tests.
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_default_is_dry_run(self) -> None:
        args = rsd._parse_args([])
        assert not args.apply

    def test_apply_flag(self) -> None:
        args = rsd._parse_args(["--apply", "--confirm", "RESET DEVELOPMENT IDENTITY DATA"])
        assert args.apply
        assert args.confirm == "RESET DEVELOPMENT IDENTITY DATA"

    def test_include_redis_requires_separate_flag(self) -> None:
        args = rsd._parse_args(
            [
                "--apply",
                "--confirm",
                "RESET DEVELOPMENT IDENTITY DATA",
                "--include-redis",
                "--redis-confirm",
                "RESET REDIS STREAMS",
            ]
        )
        assert args.include_redis
        assert args.redis_confirm == "RESET REDIS STREAMS"

    def test_report_path_parsed(self, tmp_path: Path) -> None:
        report_path = str(tmp_path / "report.json")
        args = rsd._parse_args(["--report", report_path])
        assert args.report == report_path


class TestMainConfirmationGuard:
    """main() must abort if --apply is set without the exact phrase."""

    def test_apply_without_confirm_returns_exit_2(self) -> None:
        result = rsd.main(["--apply"])
        assert result == 2

    def test_apply_with_wrong_confirm_returns_exit_2(self) -> None:
        result = rsd.main(["--apply", "--confirm", "wrong"])
        assert result == 2

    def test_redis_without_redis_confirm_returns_exit_2(self) -> None:
        result = rsd.main(
            [
                "--apply",
                "--confirm",
                "RESET DEVELOPMENT IDENTITY DATA",
                "--include-redis",
                "--redis-confirm",
                "WRONG PHRASE",
            ]
        )
        assert result == 2


# ---------------------------------------------------------------------------
# Dry-run smoke: no writes happen without --apply.
# These tests check the argument flow only; actual DB connections are skipped
# because DATABASE_URL is not set in the unit-test environment.
# ---------------------------------------------------------------------------


class TestDryRunNoWrites:
    def test_dry_run_has_no_apply(self) -> None:
        args = rsd._parse_args([])
        assert not args.apply, "dry run must be the default"

    def test_cts_truncate_sql_not_run_on_dry_run(self) -> None:
        """Verify the SQL produced by the dry-run path contains no DML side-effects.

        We cannot run the SQL without a DB, but we can confirm the builder
        produces schema-qualified TRUNCATE (not DELETE FROM which is harder to roll back).
        """
        sql = rsd._build_cts_truncate_sql(rsd._CTS_DELETE_TABLES)
        assert "TRUNCATE" in sql
        assert "DELETE FROM" not in sql


# ---------------------------------------------------------------------------
# MinIO key filter tests -- raw frames must never be touched.
# ---------------------------------------------------------------------------


class TestMinioKeyFilter:
    def test_only_reid_prefixes_are_allowed(self) -> None:
        # Enumerate what _REID_PREFIXES contains and assert frames/ is absent.
        for prefix in rsd._REID_PREFIXES:
            assert not prefix.startswith("frames/")
            assert not prefix.startswith("keyframes/")

    def test_reid_candidates_prefix_present(self) -> None:
        assert "reid-candidates/" in rsd._REID_PREFIXES

    def test_reid_candidates_frames_prefix_present(self) -> None:
        assert "reid-candidates-frames/" in rsd._REID_PREFIXES


class TestMinioSecureEnv:
    """The MinIO secure flag must come from the canonical MINIO_SECURE var."""

    def test_minio_secure_var_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIO_ENDPOINT_URL", "http://minio:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "ak")
        monkeypatch.setenv("MINIO_SECRET_KEY", "sk")
        monkeypatch.setenv("MINIO_BUCKET", "ai-media")
        monkeypatch.setenv("MINIO_SECURE", "true")
        monkeypatch.delenv("MINIO_USE_SSL", raising=False)
        *_, use_ssl = rsd._minio_config_from_env()
        assert use_ssl is True

    def test_minio_use_ssl_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIO_ENDPOINT_URL", "http://minio:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "ak")
        monkeypatch.setenv("MINIO_SECRET_KEY", "sk")
        monkeypatch.setenv("MINIO_BUCKET", "ai-media")
        monkeypatch.delenv("MINIO_SECURE", raising=False)
        monkeypatch.setenv("MINIO_USE_SSL", "true")
        *_, use_ssl = rsd._minio_config_from_env()
        assert use_ssl is True


class TestPartialMinioFailure:
    """A per-object delete failure is counted and reported, never fatal.

    Spec: 'Partial MinIO failure is reported/retryable without corrupting audit
    state.' The DB reset has already committed by the time MinIO runs, so a crop
    delete that raises must be tallied as failed and the run must continue.
    """

    @pytest.mark.asyncio
    async def test_failed_delete_is_counted_not_raised(self) -> None:
        class FlakyFetcher:
            def __init__(self) -> None:
                self.deleted: list[str] = []

            async def delete_object(self, key: str) -> None:
                if key.endswith("bad.jpg"):
                    raise RuntimeError("simulated MinIO 500")
                self.deleted.append(key)

        fetcher = FlakyFetcher()
        prefix_keys = {
            "reid-candidates/": [
                "reid-candidates/v1/ok1.jpg",
                "reid-candidates/v1/bad.jpg",
                "reid-candidates/v1/ok2.jpg",
            ],
        }
        report = await rsd._delete_reid_objects(fetcher, prefix_keys, dry_run=False)
        bucket = report["reid-candidates/"]
        assert bucket == {"count": 3, "deleted": 2, "failed": 1}
        # The two good keys were actually deleted; the failure did not abort the loop.
        assert fetcher.deleted == [
            "reid-candidates/v1/ok1.jpg",
            "reid-candidates/v1/ok2.jpg",
        ]

    @pytest.mark.asyncio
    async def test_dry_run_deletes_nothing(self) -> None:
        class CountingFetcher:
            def __init__(self) -> None:
                self.calls = 0

            async def delete_object(self, key: str) -> None:
                self.calls += 1

        fetcher = CountingFetcher()
        prefix_keys = {"reid-candidates/": ["reid-candidates/v1/a.jpg"]}
        report = await rsd._delete_reid_objects(fetcher, prefix_keys, dry_run=True)
        assert fetcher.calls == 0
        assert report["reid-candidates/"] == {"count": 1, "deleted": 0, "failed": 0}


class TestRawFrameSurvival:
    """The reset must never empty frames/; the survival check proves it.

    Spec: 'Only dedicated ReID crop prefix is deleted; raw frames/... remains.'
    """

    class _FakeFetcher:
        def __init__(self, keys_by_prefix: dict[str, list[str]]) -> None:
            self._by_prefix = keys_by_prefix

        async def list_objects_by_prefix(self, prefix: str) -> list[str]:
            return self._by_prefix.get(prefix, [])

    @pytest.mark.asyncio
    async def test_sampled_key_present_is_ok(self) -> None:
        key = "frames/cam01/2026/06/23/00/abc.jpg"
        fetcher = self._FakeFetcher({key: [key]})
        result = await rsd._check_raw_frame_survival(fetcher, key)
        assert result["exists"] is True
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_sampled_key_absent_but_frames_present_is_ok(self) -> None:
        """Retention may have removed the sampled object; frames/ as a class survives."""
        key = "frames/cam01/2026/06/17/20/stale.jpg"
        fetcher = self._FakeFetcher({"frames/": ["frames/cam01/2026/06/23/00/fresh.jpg"]})
        result = await rsd._check_raw_frame_survival(fetcher, key)
        assert result["exists"] is False
        assert result["frames_prefix_nonempty"] is True
        assert result["ok"] is True
        assert "retention" in result["note"]

    @pytest.mark.asyncio
    async def test_sampled_key_absent_and_frames_empty_fails(self) -> None:
        """If frames/ is empty too, the check must fail -- it is not always-true theater."""
        key = "frames/cam01/2026/06/17/20/stale.jpg"
        fetcher = self._FakeFetcher({})
        result = await rsd._check_raw_frame_survival(fetcher, key)
        assert result["exists"] is False
        assert result["frames_prefix_nonempty"] is False
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Shell environment-guard test (production / non-development always aborts).
#
# The CTS_ENV guard lives in the shell wrapper, not Python, so it is exercised
# by invoking the script as a subprocess. It must abort before any DB work.
# ---------------------------------------------------------------------------


class TestShellEnvGuard:
    _WRAPPER = Path(__file__).parent.parent.parent / "scripts" / "reset-identity-dev-data.sh"

    def test_wrapper_exists(self) -> None:
        assert self._WRAPPER.is_file(), f"shell wrapper missing at {self._WRAPPER}"

    def test_missing_cts_env_aborts(self) -> None:
        import subprocess

        env = {k: v for k, v in os.environ.items() if k != "CTS_ENV"}
        result = subprocess.run(
            ["bash", str(self._WRAPPER)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "CTS_ENV" in result.stderr

    def test_non_development_cts_env_aborts(self) -> None:
        import subprocess

        env = {**os.environ, "CTS_ENV": "production"}
        result = subprocess.run(
            ["bash", str(self._WRAPPER)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "development-only" in result.stderr


# ---------------------------------------------------------------------------
# Orphan-FK integration tests (require live CTS DB; skipped by default).
#
# These tests exercise _check_cts_orphan_fks against the real
# information_schema so we prove the SQL itself is correct, not just the
# pure _check_violations helper.
#
# Run with:
#   DATABASE_URL=postgresql://cts_user:change-me-cts-password@localhost:5432/continuous_tracking \
#   pytest tests/test_reset_identity_dev_data.py -m integration -v
# ---------------------------------------------------------------------------

_DB_URL_PRESENT = bool(os.environ.get("DATABASE_URL"))


@pytest.mark.integration
@pytest.mark.skipif(
    not _DB_URL_PRESENT,
    reason="DATABASE_URL not set; skipped outside dev stack",
)
class TestOrphanFkIntegration:
    """Orphan-FK pre-condition check against the real continuous_tracking schema."""

    @pytest.mark.asyncio
    async def test_full_allowlist_produces_no_violations(self) -> None:
        """Canonical delete list must pass the orphan-FK pre-condition with zero violations."""
        import asyncpg

        dsn = await rsd._cts_dsn_from_env()
        conn = await asyncpg.connect(dsn)
        try:
            violations = await rsd._check_cts_orphan_fks(conn, rsd._CTS_DELETE_TABLES)
            assert violations == [], (
                f"orphan-FK violations with canonical delete list: {violations}"
            )
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_existing_tables_partitions_allowlist(self) -> None:
        """_existing_tables must split the allowlist into present/missing with no loss.

        Resilience guarantee: a dev DB lagging the migration head still produces a
        clean (present, missing) partition whose union is the full allowlist, so the
        reset operates only over real tables instead of crashing on a missing one.
        Preserve tables must all be present regardless of head.
        """
        import asyncpg

        dsn = await rsd._cts_dsn_from_env()
        conn = await asyncpg.connect(dsn)
        try:
            present, missing = await rsd._existing_tables(
                conn, "continuous_tracking", rsd._CTS_DELETE_TABLES
            )
            assert set(present) | set(missing) == set(rsd._CTS_DELETE_TABLES)
            assert set(present) & set(missing) == set()

            p_present, p_missing = await rsd._existing_tables(
                conn, "continuous_tracking", tuple(rsd._CTS_PRESERVE_TABLES)
            )
            assert p_missing == [], f"preserve tables unexpectedly absent: {p_missing}"
            assert set(p_present) == set(rsd._CTS_PRESERVE_TABLES)
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_truncated_allowlist_fires_violation(self) -> None:
        """Removing world_observations from the allowlist must fire a violation.

        Schema fact (0001_init.up.sql line 161):
          world_observations.ph_id REFERENCES person_hypotheses(ph_id) ON DELETE CASCADE

        With world_observations absent from the allowlist but person_hypotheses still
        present as a delete target, the check must report world_observations as an orphan
        (truncating person_hypotheses CASCADE would silently also truncate world_observations).
        """
        import asyncpg

        truncated = tuple(t for t in rsd._CTS_DELETE_TABLES if t != "world_observations")
        dsn = await rsd._cts_dsn_from_env()
        conn = await asyncpg.connect(dsn)
        try:
            violations = await rsd._check_cts_orphan_fks(conn, truncated)
            assert violations, (
                "expected orphan-FK violation when world_observations is omitted "
                "from delete list (it references person_hypotheses which remains a "
                "delete target)"
            )
            assert any("world_observations" in v for v in violations), (
                f"expected 'world_observations' in violation messages, got: {violations}"
            )
        finally:
            await conn.close()
