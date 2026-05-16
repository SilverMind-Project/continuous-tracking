"""Tests for SQL statement splitting with dollar-quote awareness."""

from __future__ import annotations

from app.storage.migrations import MigrationRunner

_split = MigrationRunner._split_statements


class TestSplitStatements:
    def test_simple_statements(self) -> None:
        stmts = _split("SELECT 1; SELECT 2;")
        assert len(stmts) == 2
        assert stmts[0] == "SELECT 1"
        assert stmts[1] == "SELECT 2"

    def test_do_block_preserved(self) -> None:
        stmts = _split("DO $$\nBEGIN\n  EXECUTE 'SELECT 1';\nEND $$;\nSELECT 2;")
        assert len(stmts) == 2
        assert "DO $$" in stmts[0]
        assert "END $$" in stmts[0]
        assert "EXECUTE 'SELECT 1';" in stmts[0]
        assert stmts[1] == "SELECT 2"

    def test_named_dollar_quote(self) -> None:
        stmts = _split("$func$\nBEGIN\n  RETURN 1;\nEND $func$;\nSELECT 2;")
        assert len(stmts) == 2
        assert "$func$" in stmts[0]
        assert "RETURN 1;" in stmts[0]

    def test_multiple_do_blocks(self) -> None:
        stmts = _split("DO $$ BEGIN EXECUTE 'a'; END $$;\nDO $$ BEGIN EXECUTE 'b'; END $$;")
        assert len(stmts) == 2

    def test_comment_only_statements_dropped(self) -> None:
        # A statement that is purely a comment (no SQL) is dropped.
        assert _split("-- only a comment") == []
        assert _split("-- another comment\n-- more comment") == []

    def test_comments_alongside_sql_kept(self) -> None:
        # Comments bundled in the same statement as real SQL are preserved.
        stmts = _split("-- header comment\nSELECT 1;")
        assert len(stmts) == 1
        assert "SELECT 1" in stmts[0]

    def test_set_search_path(self) -> None:
        stmts = _split("SET search_path = continuous_tracking, public;")
        assert len(stmts) == 1
        assert "continuous_tracking" in stmts[0]

    def test_empty_sql(self) -> None:
        assert _split("") == []
        assert _split("-- just a comment") == []

    def test_no_trailing_semicolon(self) -> None:
        stmts = _split("SELECT 1")
        assert len(stmts) == 1
        assert stmts[0] == "SELECT 1"

    def test_create_materialized_view(self) -> None:
        stmts = _split(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS foo\n"
            "WITH (timescaledb.continuous) AS\n"
            "SELECT col1, col2\n"
            "FROM bar\n"
            "GROUP BY col1;"
        )
        assert len(stmts) == 1
        assert "CREATE MATERIALIZED VIEW" in stmts[0]
