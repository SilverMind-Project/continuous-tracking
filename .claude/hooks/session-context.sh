#!/usr/bin/env bash
# continuous-tracking: SessionStart context (startup and after compaction), at most 10 lines (EG-M06).
set -u
cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" 2>/dev/null || exit 0
branch=$(git branch --show-current 2>/dev/null)
dirty=$(git status --short 2>/dev/null | wc -l)
echo "[continuous-tracking] branch ${branch:-?}, ${dirty// /} files changed or untracked."
echo "Gates: make check (fast); make ci (authoritative: check-all plus every integration-marked test)."
echo "Proto changes are two-repo changes: make proto regenerates CTS and CC bindings; commit both."
echo "No test may read DATABASE_URL; integration tests use the session testcontainer."
echo "Guards: .claude/hooks/ (project) and ~/.claude/hooks/guard-machine.sh (machine). Override with ! if intended."
