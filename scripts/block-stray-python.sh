#!/usr/bin/env sh
set -eu

# Only the lines this commit adds. Scanning whole staged files made any document that already
# quotes a Dockerfile's raw install line impossible to edit; a newly added raw install still fails.
offending=$(git diff --cached -U0 --no-color | grep -E '^\+[^+]' | grep -nE '(^\+| )(pip|python|python3) install' || true)
if [ -n "${offending}" ]; then
    printf 'error: raw pip/python install detected in staged changes:\n%s\n' "${offending}" >&2
    printf 'Use `uv add` (or `uv sync`) from tracking-orchestrator/ instead.\n' >&2
    exit 1
fi
