#!/usr/bin/env sh
set -eu

offending=$(git diff --cached --name-only | xargs grep -nE '(^| )(pip|python|python3) install' 2>/dev/null || true)
if [ -n "${offending}" ]; then
    printf 'error: raw pip/python install detected in staged changes:\n%s\n' "${offending}" >&2
    printf 'Use `uv add` (or `uv sync`) from tracking-orchestrator/ instead.\n' >&2
    exit 1
fi
