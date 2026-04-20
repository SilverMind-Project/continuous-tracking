#!/usr/bin/env sh
set -eu

offending=$(git diff --cached --name-only | xargs grep -nE '^[[:space:]]*go (run|install|build|test)' 2>/dev/null || true)
if [ -n "${offending}" ]; then
    printf 'error: raw "go" invocation found; use the Makefile targets:\n%s\n' "${offending}" >&2
    exit 1
fi
