#!/usr/bin/env bash
# Development identity data reset -- thin shell guard.
#
# Enforces environment constraints before delegating to the typed Python
# implementation. Never run automatically on service startup or migration.
#
# Usage:
#   CTS_ENV=development ./scripts/reset-identity-dev-data.sh [PYTHON_ARGS...]
#
# Example (dry run, always safe):
#   ./scripts/reset-identity-dev-data.sh
#
# Example (apply with confirmation):
#   ./scripts/reset-identity-dev-data.sh \
#       --apply --confirm "RESET DEVELOPMENT IDENTITY DATA" \
#       --report /tmp/reset-report.json
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment guards -- abort early before any DB connection.
# ---------------------------------------------------------------------------
: "${CTS_ENV:?refusing to run without CTS_ENV set}"
if [[ "${CTS_ENV}" != "development" ]]; then
    echo "ERROR: reset is development-only (CTS_ENV=${CTS_ENV})" >&2
    exit 1
fi

# Require the CTS orchestrator container to be running so the script isn't
# accidentally executed against a stale environment.
if ! docker ps --format '{{.Names}}' | grep -qx "cts-orchestrator"; then
    echo "ERROR: cts-orchestrator container is not running (docker ps shows no match)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Delegate to the typed Python implementation.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/../tracking-orchestrator/scripts/reset_identity_dev_data.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python implementation not found at ${PYTHON_SCRIPT}" >&2
    exit 1
fi

# Use the project venv Python if available.
VENV_PYTHON="${SCRIPT_DIR}/../tracking-orchestrator/.venv/bin/python"
if [[ -x "${VENV_PYTHON}" ]]; then
    PYTHON="${VENV_PYTHON}"
else
    PYTHON="${PYTHON:-python3}"
fi

exec "${PYTHON}" "${PYTHON_SCRIPT}" "$@"
