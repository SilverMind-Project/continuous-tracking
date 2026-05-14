#!/usr/bin/env bash
# CR-1: Env-var drift check between docker-compose.yml and settings.yaml.
#
# Every env var referenced in tracking-orchestrator/config/settings.yaml with
# ${VAR} syntax (no default) MUST be matched by an environment: entry in
# docker-compose.yml for the tracking-orchestrator service.  Vars with ${VAR:-default}
# are optional and only generate a warning when missing.
#
# Exit 0 on success, 1 on drift.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
SETTINGS_FILE="$PROJECT_DIR/tracking-orchestrator/config/settings.yaml"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

fail() { echo -e "${RED}ERROR:${NC} $*" >&2; }
warn() { echo -e "${YELLOW}WARN:${NC} $*" >&2; }

if [ ! -f "$COMPOSE_FILE" ] || [ ! -f "$SETTINGS_FILE" ]; then
    echo "Skipping env-var drift check: required files missing."
    exit 0
fi

# Extract all env var names from settings.yaml (not comments).
ALL_SETTINGS_VARS=$(grep -v '^[[:space:]]*#' "$SETTINGS_FILE" 2>/dev/null \
    | grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*(:[-+?][^}]*)?\}' 2>/dev/null \
    | sed 's/\${//;s/[:].*//;s/}//' \
    | sort -u || true)

# Extract required vars (no default). These match ${VAR} but not ${VAR:-...}
REQUIRED_VARS=$(grep -v '^[[:space:]]*#' "$SETTINGS_FILE" 2>/dev/null \
    | grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' 2>/dev/null \
    | grep -v ':-' 2>/dev/null \
    | sed 's/\${//;s/}//' \
    | sort -u || true)

# Collect env var names from the tracking-orchestrator section of docker-compose.
COMPOSE_VARS=$(sed -n '/^  tracking-orchestrator:/,/^  [a-z]/p' "$COMPOSE_FILE" 2>/dev/null \
    | grep -oE '^\s+[A-Z][A-Za-z0-9_]*:' 2>/dev/null \
    | sed 's/^\s*//;s/://' \
    | sort -u || true)

# Also collect interpolated vars from the whole compose file.
INTERP_VARS=$(grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*[:}]' "$COMPOSE_FILE" 2>/dev/null \
    | sed 's/\${//;s/[:}]//' \
    | sort -u || true)

COMBINED_COMPOSE=$( (echo "$COMPOSE_VARS"; echo "$INTERP_VARS") | sort -u)

DRIFT=0

# Check required vars.
if [ -n "$REQUIRED_VARS" ] && [ -n "$COMBINED_COMPOSE" ]; then
    for var in $REQUIRED_VARS; do
        if ! echo "$COMBINED_COMPOSE" | grep -qx "$var"; then
            fail "$var — required by settings.yaml (no default), not in docker-compose.yml"
            DRIFT=1
        fi
    done
fi

# Warn about optional vars (have defaults) that are missing.
if [ -n "$ALL_SETTINGS_VARS" ] && [ -n "$COMBINED_COMPOSE" ]; then
    for var in $ALL_SETTINGS_VARS; do
        if echo "$REQUIRED_VARS" | grep -qx "$var" 2>/dev/null; then
            continue
        fi
        if ! echo "$COMBINED_COMPOSE" | grep -qx "$var"; then
            warn "$var — optional (has default in settings.yaml)"
        fi
    done
fi

if [ "$DRIFT" -eq 0 ]; then
    echo -e "${GREEN}No env var drift detected.${NC}"
else
    echo ""
    echo "Run this check locally: scripts/check-env-var-drift.sh"
fi

exit $DRIFT
