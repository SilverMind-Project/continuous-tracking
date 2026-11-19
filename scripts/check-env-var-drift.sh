#!/usr/bin/env bash
# CR-1: Env-var drift check between docker-compose.yml and settings.yaml.
#
# Every env var referenced in tracking-orchestrator/config/settings.yaml with
# ${VAR} syntax MUST be matched by an environment: entry in docker-compose.yml
# for the tracking-orchestrator service. Defaulted env interpolation
# is intentionally unsupported so runtime tunables stay literal in settings.yaml.
#
# Exit 0 on success, 1 on drift.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

fail() { echo -e "${RED}ERROR:${NC} $*" >&2; }

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Skipping env-var drift check: docker-compose.yml missing."
    exit 0
fi

# Collect interpolated vars from the whole compose file.
INTERP_VARS=$(grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*[:}]' "$COMPOSE_FILE" 2>/dev/null \
    | sed 's/\${//;s/[:}]//' \
    | sort -u || true)

DRIFT=0

check_service() {
    local service_name="$1"
    local settings_file="$2"

    if [ ! -f "$settings_file" ]; then
        return 0
    fi

    local required_vars defaulted_vars compose_vars combined_compose
    required_vars=$(grep -v '^[[:space:]]*#' "$settings_file" 2>/dev/null \
        | grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' 2>/dev/null \
        | grep -v ':-' 2>/dev/null \
        | sed 's/\${//;s/}//' \
        | sort -u || true)

    defaulted_vars=$(grep -v '^[[:space:]]*#' "$settings_file" 2>/dev/null \
        | grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*:-[^}]*\}' 2>/dev/null \
        | sort -u || true)

    compose_vars=$(sed -n "/^  ${service_name}:/,/^  [a-z]/p" "$COMPOSE_FILE" 2>/dev/null \
        | grep -oE '^\s+[A-Z][A-Za-z0-9_]*:' 2>/dev/null \
        | sed 's/^\s*//;s/://' \
        | sort -u || true)

    combined_compose=$( (echo "$compose_vars"; echo "$INTERP_VARS") | sort -u)

    if [ -n "$defaulted_vars" ]; then
        fail "${service_name}: defaulted env interpolation is disabled in settings.yaml: $defaulted_vars"
        DRIFT=1
    fi

    if [ -n "$required_vars" ] && [ -n "$combined_compose" ]; then
        for var in $required_vars; do
            if ! echo "$combined_compose" | grep -qx "$var"; then
                fail "${service_name}: $var required by settings.yaml (no default), not in docker-compose.yml"
                DRIFT=1
            fi
        done
    fi
}

check_service "tracking-orchestrator" "$PROJECT_DIR/tracking-orchestrator/config/settings.yaml"
check_service "rf-ingress" "$PROJECT_DIR/rf-ingress/config/settings.yaml"

if [ "$DRIFT" -eq 0 ]; then
    echo -e "${GREEN}No env var drift detected.${NC}"
else
    echo ""
    echo "Run this check locally: scripts/check-env-var-drift.sh"
fi

exit $DRIFT
