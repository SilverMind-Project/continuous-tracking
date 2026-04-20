#!/usr/bin/env sh
set -eu

expected_venv="$(cd "$(dirname "$0")/.." && pwd)/tracking-orchestrator/.venv"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "error: VIRTUAL_ENV is not set. Activate the venv first:" >&2
    echo "    source ${expected_venv}/bin/activate" >&2
    echo "  or run the command via 'uv run ...' from tracking-orchestrator/." >&2
    exit 1
fi

case "${VIRTUAL_ENV}" in
    "${expected_venv}"*) : ;;
    *)
        echo "error: VIRTUAL_ENV=${VIRTUAL_ENV} does not match expected ${expected_venv}" >&2
        exit 1
        ;;
esac

py="${VIRTUAL_ENV}/bin/python"
if [ ! -x "${py}" ]; then
    echo "error: ${py} is missing or not executable" >&2
    exit 1
fi

ver="$(${py} -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "${ver}" != "3.12" ]; then
    echo "error: expected Python 3.12, got ${ver}" >&2
    exit 1
fi
