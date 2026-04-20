#!/usr/bin/env sh

if [ -n "${CTS_REPO_ROOT:-}" ]; then
    REPO_ROOT="${CTS_REPO_ROOT}"
elif [ -f ".tool-versions" ]; then
    REPO_ROOT="$(pwd)"
elif [ -f "../.tool-versions" ]; then
    REPO_ROOT="$(cd .. && pwd)"
else
    SCRIPT_PATH="${BASH_SOURCE:-$0}"
    REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
fi
GO_VERSION="$(awk '/^golang / {print $2}' "${REPO_ROOT}/.tool-versions")"
ARCH="$(uname -m)"

case "${ARCH}" in
    aarch64|arm64) GOARCH=arm64 ;;
    x86_64) GOARCH=amd64 ;;
    *)
        echo "unsupported arch ${ARCH}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

export GOROOT="${REPO_ROOT}/tools/go-${GO_VERSION}-${GOARCH}"
export GOBIN="${REPO_ROOT}/tools/go-bin"
export GOMODCACHE="${REPO_ROOT}/tools/go-mod-cache"
export GOCACHE="${REPO_ROOT}/tools/go-build-cache"
export GOTOOLCHAIN="local"
export PATH="${GOROOT}/bin:${GOBIN}:${PATH}"
