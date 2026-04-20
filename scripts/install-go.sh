#!/usr/bin/env sh
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
GO_VERSION="$(awk '/^golang / {print $2}' "${REPO_ROOT}/.tool-versions")"

if [ -z "${GO_VERSION}" ]; then
    echo "error: golang version missing from .tool-versions" >&2
    exit 1
fi

ARCH="$(uname -m)"
case "${ARCH}" in
    aarch64|arm64) GOARCH=arm64 ;;
    x86_64) GOARCH=amd64 ;;
    *)
        echo "unsupported arch ${ARCH}" >&2
        exit 1
        ;;
esac

GO_DIR="${TOOLS_DIR}/go-${GO_VERSION}-${GOARCH}"
GO_BIN="${GO_DIR}/bin/go"

if [ -x "${GO_BIN}" ]; then
    echo "go ${GO_VERSION} already installed at ${GO_DIR}"
else
    mkdir -p "${TOOLS_DIR}"
    TARBALL="go${GO_VERSION}.linux-${GOARCH}.tar.gz"
    URL="https://go.dev/dl/${TARBALL}"
    echo "downloading ${URL}"
    curl -fsSL "${URL}" -o "${TOOLS_DIR}/${TARBALL}"
    mkdir -p "${GO_DIR}"
    tar -xzf "${TOOLS_DIR}/${TARBALL}" -C "${GO_DIR}" --strip-components=1
    rm "${TOOLS_DIR}/${TARBALL}"
fi

printf 'GO_BIN=%s\n' "${GO_BIN}"
