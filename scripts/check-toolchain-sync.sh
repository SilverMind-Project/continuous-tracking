#!/usr/bin/env bash
# Verify that Go, Python, and pinned Go developer tool versions stay aligned across
# .tool-versions, Dockerfiles, go.mod, pyproject.toml, Makefile, and GitHub Actions CI.
# Prevents single-file bumps (such as Dependabot Dockerfile PRs) from drifting the toolchain.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAIL=0

check_eq() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" != "$actual" ]; then
    echo "TOOLCHAIN DRIFT [$label]: expected '$expected', got '$actual'" >&2
    FAIL=1
  fi
}

# 1. Go toolchain version from .tool-versions
GO_VER="$(awk '/^golang / {print $2}' "$ROOT/.tool-versions")"
GO_MINOR="${GO_VER%.*}.0"

DOCKER_GO="$(grep -oE 'golang:[0-9]+\.[0-9]+\.[0-9]+' "$ROOT/rtsp-ingress/Dockerfile" | head -n1 | cut -d: -f2)"
check_eq "rtsp-ingress/Dockerfile golang" "$GO_VER" "$DOCKER_GO"

GOMOD_TOOLCHAIN="$(awk '/^toolchain go/ {sub(/^go/, "", $2); print $2}' "$ROOT/rtsp-ingress/go.mod")"
check_eq "rtsp-ingress/go.mod toolchain" "$GO_VER" "$GOMOD_TOOLCHAIN"

GOMOD_RTSP_GO="$(awk '/^go / {print $2}' "$ROOT/rtsp-ingress/go.mod")"
check_eq "rtsp-ingress/go.mod go directive" "$GO_MINOR" "$GOMOD_RTSP_GO"

GOMOD_PROTO_GO="$(awk '/^go / {print $2}' "$ROOT/proto/go.mod")"
check_eq "proto/go.mod go directive" "$GO_MINOR" "$GOMOD_PROTO_GO"

while IFS= read -r ci_go; do
  check_eq ".github/workflows/ci.yml go-version" "$GO_VER" "$ci_go"
done < <(grep -oE 'go-version: "[0-9]+\.[0-9]+\.[0-9]+"' "$ROOT/.github/workflows/ci.yml" | sed -E 's/.*"([^"]+)"/\1/')

# 2. Python major.minor from .tool-versions
PY_FULL="$(awk '/^python / {print $2}' "$ROOT/.tool-versions")"
PY_MINOR="${PY_FULL%.*}"

DOCKER_PY="$(grep -oE 'FROM python:[0-9]+\.[0-9]+' "$ROOT/tracking-orchestrator/Dockerfile" | head -n1 | cut -d: -f2)"
check_eq "tracking-orchestrator/Dockerfile python" "$PY_MINOR" "$DOCKER_PY"

MYPY_PY="$(grep -oE 'python_version = "[0-9]+\.[0-9]+"' "$ROOT/tracking-orchestrator/pyproject.toml" | head -n1 | sed -E 's/.*"([^"]+)"/\1/')"
check_eq "tracking-orchestrator/pyproject.toml mypy python_version" "$PY_MINOR" "$MYPY_PY"

while IFS= read -r ci_py; do
  check_eq ".github/workflows/ci.yml python-version" "$PY_MINOR" "$ci_py"
done < <(grep -oE 'python-version: "[0-9]+\.[0-9]+"' "$ROOT/.github/workflows/ci.yml" | sed -E 's/.*"([^"]+)"/\1/')

# 3. google.golang.org/protobuf runtime & protoc-gen-go alignment
PROTO_MOD_PB="$(awk '/google\.golang\.org\/protobuf / {print $NF}' "$ROOT/proto/go.mod" | head -n1)"
RTSP_MOD_PB="$(awk '/google\.golang\.org\/protobuf / {print $NF}' "$ROOT/rtsp-ingress/go.mod" | head -n1)"
MAKE_PGGO="$(grep -oE 'protoc-gen-go@v[0-9]+\.[0-9]+\.[0-9]+' "$ROOT/Makefile" | head -n1 | cut -d@ -f2)"
CI_PGGO="$(grep -oE 'protoc-gen-go@v[0-9]+\.[0-9]+\.[0-9]+' "$ROOT/.github/workflows/ci.yml" | head -n1 | cut -d@ -f2)"

check_eq "rtsp-ingress/go.mod google.golang.org/protobuf" "$PROTO_MOD_PB" "$RTSP_MOD_PB"
check_eq "Makefile protoc-gen-go" "$PROTO_MOD_PB" "$MAKE_PGGO"
check_eq ".github/workflows/ci.yml protoc-gen-go" "$PROTO_MOD_PB" "$CI_PGGO"

# 4. golangci-lint and buf pins between Makefile and CI
MAKE_GL="$(grep -oE 'golangci-lint@v[0-9]+\.[0-9]+\.[0-9]+' "$ROOT/Makefile" | head -n1 | cut -d@ -f2)"
CI_GL="$(grep -A4 'golangci-lint-action@' "$ROOT/.github/workflows/ci.yml" | grep -oE 'version: v[0-9]+\.[0-9]+\.[0-9]+' | head -n1 | awk '{print $2}')"
check_eq "golangci-lint version (Makefile vs ci.yml)" "$MAKE_GL" "$CI_GL"

MAKE_BUF="$(grep -oE 'buf@v[0-9]+\.[0-9]+\.[0-9]+' "$ROOT/Makefile" | head -n1 | sed 's/buf@v//')"
CI_BUF="$(grep -A3 'buf-setup-action@' "$ROOT/.github/workflows/ci.yml" | grep -oE 'version: "[0-9]+\.[0-9]+\.[0-9]+"' | head -n1 | sed -E 's/.*"([^"]+)"/\1/')"
check_eq "buf version (Makefile vs ci.yml)" "$MAKE_BUF" "$CI_BUF"

if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
echo "Toolchain pins aligned (Go ${GO_VER}, Python ${PY_MINOR}, protobuf ${PROTO_MOD_PB}, golangci-lint ${MAKE_GL}, buf ${MAKE_BUF})."
